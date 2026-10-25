// Sky/cloud temporal accumulation. The path tracer owns the cloud radiance inside the miss shader, so
// this pass filters the render-res image after both traces: sky pixels are reprojected with the frame's
// motion vectors and blended into a two-image history, while geometry pixels copy through untouched.
//
// The history is clamped to the current 3x3 neighbourhood's colour range, which is what keeps moving
// cloud edges and a changing sun from ghosting when the scene motion vector cannot describe the cloud
// field's own wind drift.
import world_common;

[[vk::binding(0, 0)]] Sampler2D currentColor;
[[vk::binding(1, 0)]] Sampler2D depthImage;
[[vk::binding(2, 0)]] Sampler2D motionImage;
[[vk::binding(3, 0)]] Sampler2D historyColor;
[[vk::binding(4, 0)]] RWTexture2D<float4> outputColor;
[[vk::binding(5, 0)]] RWTexture2D<float4> outputHistory;

struct CloudTemporalPush {
    float2 size;
    uint historyValid;
    uint pad;
    uint64_t worldPushAddr;
    uint64_t noiseAddr;
    float2 windDelta;
    float2 pad2;
};

[[vk::push_constant]] CloudTemporalPush push;

static const int NVDF_X = 128;
static const int NVDF_Y = 32;
static const int NVDF_Z = 128;
static const uint NVDF_BASE = 3309568u;
static const float NVDF_NOMINAL_COVERAGE = 0.5;
static const float NVDF_COVERAGE_OFFSET = 160.0;

public int nvdfWrap(int value, int size) {
    int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

public float sampleNvdf(float2 tilePosition, float height) {
    float3 uv = float3(frac(tilePosition.x) * float(NVDF_X) - 0.5,
                       height * float(NVDF_Y) - 0.5,
                       frac(tilePosition.y) * float(NVDF_Z) - 0.5);
    int3 cell0 = int3(floor(uv));
    int3 cell1 = cell0 + int3(1);
    float3 f = uv - float3(cell0);
    int3 c0 = int3(nvdfWrap(cell0.x, NVDF_X), clamp(cell0.y, 0, NVDF_Y - 1),
                   nvdfWrap(cell0.z, NVDF_Z));
    int3 c1 = int3(nvdfWrap(cell1.x, NVDF_X), clamp(cell1.y, 0, NVDF_Y - 1),
                   nvdfWrap(cell1.z, NVDF_Z));
    ConstPtr<uint> words = ConstPtr<uint>(push.noiseAddr);
    float n000 = asfloat(words[NVDF_BASE
            + (uint(c0.z) * uint(NVDF_Y) + uint(c0.y)) * uint(NVDF_X) + uint(c0.x)]);
    float n100 = asfloat(words[NVDF_BASE
            + (uint(c0.z) * uint(NVDF_Y) + uint(c0.y)) * uint(NVDF_X) + uint(c1.x)]);
    float n010 = asfloat(words[NVDF_BASE
            + (uint(c0.z) * uint(NVDF_Y) + uint(c1.y)) * uint(NVDF_X) + uint(c0.x)]);
    float n110 = asfloat(words[NVDF_BASE
            + (uint(c0.z) * uint(NVDF_Y) + uint(c1.y)) * uint(NVDF_X) + uint(c1.x)]);
    float n001 = asfloat(words[NVDF_BASE
            + (uint(c1.z) * uint(NVDF_Y) + uint(c0.y)) * uint(NVDF_X) + uint(c0.x)]);
    float n101 = asfloat(words[NVDF_BASE
            + (uint(c1.z) * uint(NVDF_Y) + uint(c0.y)) * uint(NVDF_X) + uint(c1.x)]);
    float n011 = asfloat(words[NVDF_BASE
            + (uint(c1.z) * uint(NVDF_Y) + uint(c1.y)) * uint(NVDF_X) + uint(c0.x)]);
    float n111 = asfloat(words[NVDF_BASE
            + (uint(c1.z) * uint(NVDF_Y) + uint(c1.y)) * uint(NVDF_X) + uint(c1.x)]);
    float n00 = lerp(n000, n100, f.x);
    float n10 = lerp(n010, n110, f.x);
    float n01 = lerp(n001, n101, f.x);
    float n11 = lerp(n011, n111, f.x);
    return lerp(lerp(n00, n10, f.y), lerp(n01, n11, f.y), f.z);
}

// First cloud-body hit along the camera ray, in blocks. The SDF is enough here: this is only a
// reprojection depth, so the detail erosion would move the history by a sub-pixel amount.
public float cloudDepth(WorldPush world, float3 origin, float3 direction) {
    float base = world.cloudParams0.x;
    float thickness = world.cloudParams0.y;
    if (abs(direction.y) < 1.0e-4) {
        return -1.0;
    }
    float lower = (base - origin.y) / direction.y;
    float upper = (base + thickness - origin.y) / direction.y;
    float tEnter = max(min(lower, upper), 0.0);
    float tExit = max(lower, upper);
    if (tExit <= tEnter) {
        return -1.0;
    }
    const int steps = 8;
    float stepLength = (tExit - tEnter) / float(steps);
    float coverage = world.cloudParams0.z;
    float nominal = NVDF_NOMINAL_COVERAGE;
    for (int i = 0; i < steps; i++) {
        float t = tEnter + (float(i) + 0.5) * stepLength;
        float3 position = origin + direction * t;
        float height = (position.y - base) / thickness;
        float2 tile = frac(float2(position.x + world.cloudParams1.x,
                                  position.z + world.cloudParams1.y) / 2048.0);
        float sdf = sampleNvdf(tile, saturate(height));
        if (sdf - (coverage - nominal) * NVDF_COVERAGE_OFFSET < 0.0) {
            return t;
        }
    }
    return -1.0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    uint2 pixel = threadId.xy;
    uint2 dimensions = uint2(push.size);
    if (any(pixel >= dimensions)) {
        return;
    }

    float2 uv = (float2(pixel) + 0.5) / push.size;
    float3 current = currentColor.SampleLevel(uv, 0.0).rgb;
    float depth = depthImage.SampleLevel(uv, 0.0).r;
    float2 motion = motionImage.SampleLevel(uv, 0.0).rg;
    float2 previousUv = uv + motion / push.size;

    // Sky motion vectors carry camera reprojection only. Recover the cloud surface depth from the body
    // SDF, then add the field's wind displacement in screen space so a wind-driven edge follows its own
    // history instead of smearing at the pixel centre.
    if (depth <= 1.0e-5 && push.historyValid != 0u && dot(push.windDelta, push.windDelta) > 0.0) {
        WorldPush world = ConstPtr<WorldPush>(push.worldPushAddr)[0];
        float2 ndc = uv * 2.0 - 1.0;
        float4 nearH = mul(world.invViewProj, float4(ndc, 1.0, 1.0));
        float4 farH = mul(world.invViewProj, float4(ndc, 0.0, 1.0));
        float3 nearP = nearH.xyz / nearH.w;
        float3 farP = farH.xyz / farH.w;
        float3 origin = nearP + world.camOffset;
        float3 direction = normalize(farP - nearP);
        float cloudT = cloudDepth(world, origin, direction);
        if (cloudT > 0.0) {
            float3 position = origin + direction * cloudT;
            float3 windPosition = position + float3(push.windDelta.x, 0.0, push.windDelta.y);
            float4 currentClip = mul(world.curViewProj, float4(position - world.camOffset, 1.0));
            float4 windClip = mul(world.curViewProj, float4(windPosition - world.camOffset, 1.0));
            if (currentClip.w > 0.0 && windClip.w > 0.0) {
                float2 currentNdc = currentClip.xy / currentClip.w;
                float2 windNdc = windClip.xy / windClip.w;
                previousUv += (windNdc - currentNdc) * 0.5;
            }
        }
    }

    // Build the current neighbourhood range before touching history. Geometry pixels never blend, and a
    // history tap outside the frame or behind the near plane is discarded outright.
    float3 lo = current;
    float3 hi = current;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            float2 tap = clamp(uv + float2(dx, dy) / push.size, float2(0.0), float2(1.0));
            float3 value = currentColor.SampleLevel(tap, 0.0).rgb;
            lo = min(lo, value);
            hi = max(hi, value);
        }
    }

    float3 result = current;
    bool isSky = depth <= 1.0e-5;
    bool inside = all(previousUv >= float2(0.0)) && all(previousUv <= float2(1.0));
    if (push.historyValid != 0u && isSky && inside) {
        float3 history = historyColor.SampleLevel(previousUv, 0.0).rgb;
        history = clamp(history, lo, hi);
        // The scene motion vector cannot see cloud wind drift. A moderate weight still removes most of
        // the per-frame sample noise without turning a moving edge into a long-lived smear.
        result = lerp(current, history, 0.80);
    }

    outputColor[pixel] = float4(result, 1.0);
    outputHistory[pixel] = float4(result, 1.0);
}
