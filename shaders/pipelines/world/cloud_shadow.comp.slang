// Cloud shadow map bake. One texel is a tile-space column; the pass integrates the cloud-body SDF
// toward the dominant celestial body and stores the resulting transmittance for terrain NEE.
//
// The map covers the same 2048-block tile as the placement/SDF fields, so the result is periodic and
// can be sampled with the world-pinned cloud anchor. It is intentionally body-only (no detail
// erosion): the high-frequency erosion does not move the shadow boundary enough to justify the cost.
import world_common;

// Keep in lock-step with RtCloudNoise and clouds.slang.
static const uint CLOUD_SHADOW_DIM = 256u;
static const float CLOUD_SHADOW_TILE = 2048.0;
static const int NVDF_X = 128;
static const int NVDF_Y = 32;
static const int NVDF_Z = 128;
static const uint NVDF_BASE = 3309568u;
static const float NVDF_NOMINAL_COVERAGE = 0.5;
static const float NVDF_COVERAGE_OFFSET = 160.0;
static const float NVDF_PROFILE_DEPTH = 40.0;
static const float CLOUD_EXTINCTION = 0.04;

public struct CloudShadowPush {
    public uint64_t worldPushAddr;
    public uint64_t noiseAddr;
    public uint64_t outputAddr;
    public float thickness;
    public float coverage;
    public float densityScale;
    public float mapTile;
    public float pad0;
    public float pad1;
}

[[vk::push_constant]] CloudShadowPush push;

public float3 celestialDirection(float angleRad, float noonTiltRad) {
    float peak = cos(angleRad);
    return float3(-sin(angleRad), cos(noonTiltRad) * peak, sin(noonTiltRad) * peak);
}

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

public float bodyDensity(float2 tileUV, float height) {
    float sdf = sampleNvdf(tileUV, height);
    float sdfLive = sdf - (push.coverage - NVDF_NOMINAL_COVERAGE) * NVDF_COVERAGE_OFFSET;
    if (sdfLive >= 0.0) {
        return 0.0;
    }
    float profile = saturate(-sdfLive / NVDF_PROFILE_DEPTH);
    float heightDensity = clamp(0.25 + 0.75 * height * push.thickness / 100.0, 0.25, 1.0);
    return profile * heightDensity;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if (any(threadId.xy >= uint2(CLOUD_SHADOW_DIM, CLOUD_SHADOW_DIM))) {
        return;
    }

    WorldPush world = ConstPtr<WorldPush>(push.worldPushAddr)[0];
    float3 sunDir = celestialDirection(world.celestial.x, world.skyLook1.x);
    float3 moonDir = celestialDirection(world.celestial.y, world.skyLook1.x);
    bool sunUp = sunDir.y > 0.02;
    bool moonUp = moonDir.y > 0.02;
    float3 lightDir = sunUp && world.skyLook0.x >= world.skyLook0.y ? sunDir
            : (moonUp ? moonDir : float3(0.0, 1.0, 0.0));

    Ptr<uint> output = Ptr<uint>(push.outputAddr);
    uint index = threadId.y * CLOUD_SHADOW_DIM + threadId.x;
    if (!sunUp && !moonUp) {
        output[index] = 255u;
        return;
    }

    float2 tile = (float2(threadId.xy) + 0.5) / float(CLOUD_SHADOW_DIM);
    float path = push.thickness / max(lightDir.y, 0.15);
    const int steps = 12;
    float opticalDepth = 0.0;
    for (int i = 0; i < steps; i++) {
        float t = (float(i) + 0.5) / float(steps);
        float3 position = float3(tile.x * push.mapTile, 0.0, tile.y * push.mapTile)
                + lightDir * (t * path);
        float2 uv = frac(position.xz / push.mapTile);
        opticalDepth += bodyDensity(uv, t) * (path / float(steps));
    }
    float transmittance = exp(-CLOUD_EXTINCTION * push.densityScale * opticalDepth);
    output[index] = uint(round(saturate(transmittance) * 255.0));
}
