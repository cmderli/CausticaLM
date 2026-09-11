// Edge-aware spatial denoiser for the path-traced image.
//
// The renderer has no temporal history of its own — the reconstruction backend owns reprojection and
// accumulation — so this pass works on one frame: an a-trous (B3-spline) filter whose tap weights follow
// the guide buffers, run at widening spacing, followed by a bilinear resolve into the display-resolution
// output.
//
// The signal is filtered demodulated (radiance divided by the tap's own diffuse+specular albedo, then
// multiplied back by the centre's) so that texture detail inside a surface is not averaged away with the
// noise. Weights reject taps across a depth, normal, albedo, roughness or luminance discontinuity, which
// is what keeps edges and material boundaries intact.
//
// Mirror-like surfaces keep their own sample: a reflection has no neighbourhood to average with, and
// blurring it would trade noise for a visibly wrong image. Those pixels are the ones a temporal
// reprojection pass is supposed to fix, not this one.
import recon_common;
import bindings;

[[vk::push_constant]] DenoisePush pc;

// ACEScg luminance weights, matching the rest of the renderer's scene-linear working space.
static const float3 ACESCG_LUMA = float3(0.2722287168, 0.6740817658, 0.0536895174);
static const float SKY_DISTANCE = 1.0e6;
static const float MIN_DEMOD = 0.01;

float luma(float3 c) {
    return dot(c, ACESCG_LUMA);
}

float2 texelUv(int2 pix, int2 size) {
    return (float2(pix) + 0.5) / float2(size);
}

// Positive view-space distance for a reversed-Z depth. Both projection terms are negative for a standard
// reversed-Z perspective matrix, so the ratio is already positive; the abs() keeps this correct if a
// projection ever flips the depth convention.
float viewDistance(float z) {
    float denom = z + pc.projM22;
    return abs(pc.projM32 / (abs(denom) < 1.0e-6 ? -1.0e-6 : denom));
}

float distanceAt(float2 uv) {
    float z = gDepth.SampleLevel(uv, 0.0).x;
    return z <= 0.0 ? SKY_DISTANCE : viewDistance(z);
}

// Radiance with the surface's own reflectance divided out, so taps with different albedo can still be
// compared in the same units.
float3 demodulate(float3 radiance, float3 diffuseAlbedo, float3 specularAlbedo) {
    return radiance / max(diffuseAlbedo + specularAlbedo, MIN_DEMOD);
}

// B3-spline tap weights (1, 4, 6, 4, 1) / 16.
float b3(int i) {
    int a = abs(i);
    if (a == 0) return 6.0 / 16.0;
    if (a == 1) return 4.0 / 16.0;
    if (a == 2) return 1.0 / 16.0;
    return 0.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int2 pix = int2(id.xy);

    if (pc.mode == 1) {
        // Bilinear resolve: the atrous iterations already re-modulated to radiance, so bringing the
        // render-res result up to display res is a plain filtered fetch.
        int2 displaySize = int2(pc.displayWidth, pc.displayHeight);
        if (any(pix >= displaySize)) {
            return;
        }
        float2 uv = (float2(pix) + 0.5) / float2(pc.displayWidth, pc.displayHeight);
        dstImage[pix] = float4(srcImage.SampleLevel(uv, 0.0).rgb, 1.0);
        return;
    }

    int2 renderSize = int2(pc.renderWidth, pc.renderHeight);
    if (any(pix >= renderSize)) {
        return;
    }
    float2 uv = texelUv(pix, renderSize);

    float4 centreNormal = gNormal.SampleLevel(uv, 0.0);
    float4 centreAlbedo = gAlbedo.SampleLevel(uv, 0.0);
    float3 centreSpecAlbedo = gSpecAlbedo.SampleLevel(uv, 0.0).rgb;
    float centreDistance = distanceAt(uv);
    float3 centreSignal = demodulate(srcImage.SampleLevel(uv, 0.0).rgb,
            centreAlbedo.rgb, centreSpecAlbedo);

    // Firefly suppression. A sample far above its own neighbourhood would otherwise be smeared into a wide
    // blot by the widening taps, so it is clamped against the 3x3 mean of the signal the filter works in.
    float3 neighbourhood = float3(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            int2 q = clamp(pix + int2(x, y), int2(0, 0), renderSize - 1);
            float2 quv = texelUv(q, renderSize);
            neighbourhood += demodulate(srcImage.SampleLevel(quv, 0.0).rgb,
                    gAlbedo.SampleLevel(quv, 0.0).rgb, gSpecAlbedo.SampleLevel(quv, 0.0).rgb);
        }
    }
    float ceiling = max(luma(neighbourhood / 9.0), 1.0e-4) * pc.fireflyClamp;
    float centreLuma = luma(centreSignal);
    if (centreLuma > ceiling) {
        centreSignal *= ceiling / centreLuma;
    }

    float3 sum = float3(0.0);
    float weightSum = 0.0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            int2 q = pix + int2(x, y) * pc.step;
            if (any(q < int2(0, 0)) || any(q >= renderSize)) {
                continue;
            }
            float2 quv = texelUv(q, renderSize);
            float4 tapNormal = gNormal.SampleLevel(quv, 0.0);
            float4 tapAlbedo = gAlbedo.SampleLevel(quv, 0.0);
            float3 tapSpecAlbedo = gSpecAlbedo.SampleLevel(quv, 0.0).rgb;
            float3 tapSignal = demodulate(srcImage.SampleLevel(quv, 0.0).rgb,
                    tapAlbedo.rgb, tapSpecAlbedo);

            float weight = b3(x) * b3(y);
            float tapDistance = distanceAt(quv);
            weight *= exp(-abs(tapDistance - centreDistance) / max(pc.depthSigma * centreDistance, 1.0e-4));
            weight *= pow(max(dot(tapNormal.xyz, centreNormal.xyz), 0.0), pc.normalPower);
            weight *= exp(-abs(luma(tapAlbedo.rgb) - luma(centreAlbedo.rgb)) / max(pc.albedoSigma, 1.0e-4));
            weight *= exp(-abs(luma(tapSignal) - centreLuma) / max(pc.lumaSigma, 1.0e-4));
            weight *= exp(-abs(tapNormal.w - centreNormal.w) / 0.1);

            sum += tapSignal * weight;
            weightSum += weight;
        }
    }
    float3 filtered = weightSum > 0.0 ? sum / weightSum : centreSignal;

    // Mirror-like taps are exempt: below the mirror roughness the centre sample is the answer, and the
    // filter fades in as the lobe widens.
    float mirror = 1.0 - smoothstep(0.0, max(pc.mirrorRoughness, 1.0e-4), centreNormal.w);
    float3 signal = lerp(filtered, centreSignal, mirror);

    dstImage[pix] = float4(signal * max(centreAlbedo.rgb + centreSpecAlbedo, MIN_DEMOD), 1.0);
}
