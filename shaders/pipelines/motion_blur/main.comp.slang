// Motion blur: integrates the reconstructed scene along each pixel's motion vector over the shutter interval.
//
// The renderer's motion guide holds render-pixel velocity with "previous - current" sign, so the direction the
// surface travels is `-motion`. A pixel's exposure is the average of the scene over the shutter interval, and
// with only this frame available that integral is gathered by sampling *forward along the travel direction*:
// the radiance a pixel receives came from where the surface is now, which is why a moving object trails
// behind itself. Sampling the other way (toward the previous position) instead spreads whatever is ahead of
// the surface onto it and produces no trail at all, which is the mistake this sign avoids.
//
// Taps are therefore at `uv + alpha * shutter * (-motion)` for alpha in [0,1], the endpoints being the
// shutter's own sample and the oldest one, and the result is their average.
//
// Two things keep it from looking like a smear instead of motion:
//   - a depth test per tap, so a moving object does not drag the background across itself (and vice versa),
//     with sky and surface treated as mutually exclusive;
//   - a clamp on the streak length, with the shutter scaled down rather than truncated, so a fast turn stays
//     a consistent shorter streak instead of a full-screen smear that stops at an arbitrary radius.
import display_common;
import bindings;

[[vk::push_constant]] MotionBlurPush pc;

float2 texelUv(float2 pixel) {
    return (pixel + 0.5) / float2(float(pc.displayWidth), float(pc.displayHeight));
}

// Positive view-space distance for a reversed-Z depth; the guide reports 0 for a ray that hit nothing.
// Both projection terms are negative for a standard reversed-Z perspective matrix, so the ratio is already
// positive and the abs() only guards against a projection that flips the depth convention.
float viewDistance(float z) {
    float denom = z + pc.projM22;
    return abs(pc.projM32 / (abs(denom) < 1.0e-6 ? -1.0e-6 : denom));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int2 pixel = int2(tid.xy);
    if (pixel.x >= pc.displayWidth || pixel.y >= pc.displayHeight) {
        return;
    }
    float2 uv = texelUv(float2(pixel));
    float3 centre = sceneImage.SampleLevel(uv, 0.0).rgb;

    // Travel direction in display pixels over the shutter interval: the guide is "previous - current", so the
    // surface moves along its negation. A guide that has not been written yet reads back as whatever the
    // allocation held, which can be non-finite; sampling with a non-finite coordinate is undefined, and the
    // finite-value test below is what keeps such a frame from reaching the texture unit at all.
    float2 motion = gMotion.SampleLevel(uv, 0.0).xy * float2(pc.velocityScaleX, pc.velocityScaleY);
    float2 offset = -motion * pc.shutterFraction;
    float length2 = dot(offset, offset);
    if (pc.samples <= 1 || pc.shutterFraction <= 0.0 || !(length2 >= 0.0) || length2 < 1.0e-8) {
        outputImage[pixel] = float4(centre, 1.0);
        return;
    }

    // Scale the whole streak down instead of cutting it off, so the shutter stays consistent: a pixel whose
    // motion exceeds the cap gets the same blur as one moving exactly at the cap.
    float length = sqrt(length2);
    float scale = min(1.0, pc.maxRadiusPixels / max(length, 1.0e-6));
    float2 step = offset * scale;
    float2 direction = step / max(length * scale, 1.0e-6);

    float centreDepth = gDepth.SampleLevel(uv, 0.0).x;
    bool centreIsSky = centreDepth <= 0.0;
    float centreDistance = centreIsSky ? 0.0 : viewDistance(centreDepth);

    float3 sum = centre;
    float weightSum = 1.0;
    int taps = clamp(pc.samples, 1, 32);
    // Endpoints included: alpha 0 is this frame's own sample and alpha 1 the oldest one in the interval.
    for (int i = 1; i < taps; i++) {
        float alpha = float(i) / float(taps - 1);
        float2 tapUv = uv + step * alpha / float2(float(pc.displayWidth), float(pc.displayHeight));
        if (tapUv.x < 0.0 || tapUv.x > 1.0 || tapUv.y < 0.0 || tapUv.y > 1.0) {
            continue; // off screen: that part of the exposure simply does not exist
        }
        float tapDepth = gDepth.SampleLevel(tapUv, 0.0).x;
        bool tapIsSky = tapDepth <= 0.0;
        float weight;
        if (tapIsSky != centreIsSky) {
            weight = 0.0; // surface against sky: a depth discontinuity, never the same surface
        } else if (tapIsSky) {
            weight = 1.0; // both sky: the sky moves with the camera and must streak too
        } else {
            // Same surface means a similar view distance. The tolerance is relative, and deliberately wide
            // enough that a slanted floor seen during a camera move still counts as one surface.
            float difference = abs(viewDistance(tapDepth) - centreDistance) / max(centreDistance, 1.0e-3);
            weight = saturate(1.0 - difference / max(pc.depthTolerance, 1.0e-6));
        }
        sum += sceneImage.SampleLevel(tapUv, 0.0).rgb * weight;
        weightSum += weight;
    }

    outputImage[pixel] = float4(sum / weightSum, 1.0);
}
