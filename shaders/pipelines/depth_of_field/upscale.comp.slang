// Depth-of-field upscale: half-resolution gathered image to display resolution.
//
// A plain bilinear magnify would blend across every silhouette, because the two texels it mixes can belong to a
// foreground and a background whose circles of confusion are completely different. Each half-resolution texel
// carries the view distance it was gathered at, so the four around a display pixel can be weighted by whether
// they describe the same surface, which keeps an in-focus edge next to a defocused background -- the case that
// matters most for this effect -- from bleeding in either direction.
//
// The blend is against the depth of the pixel being written, not against a threshold: the nearest texel always
// dominates, and only texels measuring the same distance get a full vote.
import display_common;
import bindings;

[[vk::push_constant]] DepthOfFieldPush pc;

// Declared here rather than imported; see bindings.slang for the contract and why.
[[vk::binding(0, 0)]] public Sampler2D sceneImage;
[[vk::binding(1, 0)]] [format("r32f")] public Sampler2D gDepth;
[[vk::binding(5, 0)]] public Sampler2D gatheredSampler;
[[vk::binding(6, 0)]] [format("rgba16f")] public RWTexture2D<float4> upsampleImage;
// The same scene binding 0 holds, at the resolution this stage writes. A pixel this pass hands back untouched
// must come from here rather than from the render-resolution original: magnifying that by the backend's scale
// leaves a one-pixel step against every neighbouring pixel the gather produced.
[[vk::binding(7, 0)]] public Sampler2D sharpSource;

// Reversed-Z depth to a distance in blocks along the ray; the same conversion the gather itself uses.
float viewDistance(float depth) {
    float denominator = depth + pc.projM22;
    return abs(pc.projM32 / (abs(denominator) < 1.0e-6 ? -1.0e-6 : denominator));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint width, height;
    upsampleImage.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    float2 uv = (float2(tid.xy) + 0.5) / float2(float(width), float(height));
    float centreDistance = viewDistance(gDepth.SampleLevel(uv, 0.0).x);
    // A pixel whose own disc is under a pixel is in focus, and for it this pass would be a loss: the gather runs
    // at half the render resolution, so reading the gathered image here and magnifying it back is a bilinear
    // upscale from half resolution to display -- several pixels of softening even when the disc is nothing at
    // all. The sharp image is right there and is exactly what an in-focus pixel is supposed to show, so it is
    // taken directly. Without this, enabling depth of field softens the whole frame at ANY radius, which reads
    // as the feature never working rather than as the blur being strong.
    // The same circle of confusion the gather draws, from the same thin-lens form: the blur is the hyperfocal
    // constant times how far this surface is from the focal plane, over its own distance. It has to be the same
    // expression as the gather's, because this decides whether the gather's answer is used at all -- a different
    // metric here would compare a pixel's blur against a threshold borrowed from another formula.
    float ownRadius = 0.0;
    if (pc.targetRadiusPixels > 0.0 && centreDistance > 0.0 && pc.focusDistance > 0.0
            && pc.focalLength > 0.0) {
        float ownDenominator = 1.0 - pc.focalLength / centreDistance;
        if (ownDenominator > 0.0) {
            float imageDistance = pc.focalLength / ownDenominator;
            float focusImageDistance = pc.focusDistance >= 1.0e4
                    ? pc.focalLength
                    : pc.focalLength / (1.0 - pc.focalLength / pc.focusDistance);
            ownRadius = min(pc.depthScale * abs(focusImageDistance - imageDistance) / imageDistance,
                    pc.targetRadiusPixels);
        }
    }
    // How much of this pixel's value comes from the gather, ramped rather than switched. A pixel well inside
    // focus takes the sharp image and a defocused one takes the gather, but the two do not meet at a line: the
    // radius varies continuously across the frame, and a hard threshold makes every contour where it crosses the
    // value a visible band -- most obviously along the edge of a near block against a far wall, where the two
    // sides of the discontinuity sit on opposite sides of the threshold. Fading across a small multiple of the
    // threshold turns that contour into a gradient, which is what the eye expects of a change in focus.
    float gatherBlend = smoothstep(pc.gatherMinRadiusPixels, pc.gatherMinRadiusPixels * 3.0, ownRadius);
    if (gatherBlend <= 0.0) {
        // Only reachable for a pixel with no distance or no disc at all, since the radius the gather reports is
        // floored rather than zero.
        upsampleImage[int2(tid.xy)] = float4(sharpSource.SampleLevel(uv, 0.0).rgb, 1.0);
        return;
    }
    float3 sharpColour = sharpSource.SampleLevel(uv, 0.0).rgb;
    // Weights follow the bilinear footprint, so a texel that barely contributes to the magnified value cannot
    // dominate it either. +0.5 puts the sample at the texel centre.
    float2 position = uv * float2(float(pc.gatherWidth), float(pc.gatherHeight)) - 0.5;
    float2 base = floor(position);
    float2 fraction = position - base;

    float3 sum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            int2 texel = int2(base) + int2(dx, dy);
            texel = clamp(texel, int2(0, 0), int2(pc.gatherWidth - 1, pc.gatherHeight - 1));
            float bilinear = (dx == 0 ? 1.0 - fraction.x : fraction.x)
                    * (dy == 0 ? 1.0 - fraction.y : fraction.y);
            if (bilinear <= 0.0) {
                continue;
            }
            float4 gathered = gatheredSampler.Load(int3(texel, 0));
            if (any(!isfinite(gathered.rgb))) {
                continue;
            }
            float texelDistance = viewDistance(gathered.a);
            // A SAFEGUARD, not the local depth test. The gather is already depth-aware -- it decides per tap
            // whether a surface scatters onto this pixel or is merely collected by it -- so a texel that differs
            // in depth from the receiver may be differing precisely BECAUSE the gather put a defocused
            // foreground there, which is the light a lens would scatter onto this pixel. Rejecting it here threw
            // that away and left the pixel averaging only the one or two texels on its own side of the
            // discontinuity: a one-texel-wide patch of near-sharp colour along every depth edge, which is the
            // band. The tolerance is therefore wide enough to accept a neighbour the gather chose to bleed, and
            // only narrow enough to keep the far plane -- sky, at infinity by construction -- from mixing with
            // near geometry, which is the one case that produces an obviously wrong colour rather than a soft
            // edge.
            float relative = abs(texelDistance - centreDistance) / max(centreDistance, 1.0e-3);
            float weight = bilinear * clamp(1.0 - relative / (pc.depthTolerance * 8.0), 0.0, 1.0);
            if (weight <= 0.0) {
                continue;
            }
            sum += gathered.rgb * weight;
            weightSum += weight;
        }
    }
    // Every neighbouring texel describing a different surface is a pixel smaller than one gather texel. Falling
    // back to the plain bilinear magnify keeps it from going black.
    float3 gatheredColour = weightSum > 0.0 ? sum / weightSum : sharpColour;
    // And the ramp above blends the two, so the boundary between them is a change of focus rather than a seam.
    float3 colour = lerp(sharpColour, gatheredColour, gatherBlend);
    // The undersampled pixels are exactly a source texel wide, so the alpha channel is left clear: it is not
    // part of the scene-linear radiance the display chain consumes.
    upsampleImage[int2(tid.xy)] = float4(colour, 1.0);
}
