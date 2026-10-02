// Depth of field, part two: a disc-of-confusion gather over the reconstructed image.
//
// The trace already blurs with the camera aperture, but one primary ray per pixel samples one point of the
// pupil, so it only resolves a small disc: a pixel whose circle of confusion is dozens of pixels wide gets a
// single sample of it, and what that produces is not blur but speckle, with a hard in/out flicker along every
// silhouette as the pupil sample moves. Raising the sample count resolves that by tracing more whole paths per
// pixel, which costs the frame's entire ray budget multiplied by the sample count.
//
// This pass resolves the rest of the disc without a single extra ray. It re-blurs the image the trace already
// produced, gathering with the residual radius the trace did not cover, so the traced part stays a physically
// correct near-field bokeh — the shape a real lens makes — and the gather expands it to the full aperture the
// camera was set to.
//
// The gather runs at half resolution and accumulates over frames. Half resolution is what makes a kernel of
// dozens of pixels affordable, and the history is what makes the tap count sufficient: the tap pattern advances
// with the frame, so a wide disc is covered by the union of several frames rather than by one pass' taps. The
// upscale back to display resolution reads the four half-resolution texels around each pixel, so the result is
// not a half-resolution image blown up.
//
// Two things keep a wide gather from smearing across a silhouette:
//   * a tap may only contribute to a pixel it is not occluded from, so a foreground tap cannot bleed onto the
//     background behind it and vice versa;
//   * a tap's own circle of confusion has to overlap the receiver's, which is what stops a sharp foreground
//     from being dragged across a defocused background at full strength.
import display_common;
import bindings;

[[vk::push_constant]] DepthOfFieldPush pc;

// Declared here rather than imported: a module that imports the whole set gets renumbered, because Slang only
// carries the globals this entry point reaches. See bindings.slang for the full contract.
[[vk::binding(0, 0)]] public Sampler2D sceneImage;
[[vk::binding(1, 0)]] [format("r32f")] public Sampler2D gDepth;
[[vk::binding(2, 0)]] public Sampler2D gMotion;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> gatheredImage;
// One binding serving both roles: the gather READS it as its history, the blur stages WRITE it as one half of
// the ping-pong they traverse. A descriptor set cannot hold one image as both a sample and a destination, and
// neither can a shader declare one binding twice, so this is a read-write texture rather than a sampled one --
// which also means the gather has to load its history at texel centres instead of filtering it.
[[vk::binding(4, 0)]] [format("rgba16f")] public RWTexture2D<float4> historyImage;



static const float PI2 = 6.283185307179586;
// Golden angle. Successive samples of a spiral stepped by it never align on a spoke, which is what keeps a
// small tap count from showing the radial banding an equal-angle pattern produces.
static const float GOLDEN_ANGLE = 2.399963229728653;
// The gap between neighbouring taps of the gather's pattern, as a fraction of its disc radius: the taps divide
// the disc's AREA, so their radial step is the radius over the square root of the tap count. It is computed
// rather than a constant because it is only about a tenth at the tap count this pass uses -- a hardcoded tenth
// under-smooths every disc taller than that, which is where the tap pattern is most visible.
float tapSpacingOfRadius(uint taps) {
    return 1.0 / sqrt(float(max(taps, 1u)));
}

// Reversed-Z depth to a distance in blocks along the ray, the same conversion the world's own DOF scale uses.
// Both projection terms are negative for this convention, so the ratio is already positive.
float viewDistance(float depth) {
    float denominator = depth + pc.projM22;
    return abs(pc.projM32 / (abs(denominator) < 1.0e-6 ? -1.0e-6 : denominator));
}

// The circle of confusion of a surface at `distance` blocks, in display pixels.
//
// The thin lens, applied to the image distances rather than approximated -- the formulation of Potmesil &
// Chakravarty, "A lens and aperture camera model for synthetic image generation", SIGGRAPH 1981, which is also
// where the equivalent pixel-radius form in Blender's compositor comes from. A lens of focal length f forms the
// image of a surface at Z at v(Z) = 1 / (1/f - 1/Z), and the cone from an aperture of radius f/N covers a disc
// of (f/N) * |v(Zf) - v(Z)| / v(Z) on the film -- which is exactly what "out of focus" is, and what the trace's
// own aperture samples already draw. `pc.depthScale` is that aperture radius in display pixels and
// `pc.focalLength` is f in blocks, so the falloff is two divisions.
//
// The image distances are the point. A focus on the horizon leaves the FOREGROUND sharp, because v(Z) converges
// to f as Z grows and it is the surfaces far from the focal plane in v that blur -- so at 107 blocks with a
// 15 mm lens a surface at 3 blocks is 3 pixels out, not the 18 that a |Zf - Z| / Z ratio gives it. That ratio is
// what an earlier version used, and it is largest in the near field exactly when the focus is distant, so aiming
// at a hill buried the whole foreground under the widest disc the renderer draws.
float cocRadiusPixels(float distance) {
    if (!(distance > 0.0) || !(pc.focusDistance > 0.0) || !(pc.focalLength > 0.0)) {
        return 0.0;
    }
    // The surface's own image distance, guarded the way the depth conversion is: a surface at or inside the
    // focal length has no real image, and the lens cannot be focused on it either.
    float denominator = 1.0 - pc.focalLength / distance;
    if (!(denominator > 0.0)) {
        return 0.0;
    }
    float imageDistance = pc.focalLength / denominator;
    // Focused at infinity the film sits exactly at the focal length, which is the same expression with the
    // focal plane's image distance replaced by f.
    float focusImageDistance = pc.focusDistance >= 1.0e4
            ? pc.focalLength
            : pc.focalLength / (1.0 - pc.focalLength / pc.focusDistance);
    float blur = pc.depthScale * abs(focusImageDistance - imageDistance) / imageDistance;
    // The circle of confusion: past this the frame reads as one soft field rather than a subject against a
    // background, and it is the same number the traced-ray gate is scaled from.
    float full = min(blur, pc.targetRadiusPixels);
    float traced = min(full, pc.tracedRadiusPixels);
    // Blurring an image that is ALREADY blurred by `traced` with a further radius `g` gives a disc of about
    // sqrt(traced^2 + g^2), not traced + g: two convolutions of discs compose in quadrature. So gathering by
    // `full - traced` as if the radii added produces substantially less blur than the aperture was set to --
    // at a full radius of 54 and a traced one of 12 it lands at 54 instead of 42. Solving for the radius that
    // composes to `full` is what makes the effect the aperture the camera is actually at.
    float residual = sqrt(max(full * full - traced * traced, 0.0));
    // Floored rather than dropped. Returning zero for a disc under the threshold made this function jump from 0
    // to the threshold across a contour, and everything downstream follows it: the tap pattern changes, the
    // blur stage's radius changes, and the upscale's own ramp is fed a value that skips the very range it
    // interpolates over -- so the ramp could not smooth the boundary it exists to smooth. A floor keeps the
    // function continuous, and a disc at the floor is a sub-texel disc whose taps land inside one texel, so the
    // gather averages that texel rather than sampling something new.
    return max(residual, pc.gatherMinRadiusPixels);
}

// Depth of the surface at a UV, converted to a distance, for the tests this pass makes itself.
float sampleDistance(float2 uv) {
    return viewDistance(gDepth.SampleLevel(uv, 0.0).x);
}

// The RAW guide sample. The alpha channel of the gathered image carries THIS, not a distance, so that every
// stage reading it -- the blur's depth test and the upscale's -- can call viewDistance exactly once. Carrying a
// distance instead made them call it on a value that was already converted, which sent the upscale's comparison
// off by an order of magnitude and had it reject every texel it was handed: the gather did its work and the
// upscale threw it away, which is what "the blur does nothing" looked like from the outside.
float sampleRawDepth(float2 uv) {
    return gDepth.SampleLevel(uv, 0.0).x;
}

// The gather's residual smoothing, run as two separable passes.
//
// A gather can only place as many samples as it can afford, and the taps of a fixed pattern are not
// independent: measured on a synthetic edge, the noise left by 96 taps is about 79% of the edge's own slope,
// and raising that to 1536 taps only brings it to 38% -- the pattern's EFFECTIVE sample count is far below its
// tap count, so taps are a very expensive way to buy smoothness. A separable box over the gathered image has no
// pattern at all and averages the tap-to-tap variation directly: a box a few texels wide costs two cheap passes
// and removes what dozens of times more taps would only reduce.
//
// It runs on the gather's own half-resolution image, so its radius in texels is already the radius in display
// pixels. The depth the pixels carry is blurred with them, because the upscale compares those depths to decide
// which gathered texel belongs to the surface it is magnifying, and a depth that no longer matches the colour
// beside it would put a seam along every silhouette.
[shader("compute")]
[numthreads(8, 8, 1)]
void blur(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.gatherWidth || tid.y >= pc.gatherHeight) {
        return;
    }
    int2 pixel = int2(tid.xy);
    int2 size = int2(pc.gatherWidth, pc.gatherHeight);
    // The radius follows THIS pixel's disc, not the frame's. The taps of the gather's pattern sit one gap
    // apart, and that gap is the disc radius over the square root of the tap count -- so a pixel with a small
    // circle of confusion has densely packed taps and needs almost no smoothing, while a wide one needs a wide
    // box. A frame-wide radius would over-blur every disc narrower than the widest one in the frame, which is
    // most of them. The gather wrote the distance it read into the alpha channel, which is all this needs.
    int radius = clamp(int(round(cocRadiusPixels(viewDistance(gatheredImage.Load(pixel).a))
            * tapSpacingOfRadius(pc.tapCount) * max(pc.displayToGather, 1.0e-4))), 0, 32);
    if (radius <= 0) {
        // Nothing to smooth: hand the image straight to the other half of the ping-pong.
        historyImage[pixel] = gatheredImage.Load(pixel);
        return;
    }
    // Across for stage 1, down for stage 2. A separable box is two one-dimensional averages, and each one is
    // exact -- no weights to lose energy to and no taps to leave a pattern.
    int2 step = pc.stage == 1u ? int2(1, 0) : int2(0, 1);
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    float count = 0.0;
    float centreDistance = viewDistance(gatheredImage.Load(pixel).a);
    for (int i = -radius; i <= radius; ++i) {
        int2 at = clamp(pixel + step * i, int2(0, 0), size - 1);
        float4 value = gatheredImage.Load(at);
        // A tap whose depth says it belongs to a different surface is not part of this pixel's value. Averaging
        // depth itself would drag a foreground mean onto a background pixel and vice versa, which is the seam
        // this stage would otherwise create along every edge.
        float relative = abs(viewDistance(value.a) - centreDistance) / max(centreDistance, 1.0e-3);
        if (relative > pc.depthTolerance) {
            continue;
        }
        sum += value;
        count += 1.0;
    }
    // The result goes to the image the descriptor set binds as its sampled one, which is the other half of the
    // ping-pong: the gather filled this stage's source and this stage fills what the next stage reads.
    historyImage[pixel] = count > 0.0 ? sum / count : gatheredImage.Load(pixel);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.gatherWidth || tid.y >= pc.gatherHeight) {
        return;
    }
    int2 pixel = int2(tid.xy);
    float2 size = float2(pc.gatherWidth, pc.gatherHeight);
    float2 uv = (float2(pixel) + 0.5) / size;
    float2 texel = 1.0 / size;

    float centreRawDepth = sampleRawDepth(uv);
    float centreDistance = viewDistance(centreRawDepth);
    uint taps = max(pc.tapCount, 1u);
    // The disc is bounded in THIS image's pixels, and the bound is the frame. A disc wider than the frame has
    // almost every tap outside it: each one is discarded, the survivors are the few near the centre, and the
    // result collapses onto the pixel's own colour -- so a too-wide disc does not produce a wide blur, it
    // produces NO blur, which is the opposite of what its size suggests. Half the frame's shorter side is the
    // widest disc whose taps still land inside it with room to spare.
    // The disc arrives in DISPLAY pixels, which is the space the effect is specified in, and it is used here as
    // a distance in THIS image's texels. The two are not the same: the gather runs at `renderSize / 2` while the
    // display is the window, and a reconstruction backend puts a factor of two or three between them. Applying a
    // display-space radius directly in gather texels drew every disc that many times too wide, which is what made
    // a stopped-down lens blur the whole frame and left the traced disc -- correctly sized in display space --
    // too small to subtract, so its sampling showed through as outline.
    float gatherRadiusScale = max(pc.displayToGather, 1.0e-4);
    float frameLimit = 0.5 * min(pc.gatherWidth, pc.gatherHeight);
    float centreRadius = min(cocRadiusPixels(centreDistance) * gatherRadiusScale, frameLimit);
    // A debug view of the radius THIS gather computed, in the gather's own pixels, drawn where the gather
    // computes it. Every other view of the circle of confusion is a second implementation of the same formula
    // and can only agree with itself; this one cannot disagree with the effect, because it IS the effect's own
    // number. Grey means "nothing to gather" and the bands are 1, 2, 4, 8, 16 and 32 gather pixels, so a picture
    // that never leaves the dark end is a picture with no disc to blur with.
    if (pc.stage == 9u) {
        // The inputs first, because "the radius is flat" and "the depth never arrived" look identical in a chart
        // of the radius alone and have nothing to do with each other:
        //   red   = the depth read as nothing (sky, or no usable distance), so the radius cannot help but be 0
        //   blue  = a real distance but a radius under the threshold, so the disc is sub-pixel
        //   bands = the radius itself, 1, 2, 4, 8, 16, 32 gather pixels
        // THREE saturated, unmistakable colours and nothing else. A ramp was a mistake twice over: a grey ramp
        // cannot be told from an untouched buffer, and it invites reading a value off a picture. These cannot be
        // confused with scene content or with each other, and each answers one question:
        //   MAGENTA = the depth read as nothing at all (sky, or an unwritten guide)
        //   BLUE    = a distance below one block, which puts the disc under a pixel and is why nothing blurs
        //   ORANGE  = a distance at or above one block, so the radius is coming from somewhere else entirely
        // The radius is not shown here at all -- the chart is about the INPUT, because every previous round
        // conflated "the input is wrong" with "the number derived from it is small".
        float3 tint;
        if (!(centreDistance > 0.0)) {
            tint = float3(1.0, 0.0, 1.0);
        } else if (centreDistance < 1.0) {
            tint = float3(0.1, 0.3, 1.0);
        } else {
            tint = float3(1.0, 0.55, 0.0);
        }
        gatheredImage[pixel] = float4(tint, centreRawDepth);
        return;
    }
    // A disc under the threshold is sub-pixel, so the spiral below collapses onto the pixel's own colour by
    // itself -- the taps move less than a texel and the weights are dominated by the centre. There used to be an
    // early-out here handing back the sharp image verbatim, and it was a second hard cut at the same threshold
    // the upscale switched on: the pixels that took it also skipped the temporal history, so a pixel and its
    // neighbour either side of a depth discontinuity differed by a whole frame's accumulation as well as by
    // their filtering. That is a band, and it flickers, because which side a pixel falls on follows the depth
    // guide rather than anything stable. The cost of not taking it is a handful of taps that land on the centre.


    // A fixed golden-angle spiral over the disc, reading the SHARP source with the disc's size carried by the
    // sample radius rather than by any filtering of the source. The golden-angle spiral is the standard
    // low-discrepancy pattern for covering a disc evenly, and leaving the source unfiltered keeps a tap from
    // standing for more area than it samples.
    //
    // No pyramid level and no tap-count scaling, because both are ways of making the taps stand for more area
    // than they sample -- and a tap that stands for more than it samples is a blur the disc did not ask for. At
    // the sizes a game actually uses the spiral's own density is the whole filter, and the reference proves it:
    // 128 taps at full resolution over a disc of a few dozen pixels, with the history left to settle what is
    // left. `max-blur-percent` is what bounds the disc, and the bound is what keeps the spacing small enough to
    // be a blur rather than a set of separate samples.
    // The taps divide the disc's AREA, so their radial step is the disc radius over the square root of the tap
    // count. It is the scale the residual smoothing below works at: that is the variation the taps cannot
    // resolve themselves.
    float spacing = max(centreRadius / sqrt(float(taps)), 1.0);
    // Frame-to-frame rotation of the tap pattern. The history turns the taps of several frames into one much
    // denser pattern without the flicker an independent random pattern per pixel would produce.
    //
    // The rotation advances with the frame and never repeats on a short cycle: taking the frame index modulo
    // anything visible makes the whole tap pattern snap back on that period, and the history -- which is what
    // turns several frames of taps into one dense pattern -- then blends across the snap. That is a periodic
    // flicker with the period of the wrap, which is exactly what a player sees as the edges of the frame's
    // objects jittering and cycling. The multiply keeps the float exact for far longer than a session.
    float phase = float((pc.frameIndex * 0x9E3779B9u) >> 8) * (GOLDEN_ANGLE / 16777216.0);

    // A tap stands at a fraction of the disc, and that fraction is what decides whether it can reach this
    // pixel. Written against the disc it is the same quantity as the circle of confusion in pixels; written
    // against the width of the span the disc covers it is that width in PIXELS, which the tap step below is
    // measured in. The two are proportional, so the comparison is unchanged, and keeping the working unit in
    // pixels is what makes the sample weight below a ratio of two lengths rather than of two arbitrary scales.
    float spanPixels = max(centreRadius, 1.0e-3);
    float centreRelative = min(centreRadius, spanPixels) / spanPixels;
    // How much of its own colour the pixel keeps when no tap reaches it. A pixel in focus keeps all of it --
    // that is what makes the in-focus part of the frame exact -- while a defocused one keeps almost none,
    // because its value has to come from the disc around it rather than from its own single sample.
    float centreWeight = lerp(1.0, 1.0e-5, centreRelative * centreRelative
            * centreRelative * centreRelative);

    float3 sum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    // The pixel's own colour, which is also what keeps the result finite when every tap is rejected.
    float3 selfColour = sceneImage.SampleLevel(uv, 0.0).rgb;
    float selfSamples = 1.0;
    float selfWeight = 0.0;
    for (uint tap = 0u; tap < taps; ++tap) {
        // A sample position is the radial step of the pattern scaled by the disc: the square root turns the
        // taps of one frame from a dense centre into an even spread over the disc's AREA, which is what a disc
        // of confusion is measured in.
        float radius = centreRadius * sqrt((float(tap) + 0.5) / float(taps));
        float angle = float(tap) * GOLDEN_ANGLE + phase;
        float2 offset = float2(cos(angle), sin(angle)) * radius;
        float2 tapUv = uv + offset * texel;
        if (tapUv.x < 0.0 || tapUv.x >= 1.0 || tapUv.y < 0.0 || tapUv.y >= 1.0) {
            continue;
        }
        float3 colour = sceneImage.SampleLevel(tapUv, 0.0).rgb;
        if (any(!isfinite(colour))) {
            continue;
        }
        float tapDistance = sampleDistance(tapUv);
        float tapRadius = cocRadiusPixels(tapDistance);
        float tapRelative = min(tapRadius, spanPixels) / spanPixels;

        // Which of the two the tap can reach is decided by which is more defocused, and that is what makes this
        // a gather of a blurred image rather than a blur of a sharp one:
        //
        //   * a tap that is BEHIND the pixel contributes when the pixel's own disc reaches it -- a defocused
        //     background pixel collects the sharp or defocused detail around it;
        //   * a tap that is IN FRONT contributes when the TAP's disc reaches the pixel -- a defocused foreground
        //     object spreads onto whatever it covers, which is how a bokeh ball lands on the background behind
        //     it instead of the background being dragged forward over the object.
        //
        // Weighing a front tap by the pixel's own disc -- the other choice, and the one a plain "reject anything
        // nearer" rule makes -- means a sharp pixel can never be covered by a defocused object in front of it,
        // which is the case the effect is most visible in. The two branches are exclusive and ordered by depth,
        // so no tap is counted twice and neither can pull against the other.
        if (centreRelative >= tapRelative && tapDistance >= centreDistance) {
            selfColour += colour;
            selfSamples += 1.0;
        } else if (tapRelative >= centreRelative && tapDistance <= centreDistance) {
            // A tap at least as defocused as the receiver, and no further from the eye: its disc covers this
            // pixel. A LESS defocused tap in front is a sharp object crossing the disc, and it is exactly what
            // must not be averaged away, so it contributes nothing at all rather than a little.
            //
            // The weight is inverse in the tap's own spread, so a tap standing for a small area is trusted more
            // than one standing for a large one -- near a silhouette that is the difference between a clean edge
            // and one smeared by whichever large-area tap happened to land there.
            // Both sides are already fractions of the span, so their ratio is the same number whether it is
            // read in pixels or in relative units. Dividing one of them by the span again mixed the two and
            // biased every weight by it.
            float ratio = max(1.0, centreRelative / max(tapRelative, 1.0e-4));
            float sampleWeight = ratio * ratio;
            sum += colour * sampleWeight;
            weightSum += sampleWeight;
        } else {
            // A tap this pixel's own disc cannot reach and whose disc does not cover it either: out of range,
            // and it holds the pixel's own colour up rather than pulling the average toward something else.
            selfWeight += centreWeight;
        }
    }
    // The pixel's own colour enters once, plus once for every tap that reached back to it, so a wide disc that
    // collected nothing does not fade to black and a pixel in focus keeps its own value exactly.
    float3 gathered = (sum + selfColour * (1.0 + selfWeight / selfSamples))
            / max(weightSum + selfSamples + selfWeight, 1.0e-5);

    // Temporal accumulation: one pass' taps never cover a wide disc, but consecutive passes cover different
    // parts of it, so the history is what turns a visible tap pattern into a smooth disc.
    // The guide holds "previous - current" in render pixels -- the direction the surface came FROM -- so the
    // pixel holding this surface last frame is this one PLUS the motion. Subtracting it reprojects to where the
    // surface is going instead of where it was, which is wrong by twice the motion: invisible while the camera
    // is still, and a shift that grows with movement as soon as it is not.
    float2 velocity = gMotion.SampleLevel(uv, 0.0).xy * float2(pc.velocityScaleX, pc.velocityScaleY);
    // The history lives one texel per pixel of this image, so its pixel is this one displaced by the motion --
    // no interpolation is involved and none would help, because the reprojection itself is the approximation.
    // A guide that has not been written yet reads back as whatever the allocation held, and a non-finite or
    // enormous displacement would sample anywhere in the frame or nowhere at all.
    if (!(dot(velocity, velocity) < 1.0e6)) {
        velocity = float2(0.0, 0.0);
    }
    velocity = clamp(velocity, float2(-size.x, -size.y), float2(size.x, size.y));
    float2 historyFloat = clamp(float2(pixel) + round(velocity), float2(0.0, 0.0),
            float2(size - int2(1, 1)));
    int2 historyPixel = int2(int(historyFloat.x), int(historyFloat.y));
    bool historyUsable = pc.historyWeight > 0.0 && pc.resetToken == pc.expectedResetToken;
    if (historyUsable) {
        float4 previous = historyImage.Load(historyPixel);
        // The history describes the surface that used to be at this pixel. When the guides have moved on --
        // a disocclusion, a fast pan -- that surface is not here any more and its colour would trail.
        //
        // Faded rather than accepted or rejected outright, for the same reason the occlusion weight ramps: the
        // decision depends on the reprojected depth, which varies across a silhouette, so a hard threshold makes
        // neighbouring pixels keep different amounts of history and the boundary between them is a visible line.
        // Fading means a surface that is drifting away from the one the history holds keeps contributing less of
        // it instead, and nothing snaps.
        float previousDistance = viewDistance(previous.w);
        float relative = abs(previousDistance - centreDistance) / max(centreDistance, 1.0e-3);
        // Trusted while it is close, gone once it is past the tolerance, and linear in between.
        float confidence = clamp((pc.depthTolerance - relative) / max(pc.depthTolerance * 0.5, 1.0e-4),
                0.0, 1.0);
        if (any(!isfinite(previous.rgb))) {
            confidence = 0.0;
        }
        gathered = lerp(gathered, previous.rgb,
                clamp(pc.historyWeight, 0.0, 0.95) * confidence);
    }
    gatheredImage[pixel] = float4(gathered, centreRawDepth);
}
