// Autofocus: the distance of the surface under the crosshair.
//
// The rule is one sample at the centre texel, followed by an exponential approach with a half-life: a single
// centre sample is the cheapest measurement of what the player is aiming at, and the exponential approach is
// what keeps the plane from snapping when that sample moves. The smoothing lives in exposure_resolve, which
// owns the state that survives a frame.
//
// The crosshair texel decides rather than the nearest surface in a box, because the NEAREST surface is biased
// towards whatever sticks out furthest into the frame: a window straddling a ledge, a railing, or the rim of a
// mesa contains the whole depth range, and the nearest of it is the foreground that happens to clip the corner
// -- so holding the crosshair on a wall focused on something else entirely. The texel the player aims with is
// the surface they mean.
//
// The box survives only as a fallback for a centre texel that reports no distance, which is sky or the
// horizon: a single sample there has nothing to say, and reporting nothing would freeze the plane wherever it
// last was. It is a fixed 32x32 rather than a share of the frame so the fallback stays a point rather than a
// search that reaches into the middle distance.
//
//     currCenterDepth   = <the crosshair's own texel, or the box's nearest surface if that texel is sky>
//     fadeFactor        = exp(log(0.5) * frameTime * 10 / centerDepthHalflife)   [in exposure_resolve]
//     smoothCenterDepth = mix(currCenterDepth, prevCenterDepth, fadeFactor)      [in exposure_resolve]
//
// It runs as its own pass because it is a MINIMUM and not a sum: folding it into the histogram's weighted
// average would be a different measurement wearing the same name.
import display_common;
import bindings;

[[vk::push_constant]] ExposureHistPush pc;

static const uint GROUP = 32u;
static const uint GROUP_TEXELS = 1024u;
static const float NO_SURFACE = 3.0e38;
groupshared float threadMinima[GROUP_TEXELS];

// Depth is non-linear and reversed-Z, and the forward projection's w is the distance to the hit, so
// depth = -m22 + m32 / d holds with d measured along the ray. Inverting gives d = m32 / (depth + m22). The
// acceptance is relative to the projection's scale, because an absolute epsilon would reject the far field,
// where the depth approaches zero and the denominator shrinks to m22.
float viewDistance(float depth) {
    float denominator = depth + pc.projM22;
    float acceptance = 1.0e-7 * max(1.0, abs(pc.projM22) + abs(pc.projM32));
    if (!(depth > 0.0) || abs(denominator) < acceptance) {
        return 0.0; // sky, or a depth with no usable distance: not an autofocus target
    }
    float distance = pc.projM32 / denominator;
    return distance > 0.0 ? distance : 0.0;
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 threadId : SV_GroupThreadID, uint groupIndex : SV_GroupIndex) {
    uint width, height;
    depthImage.GetDimensions(width, height);
    int2 limit = int2(int(width), int(height)) - 1;
    // The window is centred on the frame, so its top-left corner is half of it back from the centre the host
    // pushed, and one thread measures one texel of it.
    // The window is centred on the DEPTH GUIDE, not on the frame size the host knows. Those are different
    // images: the guide is traced at the render resolution, while the reconstruction backend's output -- which is
    // what the exposure pass meters, and therefore what the host called "the frame" when it pushed a centre --
    // is a full display-resolution image. Taking the centre from the latter put the window three quarters of the
    // way across the former: it sampled the upper right of the picture and reported whatever it found there as
    // what the crosshair was on.
    //
    // The centre of the guide IS the crosshair. The primary ray for the middle texel is cast through the middle
    // of the projection whatever the reconstruction scale is, so the guide's own size is the authority.
    int2 centre = int2(int(width / 2u), int(height / 2u));
    int half = int(GROUP / 2u);
    int2 origin = centre - int2(half, half);
    int2 pix = clamp(origin + int2(int(threadId.x), int(threadId.y)), int2(0, 0), limit);

    // A texel that measured nothing -- sky, or a depth with no usable distance -- must not win the minimum, and
    // the sentinel is larger than any real distance the renderer can produce.
    // The crosshair's OWN texel decides the focus, because that is what the player is aiming at. The window is
    // only a fallback for the texel that has no distance at all -- sky, or a pixel on the horizon -- where a
    // single sample has nothing to report and would otherwise freeze the plane.
    float centreDistance = viewDistance(depthImage[clamp(centre, int2(0, 0), limit)]);
    float distance = viewDistance(depthImage[pix]);
    threadMinima[groupIndex] = distance > 0.0 ? distance : NO_SURFACE;
    GroupMemoryBarrierWithGroupSync();

    // A tree reduction over the group's threads, in shared memory. One group owns the whole window, so there is
    // nothing to combine with anyone and no atomics are needed.
    for (uint stride = GROUP_TEXELS / 2u; stride > 0u; stride >>= 1u) {
        if (groupIndex < stride) {
            threadMinima[groupIndex] = min(threadMinima[groupIndex], threadMinima[groupIndex + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (groupIndex == 0u) {
        // The crosshair decides whenever it has anything to say, and the window speaks for it when it does not.
        //
        // The fallback is not optional. A centre texel reports nothing for sky, for the horizon, and for any
        // surface whose depth has no usable distance, and when that was allowed to veto the frame the focus plane
        // simply STOPPED -- it stayed wherever it last was, so the player could aim at a wall two blocks away and
        // watch a ninety-block plane refuse to move. The window is a point rather than a search, so its nearest
        // surface is still the subject the crosshair is on rather than the middle distance behind it.
        float nearest = centreDistance > 0.0 ? centreDistance : threadMinima[0];
        bool found = nearest < NO_SURFACE;
        // Lane 0 is the distance in blocks and lane 1 its weight, which the resolve reads; the resolve applies
        // the exponential approach and publishes the result. A window that found nothing reports zero weight,
        // and the resolve holds the previous focus rather than treating that as a plane at the lens.
        focusScratch[0] = found ? nearest : 0.0;
        focusScratch[1] = found ? 1.0 : 0.0;

        // A diagnostic that cannot be argued with: the window paints ITS OWN answer into the depth guide, in a
        // square exactly the size of the window. Whatever colour appears there is the number the autofocus found,
        // and how far it appears from the crosshair is where the window actually is -- which separates "the
        // measurement is wrong" from "the window is not where the code says it is".
        //   magenta = found nothing, blue = under 2 blocks, white = 2-10, orange = 10-40, red = over 40
        if (pc.focusDiagnostic != 0u) {
            float tint = !found ? 1.0
                    : nearest < 2.0 ? 0.05
                    : nearest < 10.0 ? 0.35
                    : nearest < 40.0 ? 0.65
                    : 0.95;
            // A block the size of the window, plus a marker at the CENTRE the host pushed. The marker is what
            // says where the code believes the middle of the frame is: if it is not under the crosshair, the
            // centre is what is wrong, and no amount of work on the sampling rule would have found that.
            for (uint y = 0u; y < GROUP; ++y) {
                for (uint x = 0u; x < GROUP; ++x) {
                    depthImage[clamp(origin + int2(int(x), int(y)), int2(0, 0), limit)] = tint;
                }
            }
            for (int i = -128; i <= 128; ++i) {
                depthImage[clamp(centre + int2(i, 0), int2(0, 0), limit)] = 1.0;
                depthImage[clamp(centre + int2(0, i), int2(0, 0), limit)] = 0.0;
            }
        }
    }
}
