// Push-constant structs shared between the display-chain compute passes and
// layout/layout_probe.slang's build-time reflection probe (see GenerateShaderRecords.groovy).
// Each struct is imported into exactly one production shader AND probed here so its Java writer is
// generated from the compiler's reflected offsets, preventing Java/shader ABI drift when fields move.
//
// Each struct here needs a distinct name because reflection JSON parameters are named by struct type across the whole
// probe module, and per-entry-point SPIR-V allows only one push-constant block reachable from any
// single entry point, so the probe needs one small dedicated entry point per struct.

public struct ExposureHistPush {
    public uint stride;
    public float centerWeightSigma;
    public float centerWeightFloor;
    // The frame projection's [2][2] and [3][2]. The depth guide is stored as non-linear reversed-Z, so the
    // autofocus pass needs both to turn a sample back into a distance along the primary ray while it reduces
    // the centre box.
    public float projM22;
    public float projM32;
    // The autofocus window's centre used to be pushed from here, in the pixels of whatever image the exposure
    // pass happened to be metering. That image is the reconstruction backend's DISPLAY-resolution output while
    // the window is reduced over the RENDER-resolution depth guide, so the centre landed three quarters of the
    // way across the guide and the window measured the wrong part of the picture. The shader now centres the
    // window on the guide it is reading, which is the only size that is guaranteed to agree with it.
    public uint focusBoxSize;
    // Non-zero while the autofocus debug view is up: the autofocus pass then paints its own answer into the
    // depth guide, where the focus debug view reads it, instead of leaving the number in a log.
    public uint focusDiagnostic;
};

public struct ExposureResolvePush {
    public float key;
    public float minEv;
    public float maxEv;
    // Adaptation time constants, in seconds, applied in EV space (see exposure_resolve's smoothing
    // step). Named for what the SCENE did, not for which way the exposure multiplier moved: a darkening
    // scene needs MORE exposure, so adaptDarken is the branch where target > prev.
    public float adaptDarken;
    public float adaptBrighten;
    public float frameTimeSeconds;
    public float evBias;
    public float lowPercentile;
    public float highPercentile;
    public float skyWeightCap;
    public float curveScene0;
    public float curveCompensation0;
    public float curveScene1;
    public float curveCompensation1;
    public float curveScene2;
    public float curveCompensation2;
    public float curveScene3;
    public float curveCompensation3;
    public float emissiveWeightCap;
    // evOffset takes log2(metered stored luminance) to EV100:
    // RtSceneUnits.EV100_OFFSET - log2(preExposure). preExposure is the scalar raygen already
    // multiplied into the stored radiance (1.0 when pre-exposure is disabled), so the metered
    // buffer holds L*preExposure and the resolve divides it back out to recover absolute scene EV --
    // and to keep minEv/maxEv bounding ABSOLUTE exposure rather than the residual.
    public float evOffset;
    public float preExposure;
    // resetSeq invalidates GPU exposure history without a host write or fence.
    public uint resetSeq;
    // Physical exposure the camera's aperture and shutter time imply, relative to the reference combination:
    // 2^(referenceEv100 - ev100(N, t)), in linear light. Folded into the same exposure image the automatic
    // controller writes, so the lens darkens or brightens the frame exactly as a real one would and both
    // control loops act on one number.
    public float physicalScale;
    // Non-zero selects manual exposure, and carries its absolute exposure multiplier. The controller still
    // meters and keeps its state current in that mode, but the image uses this value instead of the resolved
    // one, so switching exposure modes cannot make the picture jump.
    public float manualScale;
    // Time in seconds for the focal plane to cover half the distance to the metered depth. An autofocus that
    // followed the measurement directly would hand the image a focal plane that jumps whenever a nearer
    // surface crosses the centre of the frame, which reads as a snap rather than as a lens racking.
    public float focusHalflifeSeconds;
};

// Persistent auto-exposure controller state. This shared definition is also reflected into
// ExposureStateData.java so host reads and initialization follow Slang's Std430 layout.
public struct ExposureState {
    public float previous;
    public uint initialized;
    public float evScene;
    public float evTarget;
    public float evApplied;
    public float clipLowFrac;
    public float clipHighFrac;
    public uint resetSeq;
    public float meteringSkyScale;
    public float meteringSkyFrac;
    public float curveCompensation;
    public float effectiveSlope;
    public float meteringEmissiveScale;
    public float meteringEmissiveFrac;
    // Auto-focus distance in blocks along the primary ray, solved by the histogram pass. 0 means no
    // non-sky surface was metered, which the host treats as "hold the previous focus".
    public float focusDistance;
    // The SMOOTHED centre depth, which is what the autofocus actually focuses on, and the reason it lives here
    // rather than beside the measurement: the smoothing reads the value it wrote last frame, and the measurement
    // scratch is cleared every frame by the histogram. Keeping the history in the persistent state is what makes
    // the filter an exponential approach instead of a running average.
    public float smoothCenterDepth;
    // What the autofocus window measured THIS frame, before the exponential approach. Published separately
    // because the two answer different questions: a focal plane that is wrong is a measurement problem, and one
    // that is merely late is a smoothing problem, and the smoothed value alone cannot tell them apart.
    public float measuredFocus;
};

public struct DisplayPush {
    public int hdrEnabled;      // 0 = SDR only, 1 = also write the PQ HDR image
    public float lutSize;       // toneLut/hdrToneLut texels per axis; both LUTs share one size (asserted at load)
    public float gamma;         // post-display artistic gamma; below 1 brightens midtones
    public float hdrPeakNits;   // mastering peak baked into the currently bound HDR ACES LUT
    public int lookEnabled;     // 0 = identity, 1 = apply the scene-referred ACES look LUT
    public float lookLutSize;   // lookLut texels per axis (independent of the output-transform LUT size)
    // Scene-referred blurred highlight signal added before the LMT. The bloom pyramid's level 0 holds the
    // SUM of every band, so RtComposite folds the 1/levelCount normalisation into this value: the authored
    // look-package strength then means the same thing whichever level count the resolution supports.
    public float bloomStrength;
};

// Display-resolution motion blur, applied to the reconstructed scene before the display transform.
// Streaks come from the traced motion vectors, so the shutter interval is the only artistic quantity here.
public struct MotionBlurPush {
    public int displayWidth;
    public int displayHeight;
    public int samples;            // taps averaged along the streak, endpoints included
    // Render-pixel motion -> display-pixel motion. The guides are render resolution while this pass runs at
    // display resolution, so the velocity has to be scaled by each axis' ratio before it is used as a
    // display-space pixel offset.
    public float velocityScaleX;
    public float velocityScaleY;
    public float projM22;          // frame projection [2][2], for the reversed-Z depth guide
    public float projM32;          // frame projection [3][2]
    public float shutterFraction;  // 0 = closed (no streak), 1 = the whole frame interval
    public float maxRadiusPixels;  // longest streak, in display pixels
    public float depthTolerance;   // relative view-distance difference at which a tap stops contributing
};

// Depth-of-field gather, at half display resolution. The trace covers a small disc of confusion with its own
// aperture samples; this pass supplies the rest of it by gathering, so the two radii below are what divides the
// effect between them and the camera's aperture is their sum.
public struct DepthOfFieldPush {
    public int gatherWidth;
    public int gatherHeight;
    // The largest disc the renderer draws -- the circle of confusion -- and the part of it the trace's own
    // aperture samples have already covered. Both are radii in DISPLAY pixels, because that is the space the
    // effect is specified in: a blur is only meaningful against the picture it is in, and the lens itself is
    // expressed against the frame height. The gather runs at a fraction of that, so it converts with
    // `displayToGather` before it uses either as a distance in its own texels -- applying a display-space radius
    // directly in gather texels made every disc three times its intended size.
    public float targetRadiusPixels;
    public float tracedRadiusPixels;
    public float focusDistance;          // focal plane in blocks, scene units
    // The lens itself, in the units the shader works in: `depthScale` is the aperture's radius expressed in
    // DISPLAY pixels, and `focalLength` is the focal length in blocks. Together they are the whole thin lens,
    // and the falloff puts a surface at Z on a disc of
    //
    //     (f/N) * |v(Zf) - v(Z)| / v(Z),   v(Z) = 1 / (1/f - 1/Z)
    //
    // measured on the film and then projected onto the frame. The image distances are what a lens actually
    // equalises, so this is exact at every distance the renderer can produce -- including focus distances close
    // enough that the focal length is a real fraction of them, which is where the usual `f^2` approximation
    // drifts. Focus at infinity is the same expression with v(Zf) = f.
    public float depthScale;
    public float focalLength;
    // The gather image's size as a fraction of the frame it is blurring: DISPLAY pixels to gather texels. The
    // gather is `renderSize / 2` while the display is the window, so the two differ by whatever the
    // reconstruction backend chose -- 1/3 under DLSS-RR at quality, which is exactly the factor a radius used in
    // the wrong one of the two would be out by.
    public float displayToGather;
    public float padding0;
    // A pixel whose residual disc is below this keeps the sharp image. Without it the filter would add its own
    // noise to the in-focus parts of the frame, which are the parts that are supposed to be exact.
    public float gatherMinRadiusPixels;
    // Occlusion tolerance as a ratio: a tap is hidden when it is nearer than the receiver by more than this.
    public float foregroundTolerance;
    public float depthTolerance;         // relative view-distance difference that invalidates history
    public float historyWeight;          // 0 disables the history entirely
    public uint expectedResetToken;      // history is discarded unless it matches this frame's token
    public uint resetToken;
    public uint frameIndex;              // advances the tap pattern
    public uint tapCount;
    public float velocityScaleX;         // render-pixel motion -> gather-pixel motion
    public float velocityScaleY;
    public float projM22;                // frame projection [2][2], for the reversed-Z depth guide
    public float projM32;                // frame projection [3][2]
    // Which of the pass' stages this dispatch is: 0 gathers, 1 blurs across, 2 blurs down. They share one
    // pipeline layout because they share one descriptor set, and the stage is a push field rather than a
    // second pipeline so the three dispatches cannot drift apart in their bindings.
    public uint stage;
    // Half-width of the smoothing the two blur stages apply, in their own texels.
    public uint blurRadius;
};

public struct BloomPush {
    public int mode;            // 0 prefilter+downsample, 1 downsample, 2 tent upsample (accumulate)
    public float threshold;     // exposed scene-linear ACEScg luminance
    public float softKnee;      // absolute exposed scene-linear knee width
    public float radius;        // upsample tent radius in SOURCE texels; resolution-independent
};

public struct DebugPresentPush {
    public uint debugView;     // 1..11; this pass is only dispatched for debugView != 0
    public float centerWeightSigma;
    public float centerWeightFloor;
    // Sensor height, for the depth-of-field debug views: the lens is scaled by the render height over this, so
    // it is what turns a lens radius in scene units into the disc in pixels the effect is actually made of.
    public float sensorHeightMetres;
    public float focusDistanceBlocks;
    // Radius of the camera's entrance pupil in scene units. A point at infinity then lands as a disc of
    // lensRadius * renderHeight / sensorHeight pixels, and a point at Z keeps |Zf / Z - 1| of it -- which is
    // exactly the circle of confusion the trace and the gather divide between them.
    public float lensRadiusBlocks;
    public float projM22;
    public float projM32;
};
