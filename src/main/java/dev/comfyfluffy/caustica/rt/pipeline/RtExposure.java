package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtCamera;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtGpuExecutor;
import dev.comfyfluffy.caustica.rt.RtLookPackage;
import dev.comfyfluffy.caustica.rt.RtSceneUnits;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.ExposureStateData;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkBufferMemoryBarrier;
import org.lwjgl.vulkan.VkClearColorValue;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkImageSubresourceRange;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Owns the display exposure value shared by the RT compositor's display-mapping passes, and the virtual
 * camera's aperture, shutter time and focus distance that feed it.
 *
 * <p>Both live here because they are one quantity split across two controllers. The automatic controller
 * solves the exposure the metered scene needs; the camera's aperture and shutter time decide how much of
 * that exposure the lens actually delivers, and the difference is a scale on the same 1x1 exposure image.
 * Keeping them in one place is what makes "auto" mean the same thing for both: the automatic controller
 * converges on the target brightness while the auto aperture and shutter snap to the nearest ladder rungs
 * that realize it.
 */
public final class RtExposure {
    private RtImage image;
    private RtBuffer histogram;
    private RtBuffer focus;
    private RtBuffer state;
    private ReadbackSlot[] stateReadbacks;
    private int stateReadbackIndex = -1;
    private ReadbackSlot pendingStateReadback;
    private ExposureStateData completedState;
    /**
     * What the autofocus window measured this frame, before the exponential approach, in blocks. Published for
     * the depth-of-field diagnostic: a focal plane that is wrong is a measurement problem and one that is merely
     * late is a smoothing problem, and the latched value alone cannot tell them apart.
     */
    /** Non-zero while the autofocus debug view is up; the pass then paints its answer into the depth guide. */
    private boolean focusDiagnostic;

    public void setFocusDiagnostic(boolean enabled) {
        focusDiagnostic = enabled;
    }

    public float measuredFocusBlocks() {
        return completedState != null ? completedState.measuredFocus() : 0.0f;
    }

    /** Last measured autofocus distance in blocks, and the traced lens radius, for the throttled diagnostic. */
    private float meteredFocusBlocks;
    private float tracedRadiusBlocks;
    private RtExposurePipeline pipeline;
    private boolean logged;
    private long lastFrameNanos;
    private long lastDiagLogNanos;
    private String cachedCurveSpec;
    private ExposureCurve cachedCurve;
    private boolean resetRequested = true;
    private int resetSequence;
    /** This frame's latched pre-exposure; see {@link #beginFrame(RtGpuExecutor.GraphicsUseWaiter, Matrix4fc)}. */
    private float framePreExposure = 1.0f;
    /** Projection terms the histogram pass needs to turn the reversed-Z depth guide into a distance. */
    private float frameProjectionM22 = -1.0f;
    private float frameProjectionM32 = 0.0f;
    /** This frame's latched lens; see {@link #optics()}. */
    private RtCamera.Frame optics = new RtCamera.Frame(2.8f, 1.0f / 60.0f, 8.0f,
            RtCamera.SENSOR_HEIGHT_METRES, 0.01714f, 0.00612f, 1.0f, 1080);

    private static final long DIAG_LOG_INTERVAL_NANOS = 1_000_000_000L;
    private static final int STATE_READBACK_RING = 6;

    private static final class ReadbackSlot {
        final RtBuffer buffer;
        final RtGpuExecutor.TrackedGraphicsUse graphicsUse = new RtGpuExecutor.TrackedGraphicsUse();
        boolean valid;
        int resetSequence;

        ReadbackSlot(RtBuffer buffer) {
            this.buffer = buffer;
        }
    }

    public RtImage image() {
        return image;
    }

    public boolean ready() {
        return image != null;
    }

    public RtBuffer stateBuffer() {
        return state;
    }

    /**
     * This frame's lens: aperture, shutter time, focus distance and the exposure scale they imply. Latched
     * once per frame alongside {@link #preExposure()} so the world push, the metering push and the exposure
     * image all describe the same camera.
     */
    public RtCamera.Frame optics() {
        return optics;
    }

    /**
     * What the lens contributes to the display exposure this frame.
     *
     * <p>The exposure controller's target multiplies the stored trace, and the stored trace is what the lens was
     * pointed for, so the two are already balanced: when the auto programme chose the pair, the lens scale is
     * deliberately left at 1.0. The controller's residual then absorbs the small difference between the rung the
     * programme could reach and the exposure the scene asked for, which is exactly the job it is built for.
     *
     * <p>A manual pair is different: it is the player overriding the exposure, so its true scale is applied and
     * the image brightens or darkens as the aperture and shutter dictate. Multiplying it in on top of an
     * exposure the controller had already settled is the intended behaviour there, and the controller does not
     * fight it because it never meters the display image.
     */
    private float lensExposureScale() {
        boolean autoProgrammeOwnsThePair = "auto".equals(CausticaConfig.Rt.Camera.APERTURE_MODE.get())
                || "auto".equals(CausticaConfig.Rt.Camera.SHUTTER_MODE.get());
        return autoProgrammeOwnsThePair ? 1.0f : optics.physicalExposureScale();
    }

    /** Immutable exposure values attached to a residual-exposed EXR capture. */
    public record CaptureMetadata(
            float preExposure,
            float residualExposure,
            float absoluteExposure,
            String mode,
            float evScene,
            float evTarget,
            float evApplied,
            float physicalExposureScale
    ) {
    }

    /**
     * Snapshot the controller after the capture copy has completed.
     *
     * <p>{@code residualExposure} is read from the same 1x1 GPU image that the display shader samples.
     * The absolute multiplier can therefore be reconstructed exactly as
     * {@code preExposure * residualExposure}, even when auto exposure corrected a stale prediction.
     */
    public CaptureMetadata captureMetadata(float residualExposure) {
        if (!Float.isFinite(residualExposure) || residualExposure <= 0.0f) {
            throw new IllegalArgumentException("Invalid residual exposure " + residualExposure);
        }
        Mode currentMode = mode();
        float pre = preExposure();
        float absolute = pre * residualExposure * lensExposureScale();
        if (currentMode != Mode.AUTO || state == null || state.mapped == 0L) {
            float ev = manualEv();
            return new CaptureMetadata(pre, residualExposure, absolute, currentMode.configName,
                    Float.NaN, ev, ev, lensExposureScale());
        }
        state.invalidate();
        ExposureStateData snapshot = readState();
        return new CaptureMetadata(pre, residualExposure, absolute, currentMode.configName,
                snapshot.evScene(), snapshot.evTarget(), snapshot.evApplied(), lensExposureScale());
    }

    public void ensureResources(RtContext ctx) {
        if (image == null) {
            image = ctx.createStorageImage(1, 1, VK10.VK_FORMAT_R32_SFLOAT, "display exposure");
        }
        // The state buffer is read by the debug pass and written by the controller, and the histogram,
        // autofocus scratch and both pipelines are needed whichever exposure mode is selected: the controller
        // always meters, because its metered scene brightness is what drives the auto aperture, auto shutter
        // and autofocus programmes in manual mode too.
        if (state == null) {
            state = ctx.createBuffer(ExposureStateData.BYTE_SIZE,
                    VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    true, "exposure state");
            resetAutoHistory();
        }
        if (stateReadbacks == null) {
            ReadbackSlot[] created = new ReadbackSlot[STATE_READBACK_RING];
            try {
                for (int i = 0; i < created.length; i++) {
                    created[i] = new ReadbackSlot(ctx.createReadbackBuffer(
                            ExposureStateData.BYTE_SIZE, "exposure state readback " + i));
                }
            } catch (Throwable t) {
                for (ReadbackSlot slot : created) {
                    if (slot != null) {
                        slot.buffer.destroy();
                    }
                }
                throw t;
            }
            stateReadbacks = created;
        }
        if (histogram == null) {
            // Separate ordinary-surface/sky/emissive histograms let resolve enforce both
            // population caps exactly without a second full-image dispatch.
            histogram = ctx.createBuffer(768L * Integer.BYTES,
                    VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT, false,
                    "exposure histogram");
        }
        if (focus == null) {
            // Lane 0 is the autofocus distance and lane 1 its weight: cleared by the histogram pass, measured by
            // the autofocus pass, read by the resolve. The lanes past those carry the window's own geometry --
            // the guide's size, the centre it was given and the corner it resolved to -- which is the only way to
            // tell "the measurement is wrong" from "the window is not where the code believes it is".
            focus = ctx.createBuffer(10L * Float.BYTES,
                    VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "exposure focus");
        }
        if (pipeline == null) {
            pipeline = RtExposurePipeline.create(ctx);
        }
        logOnce();
    }

    /**
     * Meters the frame and publishes the exposure image.
     *
     * <p>The histogram and the resolve run in both exposure modes. Manual mode ignores the controller's
     * resolved value and writes its own, but the metered scene still has to reach the host: it is the input
     * to the auto aperture, auto shutter and autofocus programmes, and those are lens decisions rather than
     * exposure decisions, so they follow the scene whatever the exposure mode is.
     */
    public void record(RtContext ctx, VkCommandBuffer cmd, MemoryStack stack,
                       RtImage traceColor, RtImage guideDepth, RtImage guideAlbedo) {
        if (image == null) {
            throw new IllegalStateException("RT exposure image not created");
        }
        recordAuto(ctx, cmd, stack, traceColor, guideDepth, guideAlbedo);
    }

    public void destroy() {
        if (pipeline != null) {
            pipeline.destroy();
            pipeline = null;
        }
        if (histogram != null) {
            histogram.destroy();
            histogram = null;
        }
        if (focus != null) {
            focus.destroy();
            focus = null;
        }
        if (state != null) {
            state.destroy();
            state = null;
        }
        if (stateReadbacks != null) {
            for (ReadbackSlot slot : stateReadbacks) {
                slot.buffer.destroy();
            }
            stateReadbacks = null;
        }
        if (image != null) {
            image.destroy();
            image = null;
        }
        resetRequested = true;
        resetSequence = 0;
        stateReadbackIndex = -1;
        pendingStateReadback = null;
        completedState = null;
        framePreExposure = 1.0f;
        RtCamera.reset();
    }

    // Manual mode's exposure scale, also used as the auto-history seed (resetAutoHistory) so the very
    // first auto-exposure frame starts from the dialed-in EV bias instead of a bare 1.0.
    private float manualExposureScale() {
        return CausticaConfig.Rt.Exposure.clampScale((float) Math.pow(2.0, manualEv()));
    }

    private void recordAuto(RtContext ctx, VkCommandBuffer cmd, MemoryStack stack,
                            RtImage traceColor, RtImage guideDepth, RtImage guideAlbedo) {
        if (pipeline == null || histogram == null || focus == null || state == null) {
            throw new IllegalStateException("RT auto exposure resources not created");
        }
        pipeline.setResources(traceColor.view, guideDepth.view, guideAlbedo.view,
                histogram, focus, image.view, state);
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "exposure histogram clear")) {
            VK10.vkCmdFillBuffer(cmd, histogram.handle, 0, histogram.size, 0);
        }
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
        AutoConfig config = autoConfig();
        pipeline.dispatchHistogram(cmd, traceColor.width, traceColor.height, config);
        // The autofocus point runs after the histogram, which is what clears the scratch it reports into, and
        // before the resolve, which is what reads it.
        pipeline.dispatchAutofocus(cmd, traceColor.width, traceColor.height, config, focusDiagnostic);
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
        pipeline.dispatchResolve(cmd, config, frameTimeSeconds());
        logDiagnosticsIfDue();
    }

    /**
     * Copies the GPU-owned controller state into this frame's guarded host-readback slot. The slot is not
     * consumed until its graphics timeline value completes, so the host never races the live storage buffer.
     */
    public void recordStateReadback(VkCommandBuffer cmd, MemoryStack stack) {
        if (pendingStateReadback == null) {
            return;
        }
        VkBufferMemoryBarrier.Buffer toTransfer = VkBufferMemoryBarrier.calloc(1, stack);
        toTransfer.get(0).sType$Default()
                .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK10.VK_ACCESS_TRANSFER_READ_BIT)
                .srcQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK10.VK_QUEUE_FAMILY_IGNORED)
                .buffer(state.handle)
                .offset(0L)
                .size(ExposureStateData.BYTE_SIZE);
        VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, null, toTransfer, null);

        VkBufferCopy.Buffer copy = VkBufferCopy.calloc(1, stack)
                .srcOffset(0L).dstOffset(0L).size(ExposureStateData.BYTE_SIZE);
        VK10.vkCmdCopyBuffer(cmd, state.handle, pendingStateReadback.buffer.handle, copy);
    }

    /** Attach the readback copy only after the command buffer has been accepted for frame submission. */
    public void markStateReadbackUse(RtGpuExecutor.GraphicsUse graphicsUse) {
        if (pendingStateReadback == null) {
            return;
        }
        pendingStateReadback.resetSequence = resetSequence;
        pendingStateReadback.valid = true;
        pendingStateReadback.graphicsUse.mark(graphicsUse);
        pendingStateReadback = null;
    }

    /**
     * Throttled log of the controller's internal EVs, gated
     * behind the frame-stats toggle since that's the existing "I want renderer internals" switch.
     * Uses the latest completed timeline-guarded readback. It can be a few frames stale without racing
     * the GPU, which is sufficient for diagnostics.
     */
    private void logDiagnosticsIfDue() {
        if (!CausticaConfig.Rt.FrameStats.ENABLED.value() || completedState == null) {
            return;
        }
        long now = System.nanoTime();
        if (lastDiagLogNanos != 0L && now - lastDiagLogNanos < DIAG_LOG_INTERVAL_NANOS) {
            return;
        }
        lastDiagLogNanos = now;
        ExposureStateData snapshot = completedState;
        float evScene = snapshot.evScene();
        float evTarget = snapshot.evTarget();
        float evApplied = snapshot.evApplied();
        float clipLowFrac = snapshot.clipLowFrac();
        float clipHighFrac = snapshot.clipHighFrac();
        float skyScale = snapshot.meteringSkyScale();
        float skyFrac = snapshot.meteringSkyFrac();
        float curveCompensation = snapshot.curveCompensation();
        float effectiveSlope = snapshot.effectiveSlope();
        float emissiveScale = snapshot.meteringEmissiveScale();
        float emissiveFrac = snapshot.meteringEmissiveFrac();
        AutoConfig cfg = autoConfig();
        boolean pinnedLow = evTarget <= cfg.minEv() + 0.01f;
        boolean pinnedHigh = evTarget >= cfg.maxEv() - 0.01f;
        // evScene is EV100; evTarget/evApplied are log2 of the absolute
        // exposure multiplier, i.e. pre-exposure already divided back out, so they stay comparable
        // across frames regardless of what preExposure happened to be.
        CausticaMod.LOGGER.info(
                "RT exposure diag: evScene(EV100)={} evTarget={}{} evApplied={} preExposure={} "
                        + "aperture=f/{} shutter=1/{} physical={} focus={} "
                        + "clipLow={}% clipHigh={}% skyScale={} skyWeight={}% emissiveScale={} "
                        + "emissiveWeight={}% curveComp={} effectiveSlope={}",
                fmt(evScene), fmt(evTarget), pinnedLow ? " (at minEv clamp)" : pinnedHigh ? " (at maxEv clamp)" : "",
                fmt(evApplied), fmt(preExposure()),
                fmt(optics.aperture()), fmt(1.0f / Math.max(optics.shutterSeconds(), 1.0e-6f)),
                fmt(optics.physicalExposureScale()), fmt(optics.focusDistanceBlocks()),
                fmt(clipLowFrac * 100.0f), fmt(clipHighFrac * 100.0f),
                fmt(skyScale), fmt(skyFrac * 100.0f), fmt(emissiveScale),
                fmt(emissiveFrac * 100.0f), fmt(curveCompensation), fmt(effectiveSlope));
    }

    private static String fmt(float v) {
        return String.format(java.util.Locale.ROOT, "%.2f", v);
    }

    /**
     * One-line summary for the F3 debug screen ({@code RtExposureDebugEntry}). Unlike
     * {@link #logDiagnosticsIfDue()} this is not throttled and not gated on
     * {@code CausticaConfig.Rt.FrameStats.ENABLED} -- F3 only calls it once the player has enabled
     * that entry, and the game's own render cadence is throttle enough. Returns {@code null} when
     * there is nothing meaningful to show yet (state buffer not created).
     */
    public String debugSummaryLine() {
        if (state == null) {
            return null;
        }
        String lens = String.format(java.util.Locale.ROOT, "f/%s 1/%s s focus %s lens x%s",
                fmt(optics.aperture()), fmt(1.0f / Math.max(optics.shutterSeconds(), 1.0e-6f)),
                Float.isFinite(optics.focusDistanceBlocks())
                        ? fmt(optics.focusDistanceBlocks()) + " m" : "inf",
                fmt(optics.physicalExposureScale()));
        if (mode() != Mode.AUTO) {
            return String.format(java.util.Locale.ROOT, "Exposure: manual %s EV, %s", fmt(manualEv()), lens);
        }
        ExposureStateData snapshot = completedState;
        if (snapshot == null) {
            return "Exposure: " + lens;
        }
        float evScene = snapshot.evScene();
        float evTarget = snapshot.evTarget();
        float evApplied = snapshot.evApplied();
        AutoConfig cfg = autoConfig();
        String clamp = evTarget <= cfg.minEv() + 0.01f ? " (min clamp)"
                : evTarget >= cfg.maxEv() - 0.01f ? " (max clamp)" : "";
        return String.format(java.util.Locale.ROOT, "Exposure: EV100 %s, applied %s EV%s, %s",
                fmt(evScene), fmt(evApplied), clamp, lens);
    }

    private float frameTimeSeconds() {
        long now = System.nanoTime();
        float dt = lastFrameNanos == 0L ? 1.0f / 60.0f
                : Math.clamp((now - lastFrameNanos) / 1_000_000_000.0f, 1.0f / 240.0f, 0.25f);
        lastFrameNanos = now;
        return dt;
    }

    private void resetAutoHistory() {
        if (state == null || state.mapped == 0L) {
            return;
        }
        // Under physical units this seed can be ~15 EV off for an auto-mode daylight scene, since
        // manual-ev defaults to 0. That is a two-frame transient, not a bug: initialized == 0 makes the
        // resolve snap to its computed target rather than smooth toward it, and the frame after that
        // meters against a preExposure derived from it. Deliberately not special-cased -- a seed that
        // guessed at scene brightness would be a second, unowned exposure model.
        new ExposureStateData(
                manualExposureScale(), 0,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0,
                1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f
        ).write(stateDataBuffer());
        state.flush(0L, ExposureStateData.BYTE_SIZE);
        lastFrameNanos = 0L;
        lastDiagLogNanos = 0L;
    }

    private void logOnce() {
        if (logged) {
            return;
        }
        logged = true;
        Mode mode = mode();
        AutoConfig autoConfig = autoConfig();
        String exposureText = mode == Mode.AUTO
                ? "auto(key=" + autoConfig.key + ", minEv=" + autoConfig.minEv + ", maxEv=" + autoConfig.maxEv
                + ", adaptDarken=" + autoConfig.adaptDarken + ", adaptBrighten=" + autoConfig.adaptBrighten
                + ", evBias=" + autoConfig.evBias + ", percentiles=" + autoConfig.lowPercentile
                + ".." + autoConfig.highPercentile + ", stride=" + autoConfig.stride
                + ", centerWeight=" + autoConfig.centerWeightSigma + "/" + autoConfig.centerWeightFloor
                + ", skyCap=" + autoConfig.skyWeightCap
                + ", emissiveCap=" + autoConfig.emissiveWeightCap
                + ", curve=" + CausticaConfig.Rt.Exposure.curve() + ")"
                : Float.toString(manualExposureScale());
        CausticaMod.LOGGER.info("RT display exposure: mode={}, exposure={}, "
                        + "tonemap=aces2.0(lookPackage={},gamma={}), DLSS-RR exposure=NGX auto",
                mode.configName, exposureText, RtLookPackage.current().id(),
                CausticaConfig.Rt.Tonemap.GAMMA.value());
    }

    private static Mode mode() {
        return Mode.parse(CausticaConfig.Rt.Exposure.MODE.get());
    }

    private static float manualEv() {
        return CausticaConfig.Rt.Exposure.MANUAL_EV.value();
    }

    private AutoConfig autoConfig() {
        return new AutoConfig(
                CausticaConfig.Rt.Exposure.KEY.value(),
                CausticaConfig.Rt.Exposure.minEv(),
                CausticaConfig.Rt.Exposure.maxEv(),
                CausticaConfig.Rt.Exposure.ADAPT_DARKEN.value(),
                CausticaConfig.Rt.Exposure.ADAPT_BRIGHTEN.value(),
                manualEv(),
                CausticaConfig.Rt.Exposure.LOW_PERCENTILE.value(),
                CausticaConfig.Rt.Exposure.HIGH_PERCENTILE.value(),
                CausticaConfig.Rt.Exposure.STRIDE.value(),
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_SIGMA.value(),
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_FLOOR.value(),
                CausticaConfig.Rt.Exposure.SKY_WEIGHT_CAP.value(),
                CausticaConfig.Rt.Exposure.EMISSIVE_WEIGHT_CAP.value(),
                curveConfig(),
                preExposure(),
                resetSequence,
                lensExposureScale(),
                mode() == Mode.AUTO ? 0.0f : manualExposureScale(),
                frameProjectionM22,
                frameProjectionM32,
                RtCamera.FOCUS_HALFLIFE_SECONDS);
    }

    /**
     * Latches this frame's pre-exposure and lens. MUST be called once per frame before the world push
     * constants are written, and must not be re-latched afterwards.
     *
     * <p>The raygen multiply and the resolve's divide have to use the <em>same</em> value or they
     * stop cancelling and the frame comes out mis-scaled. Both read {@link #preExposure()}, but at
     * different points in CPU time. The completed readback can be several frames old, so latching once
     * ensures both consumers use one prediction; the residual absorbs whatever it failed to predict.
     *
     * <p>The lens is latched here for the same reason: the world push carries its aperture and focus
     * distance, the display pass carries the exposure scale they imply, and a frame in which those two
     * disagreed would focus one image and expose another.
     *
     * @param projection   this frame's projection matrix, whose vertical FOV sets the focal length
     * @param renderHeight the height the trace runs at, which sizes the blurred-circle cap
     */
    public void beginFrame(RtGpuExecutor.GraphicsUseWaiter graphicsUseWaiter, Matrix4fc projection,
                           int renderHeight) {
        Mode currentMode = mode();
        boolean reset = currentMode == Mode.AUTO && resetRequested;
        if (reset) {
            resetSequence++;
            lastFrameNanos = 0L;
            completedState = null;
            resetRequested = false;
        }

        pendingStateReadback = null;
        if (stateReadbacks != null) {
            stateReadbackIndex = (stateReadbackIndex + 1) % stateReadbacks.length;
            ReadbackSlot slot = stateReadbacks[stateReadbackIndex];
            graphicsUseWaiter.await(slot.graphicsUse);
            if (slot.valid && slot.resetSequence == resetSequence) {
                slot.buffer.invalidate();
                completedState = ExposureStateData.read(MemoryUtil.memByteBuffer(
                        slot.buffer.mapped, ExposureStateData.BYTE_SIZE).order(ByteOrder.nativeOrder()));
            }
            pendingStateReadback = slot;
        }

        frameProjectionM22 = projection.m22();
        frameProjectionM32 = projection.m32();
        // On a reset frame the previous world's exposure is a poor storage-scale prediction. Unity is
        // neutral and the resolve removes it exactly; subsequent frames resume last-frame prediction.
        framePreExposure = reset ? 1.0f : computePreExposure();
        latchOptics(projection, renderHeight);
    }

    /**
     * Solves the lens for this frame. The lens is aimed at the exposure the metered scene needs — the same
     * quantity the controller's own target is built from — rather than at the controller's remaining correction.
     *
     * <p>That distinction is what keeps the two loops from chasing each other. The controller meters the
     * pre-exposure image, so its output never feeds its own input, but {@code evTarget - evApplied} is only zero
     * once it has converged: while it adapts, that difference sweeps across its range, and a lens driven by it
     * racks the iris back and forth on every scene change and hands the difference back through the display
     * scale. The scene's own EV100 is a property of the frame rather than of the controller, so driving from it
     * makes the lens settle at a contribution of exactly 1.0 instead of oscillating around it.
     */
    private void latchOptics(Matrix4fc projection, int renderHeight) {
        float meteredFocus = 0.0f;
        float lensEv100 = Float.NaN;
        if (completedState != null) {
            meteredFocus = completedState.focusDistance();
            // evScene is the absolute EV100 of the metered scene. An unwritten state buffer reads as zero, which
            // is not a brightness any world has.
            float evScene = completedState.evScene();
            if (Float.isFinite(evScene) && evScene != 0.0f) {
                lensEv100 = lensTargetEv100(evScene);
            }
        }
        meteredFocusBlocks = meteredFocus;
        optics = RtCamera.latch(projection, lensEv100, meteredFocus, renderHeight);
        // What the primary pass will actually use as its lens radius, so the diagnostic shows the split between
        // the traced disc and the one the gather supplies rather than only their sum.
        tracedRadiusBlocks = CausticaConfig.Rt.Camera.depthOfField()
                ? optics.defocusScaleBlocks() * RtCamera.tracedDiscFraction(
                        optics.defocusScaleBlocks() * optics.renderHeight() / RtCamera.SENSOR_HEIGHT_METRES,
                        CausticaConfig.Rt.Camera.physicalDepthOfField()
                                ? RtCamera.TRACED_DISC_PIXELS : 0.0f)
                : 0.0f;
    }

    /**
     * The EV100 the virtual lens exposes for: the metered scene's own value, shifted onto the lens' scale so that
     * the reference scene is the one the reference pair is neutral for.
     *
     * <p>The lens' scale is 1.0 for f/2.8 at 1/60 s and is what a MANUAL pair contributes to the display
     * exposure. Under the auto programme it contributes nothing (see {@link #lensExposureScale}), so this target
     * only decides which rung the programme picks: it has to follow the scene's brightness, so the iris opens in
     * a cave and stops down at noon.
     *
     * <p>The shift is {@code REFERENCE_EV100 - EV100_OFFSET} rather than either constant alone. The lens' zero is
     * the reference pair's own EV in the camera's scale; the scene convention's zero is where a 1 cd/m² surface
     * sits. Subtracting the offset maps the scene onto the lens, and the reference puts the neutral point at the
     * pair the auto programme treats as its default. Omitting the offset (or using the reference twice) biases
     * every frame by a constant number of stops, which is what walked the image toward black.
     */
    public static float lensTargetEv100(float evScene) {
        return evScene + CausticaConfig.Rt.Camera.REFERENCE_EV100 - RtSceneUnits.EV100_OFFSET;
    }

    /** Request a GPU-side history reset on the next auto-exposure frame. */
    public void requestReset() {
        resetRequested = true;
    }

    /**
     * The scalar raygen multiplies into scene radiance before the fp16 write, so stored values sit near
     * {@code key} at any absolute
     * scene brightness instead of spanning the ~26 EV that physical units require.
     *
     * <p>Correctness does not depend on this being <em>current</em> — the display pass divides by
     * exactly the same latched value, so any pre-exposure cancels algebraically. Staleness only
     * affects how well-centred the stored values are, which is why last frame's readback is fine and
     * no fence is needed. 1.0 disables the mechanism.
     */
    public float preExposure() {
        return framePreExposure;
    }

    private float computePreExposure() {
        if (!CausticaConfig.Rt.Exposure.PRE_EXPOSURE.value()) {
            return 1.0f;
        }
        // Manual mode has a known fixed absolute exposure, so pre-exposing by it makes the residual
        // exactly 1.0 -- the best-centred choice available, and it needs no readback.
        if (mode() != Mode.AUTO) {
            return manualExposureScale();
        }
        if (completedState == null) {
            return 1.0f;
        }
        // Deliberately NOT Exposure.clampScale: its 1e-4 floor is a bound on the artistic exposure
        // multiplier, and physical units put noon at ~3e-5 absolute, which that floor would
        // truncate -- silently de-centring exactly the case pre-exposure exists to handle. The
        // controller's own minEv/maxEv already bound this value; here we only reject garbage.
        float previous = completedState.previous();
        return Float.isFinite(previous) && previous > 0.0f ? previous : 1.0f;
    }

    private ByteBuffer stateDataBuffer() {
        return MemoryUtil.memByteBuffer(state.mapped, ExposureStateData.BYTE_SIZE).order(ByteOrder.nativeOrder());
    }

    private ExposureStateData readState() {
        return ExposureStateData.read(stateDataBuffer());
    }

    record AutoConfig(float key, float minEv, float maxEv, float adaptDarken, float adaptBrighten,
                      float evBias,
                      float lowPercentile, float highPercentile, int stride,
                      float centerWeightSigma, float centerWeightFloor, float skyWeightCap,
                      float emissiveWeightCap,
                      ExposureCurve curve, float preExposure, int resetSequence,
                      float physicalScale, float manualScale, float projM22, float projM32,
                      float focusHalflifeSeconds) {
        /**
         * Offset taking the resolve's {@code log2(metered stored luminance)} to EV100. The metered
         * buffer holds {@code L * preExposure}, so the pre-exposure has to come back out before the
         * unit convention's offset applies.
         */
        float evOffset() {
            return RtSceneUnits.EV100_OFFSET - (float) (Math.log(Math.max(preExposure, 1.0e-12f)) / Math.log(2.0));
        }
    }

    private ExposureCurve curveConfig() {
        String spec = CausticaConfig.Rt.Exposure.curve();
        if (cachedCurve != null && Objects.equals(cachedCurveSpec, spec)) {
            return cachedCurve;
        }
        ExposureCurve parsed;
        try {
            parsed = parseCurve(spec);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException("Invalid exposure curve in look package '"
                    + RtLookPackage.current().id() + "': " + spec, e);
        }
        cachedCurveSpec = spec;
        cachedCurve = parsed;
        return parsed;
    }

    static ExposureCurve parseCurve(String spec) {
        if (spec == null) {
            throw new IllegalArgumentException("curve is null");
        }
        if ("full".equalsIgnoreCase(spec.trim())) {
            return new ExposureCurve(-6.0f, 0.0f, -3.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f);
        }
        String[] encodedPoints = spec.split(",");
        if (encodedPoints.length != 4) {
            throw new IllegalArgumentException("expected exactly four scene:compensation points");
        }
        float[] scene = new float[4];
        float[] compensation = new float[4];
        for (int i = 0; i < encodedPoints.length; i++) {
            String[] pair = encodedPoints[i].trim().split(":", -1);
            if (pair.length != 2) {
                throw new IllegalArgumentException("point " + (i + 1) + " is not scene:compensation");
            }
            try {
                scene[i] = Float.parseFloat(pair[0].trim());
                compensation[i] = Float.parseFloat(pair[1].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("point " + (i + 1) + " contains a non-number", e);
            }
            if (!Float.isFinite(scene[i]) || !Float.isFinite(compensation[i])) {
                throw new IllegalArgumentException("point " + (i + 1) + " is not finite");
            }
        }
        // Four elements: insertion sort avoids a temporary point-object list.
        for (int i = 1; i < 4; i++) {
            float sceneValue = scene[i];
            float compensationValue = compensation[i];
            int j = i - 1;
            while (j >= 0 && scene[j] > sceneValue) {
                scene[j + 1] = scene[j];
                compensation[j + 1] = compensation[j];
                j--;
            }
            scene[j + 1] = sceneValue;
            compensation[j + 1] = compensationValue;
        }
        for (int i = 1; i < 4; i++) {
            if (scene[i] - scene[i - 1] < 1.0e-4f) {
                throw new IllegalArgumentException("scene EV points must be distinct");
            }
        }
        return new ExposureCurve(scene[0], compensation[0], scene[1], compensation[1],
                scene[2], compensation[2], scene[3], compensation[3]);
    }

    record ExposureCurve(float scene0, float compensation0, float scene1, float compensation1,
                         float scene2, float compensation2, float scene3, float compensation3) {
        float compensationAt(float sceneEv) {
            if (sceneEv <= scene0) {
                return compensation0;
            }
            if (sceneEv < scene1) {
                return interpolate(sceneEv, scene0, compensation0, scene1, compensation1);
            }
            if (sceneEv < scene2) {
                return interpolate(sceneEv, scene1, compensation1, scene2, compensation2);
            }
            if (sceneEv < scene3) {
                return interpolate(sceneEv, scene2, compensation2, scene3, compensation3);
            }
            return compensation3;
        }

        float effectiveSlopeAt(float sceneEv) {
            if (sceneEv <= scene0 || sceneEv >= scene3) {
                return 1.0f;
            }
            if (sceneEv < scene1) {
                return 1.0f - slope(scene0, compensation0, scene1, compensation1);
            }
            if (sceneEv < scene2) {
                return 1.0f - slope(scene1, compensation1, scene2, compensation2);
            }
            return 1.0f - slope(scene2, compensation2, scene3, compensation3);
        }

        private static float interpolate(float x, float x0, float y0, float x1, float y1) {
            float t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }

        private static float slope(float x0, float y0, float x1, float y1) {
            return (y1 - y0) / (x1 - x0);
        }
    }

    private enum Mode {
        MANUAL("manual"),
        AUTO("auto");

        private final String configName;

        Mode(String configName) {
            this.configName = configName;
        }

        static Mode parse(String value) {
            if (value != null) {
                for (Mode mode : values()) {
                    if (mode.configName.equalsIgnoreCase(value)) {
                        return mode;
                    }
                }
            }
            return AUTO;
        }
    }
}
