package dev.comfyfluffy.caustica.client;

import com.mojang.serialization.Codec;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaConfig.BooleanSetting;
import dev.comfyfluffy.caustica.CausticaConfig.FloatSetting;
import dev.comfyfluffy.caustica.CausticaConfig.IntSetting;
import dev.comfyfluffy.caustica.CausticaConfig.StringSetting;
import dev.comfyfluffy.caustica.rt.RtDeviceBringup;
import dev.comfyfluffy.caustica.rt.RtNrd;
import dev.comfyfluffy.caustica.rt.RtRecon;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import net.minecraft.client.Minecraft;
import net.minecraft.client.OptionInstance;
import net.minecraft.client.Options;
import net.minecraft.network.chat.Component;

/**
 * Builds the {@link OptionInstance} widgets shown in the RT section of the vanilla Video Settings screen
 * (injected by {@code VideoSettingsScreenMixin}). Each option is bound straight to a {@link CausticaConfig}
 * runtime setting: the initial value is read from the current config, and the value-update listener writes
 * back through {@code set(...)} so changes take effect on the next frame.
 *
 * <p>Only settings the renderer re-reads per-frame are exposed here — toggles that would require a device or
 * buffer-pool rebuild (worker threads, OMM, max-entity capacities, PBR material flags) are intentionally
 * left to the {@code -Dcaustica.*} startup surface. DLSS-RR quality is the exception: the render resolution
 * is queried from NGX for the chosen quality mode on every resize (see
 * {@code RtDlssRr.queryOptimalRenderSize}), and the RR feature itself is recreated live whenever
 * {@code quality} changes (see {@code RtDlssRr.ensureFeature}), so it is safe to expose here.
 */
public final class RtVideoOptions {
    private RtVideoOptions() {
    }

    /**
     * Runtime-tunable RT options, in display order. Paired two-per-row by {@code OptionsList.addSmall}.
     * The HDR entries are omitted entirely (not just disabled) when this session's swapchain isn't
     * PQ-capable ({@code CausticaConfig.Rt.Hdr.swapchainPqAvailable()}) — offering a toggle/sliders that
     * can never do anything is worse than not showing them, and unlike most settings here this one is
     * fixed by hardware/OS/compositor at surface-creation time. The current swapchain may still be native
     * SDR; changing the toggle invalidates its configuration and recreates it in the selected format.
     */
    public static OptionInstance<?>[] runtimeOptions() {
        List<OptionInstance<?>> options = new ArrayList<>(List.of(
            exposureMode(),
            manualEv(),
            gamma(),
            spp(),
            maxBounces(),
            entities(),
            particles(),
            waterWaves(),
            denoiser(),
            superResolutionMethod(),
            superResolution(),
            depthOfField(),
            depthOfFieldMode(),
            artisticBlur(),
            maxBlurPercent(),
            depthOfFieldSamples(),
            apertureMode(),
            aperture(),
            shutterMode(),
            shutterTime(),
            focusMode(),
            focusDistance(),
            motionBlur(),
            motionBlurShutter(),
            motionBlurSamples()
        ));
        if (CausticaConfig.Rt.Hdr.swapchainPqAvailable()) {
            options.add(hdrEnabled());
            options.add(hdrUiBrightness());
            options.add(hdrPeak());
        }
        options.add(debugView());
        return options.toArray(OptionInstance<?>[]::new);
    }

    private static OptionInstance<String> exposureMode() {
        StringSetting setting = CausticaConfig.Rt.Exposure.MODE;
        return new OptionInstance<>(
            "caustica.options.rt.exposureMode",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.exposureMode.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.exposureMode." + value),
            new OptionInstance.Enum<>(List.of("auto", "manual"), Codec.STRING),
            setting.get(),
            setting::set);
    }

    private static OptionInstance<Integer> manualEv() {
        FloatSetting setting = CausticaConfig.Rt.Exposure.MANUAL_EV;
        return new OptionInstance<>(
            "caustica.options.rt.manualEv",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.manualEv.tooltip")),
            (caption, tenths) -> {
                float ev = tenths / 10.0f;
                String sign = ev > 0.0f ? "+" : "";
                return Options.genericValueLabel(caption,
                        Component.literal(sign + String.format(Locale.ROOT, "%.1f EV", ev)));
            },
            new OptionInstance.IntRange(-150, 150),
            Math.clamp(Math.round(setting.value() * 10.0f), -150, 150),
            tenths -> setting.set(tenths / 10.0f));
    }

    private static OptionInstance<Integer> gamma() {
        FloatSetting setting = CausticaConfig.Rt.Tonemap.GAMMA;
        return new OptionInstance<>(
            "caustica.options.rt.gamma",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.gamma.tooltip")),
            (caption, hundredths) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(Locale.ROOT, "%.2f", hundredths / 100.0f))),
            new OptionInstance.IntRange(50, 150),
            Math.clamp(Math.round(setting.value() * 100.0f), 50, 150),
            hundredths -> setting.set(hundredths / 100.0f));
    }

    private static OptionInstance<Integer> spp() {
        IntSetting setting = CausticaConfig.Rt.Composite.SPP;
        return new OptionInstance<>(
            "caustica.options.rt.spp",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.spp.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(1, 8),
            Math.clamp(setting.value(), 1, 8),
            setting::set);
    }

    private static OptionInstance<Integer> maxBounces() {
        IntSetting setting = CausticaConfig.Rt.Composite.MAX_BOUNCES;
        return new OptionInstance<>(
            "caustica.options.rt.maxBounces",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.maxBounces.tooltip")),
            (caption, value) -> Options.genericValueLabel(caption, value),
            new OptionInstance.IntRange(2, 8),
            Math.clamp(setting.value(), 2, 8),
            setting::set);
    }

    private static OptionInstance<Boolean> entities() {
        return bool("caustica.options.rt.entities", CausticaConfig.Rt.Entities.ENABLED);
    }

    private static OptionInstance<Boolean> particles() {
        return bool("caustica.options.rt.particles", CausticaConfig.Rt.Entities.PARTICLES_ENABLED);
    }

    private static OptionInstance<Boolean> waterWaves() {
        return bool("caustica.options.rt.waterWaves", CausticaConfig.Rt.Composite.WATER_WAVES);
    }

    /**
     * Which denoiser removes the path tracer's noise. Only the choices this machine can actually run are
     * offered — the same reasoning as the HDR entries below: an option that can never take effect is worse
     * than an absent one. "Auto" is always present and already resolves to the best available denoiser, so
     * hiding an unavailable entry costs nothing.
     */
    private static OptionInstance<String> denoiser() {
        StringSetting setting = CausticaConfig.Rt.Reconstruction.DENOISER;
        List<String> values = new ArrayList<>();
        values.add("auto");
        if (RtDeviceBringup.ngxCapable()) {
            values.add("dlss-rr");
        }
        if (RtNrd.INSTANCE.isAvailable()) {
            values.add("nrd");
        }
        values.add("builtin");
        values.add("off");
        String initial = values.contains(setting.get()) ? setting.get() : "auto";
        return new OptionInstance<>(
            "caustica.options.rt.denoiser",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.denoiser.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.denoiser." + value),
            new OptionInstance.Enum<>(values, Codec.STRING),
            initial,
            setting::set);
    }

    /**
     * How the render-resolution result is scaled to the display: FSR's temporal upscaler, or the renderer's
     * own bilinear resolve. Always offered, because "bilinear" is meaningful everywhere and "auto" adapts;
     * picking FSR without its runtime present falls back to the resolve and says so in the log.
     */
    private static OptionInstance<String> superResolutionMethod() {
        StringSetting setting = CausticaConfig.Rt.Reconstruction.UPSCALER;
        return new OptionInstance<>(
            "caustica.options.rt.superResolutionMethod",
            OptionInstance.cachedConstantTooltip(
                    Component.translatable("caustica.options.rt.superResolutionMethod.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.superResolutionMethod." + value),
            new OptionInstance.Enum<>(CausticaConfig.Rt.Reconstruction.UPSCALERS, Codec.STRING),
            setting.get(),
            setting::set);
    }

    /**
     * NVSDK_NGX_PerfQuality_Value per super-resolution mode, in ladder order: native-aa (DLAA), quality,
     * balanced, performance, ultra-performance. DLSS Ray Reconstruction only accepts Performance(0),
     * Balanced(1), Quality(2), Ultra-Performance(3) and DLAA(5); Ultra Quality(4) is excluded because its
     * optimal-settings query returns a zeroed render size.
     */
    private static final List<Integer> DLSS_BY_MODE = List.of(5, 2, 1, 0, 3);

    /**
     * Super-resolution mode: the render/display ratio the trace runs at. One control drives whichever
     * backend currently owns the upscale — DLSS Ray Reconstruction stores its own quality values, the
     * other backends share the FSR ladder — because the modes mean the same thing to the user either way.
     */
    private static OptionInstance<Integer> superResolution() {
        List<Integer> modes = CausticaConfig.Rt.Reconstruction.QUALITY_STEPS;
        boolean dlss = RtRecon.INSTANCE.superResolutionIsDlss();
        int initialPosition = dlss
                ? Math.max(DLSS_BY_MODE.indexOf(CausticaConfig.Rt.DlssRr.QUALITY.value()), 0)
                : Math.max(modes.indexOf(CausticaConfig.Rt.Reconstruction.QUALITY.value()), 0);
        return new OptionInstance<>(
            "caustica.options.rt.superResolution",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.superResolution.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption,
                    Component.translatable("caustica.options.rt.superResolution." + position)),
            new OptionInstance.IntRange(0, modes.size() - 1),
            initialPosition,
            position -> {
                // Resolved when the value changes, not when the screen opened, so switching the denoiser
                // above and then picking a mode writes to whichever backend is active by then.
                if (RtRecon.INSTANCE.superResolutionIsDlss()) {
                    CausticaConfig.Rt.DlssRr.QUALITY.set(DLSS_BY_MODE.get(position));
                } else {
                    CausticaConfig.Rt.Reconstruction.QUALITY.set(modes.get(position));
                }
            });
    }

    private static OptionInstance<Boolean> depthOfField() {
        BooleanSetting setting = CausticaConfig.Rt.Camera.DEPTH_OF_FIELD_ENABLED;
        return OptionInstance.createBoolean(
            "caustica.options.rt.depthOfField",
            OptionInstance.cachedConstantTooltip(
                Component.translatable("caustica.options.rt.depthOfField.tooltip")),
            setting.value(),
            setting::set);
    }

    /**
     * Auto or manual aperture. Auto solves the f-number from the metered scene every frame, so the depth of
     * field follows the light; manual pins the f-number the way a lens ring does.
     */
    private static OptionInstance<String> apertureMode() {
        StringSetting setting = CausticaConfig.Rt.Camera.APERTURE_MODE;
        return new OptionInstance<>(
            "caustica.options.rt.apertureMode",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.apertureMode.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.apertureMode." + value),
            new OptionInstance.Enum<>(List.of("auto", "manual"), Codec.STRING),
            setting.get(),
            setting::set);
    }

    /**
     * Aperture in f-stops, stepped in whole stops: f/1.4 through f/32. One stop is one EV of exposure, so
     * the ladder reads straight off the same scale the exposure value is measured on.
     */
    private static OptionInstance<Integer> aperture() {
        FloatSetting setting = CausticaConfig.Rt.Camera.MANUAL_APERTURE;
        return new OptionInstance<>(
            "caustica.options.rt.aperture",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.aperture.tooltip")),
            (caption, stop) -> Options.genericValueLabel(caption,
                    Component.literal("f/" + formatAperture(CausticaConfig.Rt.Camera.fStop(stop)))
                            .append(autoSuffix(CausticaConfig.Rt.Camera.APERTURE_MODE))),
            new OptionInstance.IntRange(CausticaConfig.Rt.Camera.APERTURE_MIN_STOP,
                    CausticaConfig.Rt.Camera.APERTURE_MAX_STOP),
            apertureStop(setting.value()),
            stop -> setting.set(CausticaConfig.Rt.Camera.fStop(stop)));
    }

    private static OptionInstance<String> shutterMode() {
        StringSetting setting = CausticaConfig.Rt.Camera.SHUTTER_MODE;
        return new OptionInstance<>(
            "caustica.options.rt.shutterMode",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.shutterMode.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.shutterMode." + value),
            new OptionInstance.Enum<>(List.of("auto", "manual"), Codec.STRING),
            setting.get(),
            setting::set);
    }

    /**
     * Shutter time on the standard camera ladder, shown as the reciprocal a photographer reads off the
     * dial. The exposure contribution is the same quantity: one step is one stop.
     */
    private static OptionInstance<Integer> shutterTime() {
        FloatSetting setting = CausticaConfig.Rt.Camera.MANUAL_SHUTTER_TIME;
        List<Integer> ladder = CausticaConfig.Rt.Camera.SHUTTER_DENOMINATORS;
        int initial = shutterDenominator(setting.value());
        int initialPosition = Math.max(ladder.indexOf(initial), 0);
        return new OptionInstance<>(
            "caustica.options.rt.shutterTime",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.shutterTime.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption,
                    Component.literal("1/" + ladder.get(position) + " s")
                            .append(autoSuffix(CausticaConfig.Rt.Camera.SHUTTER_MODE))),
            new OptionInstance.IntRange(0, ladder.size() - 1),
            initialPosition,
            position -> setting.set(CausticaConfig.Rt.Camera.shutterSeconds(ladder.get(position))));
    }

    /** Auto focus follows the metered centre of the frame; manual uses a fixed distance. */
    private static OptionInstance<String> focusMode() {
        StringSetting setting = CausticaConfig.Rt.Camera.FOCUS_MODE;
        return new OptionInstance<>(
            "caustica.options.rt.focusMode",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.focusMode.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.focusMode." + value),
            new OptionInstance.Enum<>(List.of("auto", "manual"), Codec.STRING),
            setting.get(),
            setting::set);
    }

    /** Focus distance in blocks; one block is one metre, so the number is a real distance. */
    private static OptionInstance<Integer> focusDistance() {
        FloatSetting setting = CausticaConfig.Rt.Camera.FOCUS_DISTANCE;
        return new OptionInstance<>(
            "caustica.options.rt.focusDistance",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.focusDistance.tooltip")),
            (caption, blocks) -> Options.genericValueLabel(caption, Component.literal(blocks + " m")
                    .append(autoSuffix(CausticaConfig.Rt.Camera.FOCUS_MODE))),
            new OptionInstance.IntRange(1, 512),
            Math.clamp(Math.round(setting.value()), 1, 512),
            blocks -> setting.set(blocks.floatValue()));
    }

    /**
     * The circle of confusion the lens modes work to, as a percentage of the frame height. It is the sharp range
     * rather than the blur: a smaller disc means a stricter definition of "sharp", so less of the frame is inside
     * it and the depth of field looks shallower.
     */
    private static OptionInstance<Integer> maxBlurPercent() {
        FloatSetting setting = CausticaConfig.Rt.Camera.MAX_BLUR_PERCENT;
        return new OptionInstance<>(
            "caustica.options.rt.dofMaxBlur",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.dofMaxBlur.tooltip")),
            (caption, percent) -> Options.genericValueLabel(caption,
                    Component.literal(percent + "%")),
            new OptionInstance.IntRange(1, 100),
            Math.clamp(Math.round(setting.value()), 1, 100),
            percent -> setting.set(percent.floatValue()));
    }

    /** How depth of field is produced: a gather over the finished image, or primary rays through the aperture. */
    private static OptionInstance<String> depthOfFieldMode() {
        StringSetting setting = CausticaConfig.Rt.Camera.DOF_MODE;
        return new OptionInstance<>(
            "caustica.options.rt.dofMode",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.dofMode.tooltip")),
            (caption, value) -> Component.translatable("caustica.options.rt.dofMode." + value),
            new OptionInstance.Enum<>(CausticaConfig.Rt.Camera.DOF_MODES, Codec.STRING),
            setting.get(),
            setting::set);
    }

    /**
     * How much of the lens' own blur to draw, in tenths of a multiple. Unity is the lens itself, so the steps
     * have to be finer than whole multiples for the honest setting to be selectable.
     */
    private static OptionInstance<Integer> artisticBlur() {
        FloatSetting setting = CausticaConfig.Rt.Camera.ARTISTIC_BLUR;
        return new OptionInstance<>(
            "caustica.options.rt.dofArtisticBlur",
            OptionInstance.cachedConstantTooltip(
                Component.translatable("caustica.options.rt.dofArtisticBlur.tooltip")),
            (caption, tenths) -> Options.genericValueLabel(caption,
                    Component.literal(String.format(java.util.Locale.ROOT, "%.1fx", tenths / 10.0))),
            new OptionInstance.IntRange(2, 80),
            Math.clamp(Math.round(setting.value() * 10.0f), 2, 80),
            tenths -> setting.set(tenths / 10.0f));
    }

    /**
     * Points of the lens aperture each pixel traces a path through, for the physical mode. It is not the blur
     * size: the trace covers only the part of the disc one sample per pixel can resolve, and the gather supplies
     * the rest. These samples are what keep that traced part from flickering.
     */
    private static OptionInstance<Integer> depthOfFieldSamples() {
        IntSetting setting = CausticaConfig.Rt.Camera.DOF_SAMPLES;
        List<Integer> ladder = CausticaConfig.Rt.Camera.DOF_SAMPLE_STEPS;
        int initialPosition = Math.max(ladder.indexOf(setting.value()), 0);
        return new OptionInstance<>(
            "caustica.options.rt.dofSamples",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.dofSamples.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption,
                    Component.literal(Integer.toString(ladder.get(position)))),
            new OptionInstance.IntRange(0, ladder.size() - 1),
            initialPosition,
            position -> setting.set(ladder.get(position)));
    }

    /** Nearest aperture rung index for a stored f-number, so the widget opens on the value in the config. */
    private static int apertureStop(float fNumber) {
        int stop = Math.round(CausticaConfig.Rt.Camera.apertureStops(fNumber)) - 1;
        return Math.clamp(stop, CausticaConfig.Rt.Camera.APERTURE_MIN_STOP,
                CausticaConfig.Rt.Camera.APERTURE_MAX_STOP);
    }

    private static int shutterDenominator(float seconds) {
        int best = 60;
        float bestDistance = Float.MAX_VALUE;
        for (int denominator : CausticaConfig.Rt.Camera.SHUTTER_DENOMINATORS) {
            float distance = Math.abs((float) Math.log(
                    CausticaConfig.Rt.Camera.shutterSeconds(denominator) / Math.max(seconds, 1.0e-6f)));
            if (distance < bestDistance) {
                bestDistance = distance;
                best = denominator;
            }
        }
        return best;
    }

    /** F-numbers are printed the way lenses are marked: 1.4, 2.8, 11, not 2.828427. */
    private static String formatAperture(float fNumber) {
        return fNumber < 10.0f
                ? String.format(Locale.ROOT, "%.1f", fNumber)
                : Integer.toString(Math.round(fNumber));
    }

    /**
     * Marks a value the matching auto mode is currently driving. The value shown is the setting the auto
     * programme writes back, so it tracks the solved value live; the marker is what tells the player that
     * moving the slider will not hold until the mode is switched to Manual.
     */
    private static Component autoSuffix(StringSetting modeSetting) {
        return "auto".equals(modeSetting.get())
                ? Component.translatable("caustica.options.rt.autoValue")
                : Component.empty();
    }

    private static OptionInstance<Boolean> motionBlur() {
        BooleanSetting setting = CausticaConfig.Rt.MotionBlur.ENABLED;
        return OptionInstance.createBoolean(
            "caustica.options.rt.motionBlur",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.motionBlur.tooltip")),
            setting.value(),
            setting::set);
    }
    // Film's shutter angle in degrees. 180 (the standard) keeps the virtual shutter open for half the frame
    // interval, 360 for all of it, and 0 disables the streak without turning the option off.
    private static OptionInstance<Integer> motionBlurShutter() {
        IntSetting setting = CausticaConfig.Rt.MotionBlur.SHUTTER_ANGLE;
        return new OptionInstance<>(
            "caustica.options.rt.motionBlurShutter",
            OptionInstance.cachedConstantTooltip(
                Component.translatable("caustica.options.rt.motionBlurShutter.tooltip")),
            (caption, degrees) -> Options.genericValueLabel(caption, Component.literal(degrees + "\u00b0")),
            new OptionInstance.IntRange(0, 360),
            Math.clamp(setting.value(), 0, 360),
            setting::set);
    }

    private static OptionInstance<Integer> motionBlurSamples() {
        IntSetting setting = CausticaConfig.Rt.MotionBlur.SAMPLES;
        List<Integer> steps = CausticaConfig.Rt.MotionBlur.SAMPLE_STEPS;
        int initial = steps.contains(setting.value()) ? steps.indexOf(setting.value()) : steps.indexOf(8);
        return new OptionInstance<>(
            "caustica.options.rt.motionBlurSamples",
            OptionInstance.cachedConstantTooltip(
                Component.translatable("caustica.options.rt.motionBlurSamples.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption, Component.literal(steps.get(position).toString())),
            new OptionInstance.IntRange(0, steps.size() - 1),
            Math.max(initial, 0),
            position -> setting.set(steps.get(position)));
    }

    private static OptionInstance<Boolean> hdrEnabled() {
        BooleanSetting setting = CausticaConfig.Rt.Hdr.ENABLED;
        return OptionInstance.createBoolean(
            "caustica.options.rt.hdr",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdr.tooltip")),
            setting.value(),
            enabled -> {
                if (setting.value() != enabled) {
                    setting.set(enabled);
                    // Reuse the framebuffer-resize path at the next safe frame boundary. GpuSurface
                    // refuses configure() while an image is acquired, so doing it directly here is unsafe.
                    Minecraft.getInstance().invalidateSurfaceConfiguration();
                }
            });
    }

    private static OptionInstance<Integer> hdrUiBrightness() {
        FloatSetting setting = CausticaConfig.Rt.Hdr.UI_NITS;
        return new OptionInstance<>(
            "caustica.options.rt.hdrUiBrightness",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdrUiBrightness.tooltip")),
            (caption, nits) -> Options.genericValueLabel(caption, Component.literal(nits + " nits")),
            new OptionInstance.IntRange(80, 500),
            Math.clamp(Math.round(setting.value()), 80, 500),
            nits -> setting.set(nits.floatValue()));
    }

    // Each step selects a baked ACES HDR mastering target. Changes take effect on the next frame.
    private static OptionInstance<Integer> hdrPeak() {
        IntSetting setting = CausticaConfig.Rt.Hdr.PEAK_NITS;
        List<Integer> steps = CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS;
        int initialPeak = steps.contains(setting.value()) ? setting.value() : 1000;
        int initialPosition = steps.indexOf(initialPeak);
        return new OptionInstance<>(
            "caustica.options.rt.hdrPeak",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.hdrPeak.tooltip")),
            (caption, position) -> Options.genericValueLabel(caption, Component.literal(steps.get(position) + " nits")),
            new OptionInstance.IntRange(0, steps.size() - 1),
            Math.max(initialPosition, 0),
            position -> setting.set(steps.get(position)));
    }

    private static OptionInstance<Integer> debugView() {
        IntSetting setting = CausticaConfig.Rt.Composite.DEBUG_VIEW;
        return new OptionInstance<>(
            "caustica.options.rt.debugView",
            OptionInstance.cachedConstantTooltip(Component.translatable("caustica.options.rt.debugView.tooltip")),
            // CycleButton (used for Enum values) already prepends "caption: " itself (DisplayState.
            // NAME_AND_VALUE), so this must return only the value's text, not caption + value again.
            (caption, value) -> Component.translatable("caustica.options.rt.debugView." + value),
            new OptionInstance.Enum<>(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), Codec.INT),
            Math.clamp(setting.value(), 0, 12),
            setting::set);
    }

    private static OptionInstance<Boolean> bool(String captionKey, BooleanSetting setting) {
        return OptionInstance.createBoolean(
            captionKey,
            OptionInstance.cachedConstantTooltip(Component.translatable(captionKey + ".tooltip")),
            setting.value(),
            setting::set);
    }
}
