package dev.comfyfluffy.caustica.rt.ffx;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vulkan.VulkanDevice;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.mixin.GpuDeviceAccessor;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.accel.RtImage;

import net.fabricmc.loader.api.FabricLoader;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkInstance;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * FSR 3.1 temporal upscaler over the AMD FidelityFX API, used as the upscaling half of the built-in
 * reconstruction chain: the denoiser filters at render resolution and this brings the result to display
 * resolution with reprojection, instead of the bilinear resolve.
 *
 * <p>It is strictly optional. The runtime is a vendor-neutral Vulkan library but is not bundled with the
 * mod, so everything here is best-effort: when the library or a usable context is absent,
 * {@link #isAvailable()} stays false and the chain keeps its own resolve. The library is loaded lazily
 * and the failure is latched, so a missing runtime costs one filesystem probe per session, not one per
 * frame.
 */
public final class FsrUpscaler {
    public static final FsrUpscaler INSTANCE = new FsrUpscaler();

    private static final PlatformNatives NATIVES = PlatformNatives.current();
    private static final long RESET_WAIT_TIMEOUT_NS = 200_000_000L;

    private FfxLibrary lib;
    private FfxUpscaleContext context;
    private boolean probeAttempted;
    private boolean available;
    private boolean failed;
    private boolean resetHistory;
    private long lastFrameNanos;

    private int contextRenderWidth = -1;
    private int contextRenderHeight = -1;
    private int contextDisplayWidth = -1;
    private int contextDisplayHeight = -1;
    private int contextQuality = Integer.MIN_VALUE;

    private FsrUpscaler() {
    }

    /**
     * Whether this system can run FSR at all. Probed once; the probe loads the library and asks it for the
     * upscale provider versions, which validates both the FFM plumbing and that the library really is the
     * FidelityFX runtime rather than a same-named file.
     */
    public boolean isAvailable() {
        if (probeAttempted) {
            return available;
        }
        probeAttempted = true;
        Path library = locateLibrary();
        if (library == null) {
            CausticaMod.LOGGER.info("FSR unavailable: {} not found (bundled natives, run/natives, or -Dcaustica.fsr.path)",
                    NATIVES.libraryName());
            return false;
        }
        try {
            FfxLibrary loaded = FfxLibrary.load(library);
            var versions = FfxUpscaleContext.queryAvailableVersions(loaded);
            available = !versions.isEmpty();
            if (available) {
                lib = loaded;
                CausticaMod.LOGGER.info("FSR upscaling available ({} version {})", library,
                        versions.get(versions.size() - 1).name());
            } else {
                CausticaMod.LOGGER.warn("FSR runtime at {} exposes no upscale provider", library);
            }
        } catch (Throwable t) {
            CausticaMod.LOGGER.warn("FSR runtime at {} could not be loaded; upscaling stays on the built-in resolve",
                    library, t);
        }
        return available;
    }

    /**
     * Create or resize the upscale context for this resolution pair. Returns false when FSR cannot run,
     * which leaves the caller on its own resolve path.
     */
    public boolean ensure(int renderWidth, int renderHeight, int displayWidth, int displayHeight, int quality) {
        if (!isAvailable() || failed) {
            return false;
        }
        if (context != null && contextRenderWidth == renderWidth && contextRenderHeight == renderHeight
                && contextDisplayWidth == displayWidth && contextDisplayHeight == displayHeight
                && contextQuality == quality) {
            return true;
        }
        if (!(((GpuDeviceAccessor) RenderSystem.getDevice()).caustica$getBackend() instanceof VulkanDevice device)) {
            return false;
        }
        try {
            releaseContext();
            // HIGH_DYNAMIC_RANGE: the input is scene-linear ACEScg with values above 1, and the exposure is
            // already applied by the tracer, so FSR must not estimate one. DEPTH_INVERTED: the depth guide
            // is reversed-Z. DEBUG_CHECKING makes the runtime report API misuse through the log callback
            // instead of silently producing a wrong image.
            int flags = FfxUpscaleContext.FLAG_HIGH_DYNAMIC_RANGE
                    | FfxUpscaleContext.FLAG_DEPTH_INVERTED
                    | FfxUpscaleContext.FLAG_DEBUG_CHECKING;
            long getDeviceProcAddr;
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkInstance instance = device.vkDevice().getPhysicalDevice().getInstance();
                getDeviceProcAddr = VK10.vkGetInstanceProcAddr(instance, stack.ASCII("vkGetDeviceProcAddr"));
            }
            context = FfxUpscaleContext.create(lib,
                    device.vkDevice().address(),
                    device.vkDevice().getPhysicalDevice().address(),
                    getDeviceProcAddr,
                    renderWidth, renderHeight,
                    displayWidth, displayHeight,
                    flags, true);
            contextRenderWidth = renderWidth;
            contextRenderHeight = renderHeight;
            contextDisplayWidth = displayWidth;
            contextDisplayHeight = displayHeight;
            contextQuality = quality;
            resetHistory = true;
            CausticaMod.LOGGER.info("FSR upscale context created: {}x{} -> {}x{}", renderWidth, renderHeight,
                    displayWidth, displayHeight);
            return true;
        } catch (Throwable t) {
            failed = true;
            releaseContext();
            CausticaMod.LOGGER.error("FSR context creation failed; upscaling stays on the built-in resolve", t);
            return false;
        }
    }

    /**
     * Record the upscale: {@code color} (the denoised render-res image), {@code depth} and {@code motion}
     * are reprojected into {@code out} at display resolution. Jitter is the sub-pixel camera offset already
     * applied to the primary ray, in render pixels; the depth guide is reversed-Z, so the camera planes are
     * passed in FSR's swapped convention.
     */
    public boolean dispatch(VkCommandBuffer cmd, RtImage color, RtImage depth, RtImage motion, RtImage out,
                            int renderWidth, int renderHeight, int displayWidth, int displayHeight,
                            float jitterX, float jitterY, float projectionM11, float projectionM22,
                            float projectionM32) {
        if (context == null || failed) {
            return false;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            float frameMs = nextFrameMs();
            CameraPlanes planes = cameraPlanes(projectionM22, projectionM32);
            context.dispatchUpscale(new FfxUpscaleContext.DispatchParams(
                    cmd.address(),
                    new FfxUpscaleContext.Resource(color.image, FfxUpscaleContext.FORMAT_R16G16B16A16_FLOAT,
                            renderWidth, renderHeight, FfxUpscaleContext.USAGE_READ_ONLY,
                            FfxUpscaleContext.STATE_COMMON),
                    new FfxUpscaleContext.Resource(depth.image, FfxUpscaleContext.FORMAT_R32_FLOAT,
                            renderWidth, renderHeight, FfxUpscaleContext.USAGE_DEPTHTARGET,
                            FfxUpscaleContext.STATE_COMMON),
                    new FfxUpscaleContext.Resource(motion.image, FfxUpscaleContext.FORMAT_R16G16_FLOAT,
                            renderWidth, renderHeight, FfxUpscaleContext.USAGE_READ_ONLY,
                            FfxUpscaleContext.STATE_COMMON),
                    new FfxUpscaleContext.Resource(out.image, FfxUpscaleContext.FORMAT_R16G16B16A16_FLOAT,
                            displayWidth, displayHeight, FfxUpscaleContext.USAGE_UAV,
                            FfxUpscaleContext.STATE_UNORDERED_ACCESS),
                    jitterX, jitterY,
                    // Our motion vectors are already in render-pixel units and jitter-free, so the scale is 1
                    // in both axes (the flag for display-res MVs stays unset: they are render-res).
                    1.0f, 1.0f,
                    renderWidth, renderHeight,
                    displayWidth, displayHeight,
                    frameMs, resetHistory,
                    planes.near(), planes.far(), fovY(projectionM11),
                    0));
            resetHistory = false;
            return true;
        } catch (Throwable t) {
            failed = true;
            CausticaMod.LOGGER.error("FSR dispatch failed; upscaling stays on the built-in resolve", t);
            return false;
        }
    }

    private float nextFrameMs() {
        long now = System.nanoTime();
        float frameMs = lastFrameNanos == 0 ? 16.6f
                : Math.clamp((now - lastFrameNanos) / 1_000_000.0f, 0.1f, 200.0f);
        lastFrameNanos = now;
        return frameMs;
    }

    /** FSR's naming under the DEPTH_INVERTED flag: {@code near} carries the far-plane distance. */
    private record CameraPlanes(float near, float far) {
    }

    /**
     * Near and far plane distances recovered from the projection matrix. Under reversed-Z FSR wants
     * {@code cameraNear} to carry the far-plane distance and vice versa, so the returned pair is already
     * swapped for the DEPTH_INVERTED flag this context is created with.
     */
    private static CameraPlanes cameraPlanes(float projectionM22, float projectionM32) {
        float far = Math.abs(projectionM22) < 1.0e-9f ? 1.0e9f : -projectionM32 / projectionM22;
        float near = far / (1.0f + projectionM22) * projectionM22;
        if (!(far > 0.0f) || !Float.isFinite(far)) {
            far = 1.0e9f;
        }
        if (!(near > 0.0f) || !Float.isFinite(near) || near > far) {
            near = 0.05f;
        }
        return new CameraPlanes(far, near);
    }

    private static float fovY(float projectionM11) {
        return projectionM11 > 1.0e-6f ? (float) (2.0 * Math.atan(1.0 / projectionM11)) : (float) Math.toRadians(70.0);
    }

    public void destroy() {
        releaseContext();
        lib = null;
        probeAttempted = false;
        available = false;
        failed = false;
        lastFrameNanos = 0;
    }

    private void releaseContext() {
        if (context == null) {
            return;
        }
        try {
            // No frame may still be reading the resources the context owns.
            RtContext ctx = RtContext.currentOrNull();
            if (ctx != null) {
                ctx.waitIdle();
            }
            context.close();
        } catch (Throwable t) {
            CausticaMod.LOGGER.warn("FSR context release failed", t);
        }
        context = null;
        contextRenderWidth = -1;
        contextRenderHeight = -1;
        contextDisplayWidth = -1;
        contextDisplayHeight = -1;
        contextQuality = Integer.MIN_VALUE;
    }

    private static Path locateLibrary() {
        String override = CausticaConfig.Fsr.PATH.get();
        if (override != null && !override.isBlank()) {
            Path path = Path.of(override);
            if (Files.isDirectory(path)) {
                path = path.resolve(NATIVES.libraryName());
            }
            return Files.isRegularFile(path) ? path : null;
        }
        Path bundled = extractBundledLibrary();
        if (bundled != null) {
            return bundled;
        }
        Path gameDir = FabricLoader.getInstance().getGameDir();
        Path[] candidates = {
                gameDir.resolve("natives").resolve(NATIVES.libraryName()),
                gameDir.resolve(NATIVES.libraryName())};
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                return candidate;
            }
        }
        return null;
    }

    private static Path extractBundledLibrary() {
        Path dir = FabricLoader.getInstance().getGameDir().resolve("caustica-fsr")
                .resolve("natives").resolve(NATIVES.platformDir());
        Path library = dir.resolve(NATIVES.libraryName());
        String resource = "/caustica/natives/" + NATIVES.platformDir() + "/" + NATIVES.libraryName();
        try (InputStream in = FsrUpscaler.class.getResourceAsStream(resource)) {
            if (in == null) {
                return null;
            }
            byte[] bytes = in.readAllBytes();
            Files.createDirectories(dir);
            if (!Files.isRegularFile(library) || !sameBytes(library, bytes)) {
                Files.write(library, bytes);
            }
            return library;
        } catch (IOException e) {
            CausticaMod.LOGGER.warn("Could not extract the bundled FSR runtime to {}", dir, e);
            return null;
        }
    }

    private static boolean sameBytes(Path path, byte[] bytes) {
        try {
            return Files.size(path) == bytes.length && Arrays.equals(Files.readAllBytes(path), bytes);
        } catch (IOException e) {
            return false;
        }
    }

    private record PlatformNatives(String platformDir, String libraryName) {
        private static PlatformNatives current() {
            String os = System.getProperty("os.name", "").toLowerCase();
            String arch = System.getProperty("os.arch", "").toLowerCase();
            boolean x64 = arch.equals("x86_64") || arch.equals("amd64");
            if (os.contains("win") && x64) {
                return new PlatformNatives("windows-x64", "amd_fidelityfx_vk.dll");
            }
            if (os.contains("linux") && x64) {
                return new PlatformNatives("linux-x64", "libamd_fidelityfx_vk.so");
            }
            return new PlatformNatives(os + "/" + arch, System.mapLibraryName("amd_fidelityfx_vk"));
        }
    }
}
