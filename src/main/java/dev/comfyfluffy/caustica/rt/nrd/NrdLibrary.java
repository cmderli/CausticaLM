package dev.comfyfluffy.caustica.rt.nrd;

import org.joml.Matrix4fc;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * FFM bindings for the caustica_nrd shim, the flat C ABI that exposes NVIDIA's Real-time Denoisers to this
 * renderer. The shim owns every Vulkan object NRD needs and reads only {@code vkGetDeviceProcAddr}; this
 * class owns the struct layouts and the three calls.
 *
 * <p>The struct layouts are hand-laid-out against the shim's header (see {@code native/nrd_shim}): every
 * offset below is annotated with the C member it corresponds to, and {@link #FRAME_INFO_BYTES} is asserted
 * against the struct's natural 8-byte alignment. NRD's own headers never appear here.
 *
 * <p>Matrix storage is column-major with column-vector usage ({@code clip = matrix * position}), which is
 * exactly JOML's memory order, so {@link #putMatrix} writes the elements in their natural order.
 */
public final class NrdLibrary {
    /** Only "0 == success" is contractual; the shim returns negative codes otherwise. */
    public static final int OK = 0;

    /** {@code CausticaNrdCreateInfo}: 3 pointers + 4 u32 = 24 + 16 bytes. */
    public static final int CREATE_INFO_BYTES = 40;
    /**
     * {@code CausticaNrdFrameInfo}: 24 bytes of scalars, four 16-float matrices, seven {image,view} pairs,
     * then the trailing worldPrevToWorld and worldToViewPrev matrices.
     */
    public static final int FRAME_INFO_BYTES = 520;

    // CausticaNrdFrameInfo offsets.
    private static final int F_COMMAND_BUFFER = 0;
    private static final int F_FRAME_INDEX = 8;
    private static final int F_JITTER_X = 12;
    private static final int F_JITTER_Y = 16;
    private static final int F_DENOISING_RANGE = 20;
    private static final int F_VIEW_TO_CLIP = 24;
    private static final int F_CLIP_TO_VIEW = 88;
    private static final int F_WORLD_TO_VIEW = 152;
    private static final int F_VIEW_TO_WORLD = 216;
    private static final int F_VIEW_Z = 280;
    private static final int F_NORMAL_ROUGHNESS = 296;
    private static final int F_MOTION = 312;
    private static final int F_DIFF_IN = 328;
    private static final int F_SPEC_IN = 344;
    private static final int F_DIFF_OUT = 360;
    private static final int F_SPEC_OUT = 376;
    private static final int F_WORLD_PREV_TO_WORLD = 392;
    private static final int F_WORLD_TO_VIEW_PREV = 456;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final FunctionDescriptor CREATE_DESC =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
    private static final FunctionDescriptor DENOISE_DESC =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS);
    private static final FunctionDescriptor DESTROY_DESC = FunctionDescriptor.ofVoid();

    private final Path source;
    private final MethodHandle create;
    private final MethodHandle denoise;
    private final MethodHandle destroy;

    private NrdLibrary(Path source, SymbolLookup lookup) {
        this.source = source;
        this.create = LINKER.downcallHandle(find(lookup, "caustica_nrd_create"), CREATE_DESC);
        this.denoise = LINKER.downcallHandle(find(lookup, "caustica_nrd_denoise"), DENOISE_DESC);
        this.destroy = LINKER.downcallHandle(find(lookup, "caustica_nrd_destroy"), DESTROY_DESC);
    }

    private static MemorySegment find(SymbolLookup lookup, String name) {
        return lookup.find(name).orElseThrow(
                () -> new IllegalStateException("caustica_nrd shim does not export " + name));
    }

    /** Loads the shim from the given path. The library stays loaded for the lifetime of the JVM. */
    public static NrdLibrary load(Path library) {
        return new NrdLibrary(library, SymbolLookup.libraryLookup(library, Arena.global()));
    }

    public Path source() {
        return source;
    }

    /** @param createInfo a {@link #CREATE_INFO_BYTES}-byte {@code CausticaNrdCreateInfo} */
    public int create(MemorySegment createInfo) {
        try {
            return (int) create.invokeExact(createInfo);
        } catch (Throwable t) {
            throw new IllegalStateException("caustica_nrd_create invocation failed", t);
        }
    }

    /** @param frameInfo a {@link #FRAME_INFO_BYTES}-byte {@code CausticaNrdFrameInfo} */
    public int denoise(MemorySegment frameInfo) {
        try {
            return (int) denoise.invokeExact(frameInfo);
        } catch (Throwable t) {
            throw new IllegalStateException("caustica_nrd_denoise invocation failed", t);
        }
    }

    public void destroyInstance() {
        try {
            destroy.invokeExact();
        } catch (Throwable t) {
            throw new IllegalStateException("caustica_nrd_destroy invocation failed", t);
        }
    }

    // ---- struct writing -------------------------------------------------------------------------

    public static void putCreateInfo(MemorySegment info, long device, long physicalDevice,
                                     long getDeviceProcAddr, int renderWidth, int renderHeight, int denoiser) {
        info.asSlice(0, CREATE_INFO_BYTES).fill((byte) 0);
        info.set(ValueLayout.JAVA_LONG, 0, device);
        info.set(ValueLayout.JAVA_LONG, 8, physicalDevice);
        info.set(ValueLayout.JAVA_LONG, 16, getDeviceProcAddr);
        info.set(ValueLayout.JAVA_INT, 24, renderWidth);
        info.set(ValueLayout.JAVA_INT, 28, renderHeight);
        info.set(ValueLayout.JAVA_INT, 32, denoiser);
    }

    public static void putFrameInfo(MemorySegment info, long commandBuffer, int frameIndex,
                                    float jitterX, float jitterY, float denoisingRange,
                                    Matrix4fc viewToClip, Matrix4fc clipToView,
                                    Matrix4fc worldToView, Matrix4fc viewToWorld,
                                    long viewZImage, long viewZView,
                                    long normalRoughnessImage, long normalRoughnessView,
                                    long motionImage, long motionView,
                                    long diffInImage, long diffInView,
                                    long specInImage, long specInView,
                                    long diffOutImage, long diffOutView,
                                    long specOutImage, long specOutView,
                                    Matrix4fc worldPrevToWorld, Matrix4fc worldToViewPrev) {
        info.asSlice(0, FRAME_INFO_BYTES).fill((byte) 0);
        info.set(ValueLayout.JAVA_LONG, F_COMMAND_BUFFER, commandBuffer);
        info.set(ValueLayout.JAVA_INT, F_FRAME_INDEX, frameIndex);
        info.set(ValueLayout.JAVA_FLOAT, F_JITTER_X, jitterX);
        info.set(ValueLayout.JAVA_FLOAT, F_JITTER_Y, jitterY);
        info.set(ValueLayout.JAVA_FLOAT, F_DENOISING_RANGE, denoisingRange);
        putMatrix(info, F_VIEW_TO_CLIP, viewToClip);
        putMatrix(info, F_CLIP_TO_VIEW, clipToView);
        putMatrix(info, F_WORLD_TO_VIEW, worldToView);
        putMatrix(info, F_VIEW_TO_WORLD, viewToWorld);
        putImageView(info, F_VIEW_Z, viewZImage, viewZView);
        putImageView(info, F_NORMAL_ROUGHNESS, normalRoughnessImage, normalRoughnessView);
        putImageView(info, F_MOTION, motionImage, motionView);
        putImageView(info, F_DIFF_IN, diffInImage, diffInView);
        putImageView(info, F_SPEC_IN, specInImage, specInView);
        putImageView(info, F_DIFF_OUT, diffOutImage, diffOutView);
        putImageView(info, F_SPEC_OUT, specOutImage, specOutView);
        putMatrix(info, F_WORLD_PREV_TO_WORLD, worldPrevToWorld);
        putMatrix(info, F_WORLD_TO_VIEW_PREV, worldToViewPrev);
    }

    /**
     * Writes a JOML matrix in its natural element order. NRD wants column-major storage with column-vector
     * usage, which is exactly what JOML already holds, so no transpose happens here.
     */
    public static void putMatrix(MemorySegment info, int offset, Matrix4fc m) {
        info.set(ValueLayout.JAVA_FLOAT, offset, m.m00());
        info.set(ValueLayout.JAVA_FLOAT, offset + 4, m.m01());
        info.set(ValueLayout.JAVA_FLOAT, offset + 8, m.m02());
        info.set(ValueLayout.JAVA_FLOAT, offset + 12, m.m03());
        info.set(ValueLayout.JAVA_FLOAT, offset + 16, m.m10());
        info.set(ValueLayout.JAVA_FLOAT, offset + 20, m.m11());
        info.set(ValueLayout.JAVA_FLOAT, offset + 24, m.m12());
        info.set(ValueLayout.JAVA_FLOAT, offset + 28, m.m13());
        info.set(ValueLayout.JAVA_FLOAT, offset + 32, m.m20());
        info.set(ValueLayout.JAVA_FLOAT, offset + 36, m.m21());
        info.set(ValueLayout.JAVA_FLOAT, offset + 40, m.m22());
        info.set(ValueLayout.JAVA_FLOAT, offset + 44, m.m23());
        info.set(ValueLayout.JAVA_FLOAT, offset + 48, m.m30());
        info.set(ValueLayout.JAVA_FLOAT, offset + 52, m.m31());
        info.set(ValueLayout.JAVA_FLOAT, offset + 56, m.m32());
        info.set(ValueLayout.JAVA_FLOAT, offset + 60, m.m33());
    }

    private static void putImageView(MemorySegment info, int offset, long image, long view) {
        info.set(ValueLayout.JAVA_LONG, offset, image);
        info.set(ValueLayout.JAVA_LONG, offset + 8, view);
    }

    public static String describeReturn(int code) {
        return switch (code) {
            case 0 -> "CAUSTICA_NRD_OK";
            case -1 -> "bad arguments or no instance";
            case -2 -> "Vulkan object creation failed";
            case -3 -> "an nrd:: call failed";
            case -4 -> "internal error";
            default -> "unknown (" + code + ")";
        };
    }
}
