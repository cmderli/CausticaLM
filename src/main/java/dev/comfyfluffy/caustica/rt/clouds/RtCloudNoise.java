package dev.comfyfluffy.caustica.rt.clouds;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.stream.IntStream;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkMemoryBarrier;

/**
 * The cloud noise fields and baked cloud-body SDF, generated once and published to the world shaders
 * as a BDA buffer.
 *
 * <p>The buffer is a flat array of 32-bit words, one RGBA8 voxel each for the noise fields and one
 * float word per SDF voxel, laid out in the order shape, detail, weather, placement, SDF.
 * {@code clouds.slang} mirrors these offsets in {@code CLOUD_*_BASE}; the two descriptions are one ABI.
 *
 * <ul>
 *   <li><b>Shape</b> 96³ RGBA8 — Perlin-Worley (red) and Worley FBM (green/blue/alpha). Reserved in
 *       the layout; the density function is defined by the SDF below.</li>
 *   <li><b>Detail</b> 128³ RGBA8 — Nubis-derived cumulus turbulence: red billows, green wisps,
 *       blue/alpha a low-frequency distortion pair used to curl the detail lookup.</li>
 *   <li><b>Weather</b> 256² RGBA8 — large-scale moisture that modulates the live coverage.</li>
 *   <li><b>Placement</b> 512² RGBA8 — Nubis-derived per-column cloud model: red cluster field, green
 *       top jitter, blue base lift. It is consumed by the SDF bake, not by the density shader.</li>
 *   <li><b>SDF</b> 128×32×128 float — signed distance to the cloud body in blocks, negative inside,
 *       baked at nominal coverage 0.5 over one 2048-block horizontal tile and the configured slab
 *       thickness. +1000 marks a voxel with no cloud body in reach.</li>
 * </ul>
 *
 * <p>The placement, detail and body model are adapted from NUMOS
 * (https://github.com/Kim2091/numos-sky, Copyright 2026 Kim2091 (Sparkles), Apache-2.0); see
 * {@code THIRD_PARTY_NOTICES.md} and {@code licenses/NUMOS-LICENSE.txt}. The bake is a Java
 * implementation of NUMOS' occupancy → JFA → resolve chain rather than a copy of its compute passes.
 *
 * <p>All fields tile at their world period. The SDF's 2048-block tile, the detail volume and the
 * weather map divide the 4096-block anchor period, so one anchor keeps the sampled fields
 * world-locked across a terrain rebase. Generation runs in parallel at world bring-up and then never
 * again.
 */
public final class RtCloudNoise {
    public static final int SHAPE_DIM = 96;
    public static final int DETAIL_DIM = 128;
    public static final int WEATHER_DIM = 256;
    public static final int PLACEMENT_DIM = 512;

    // Cloud-body SDF grid. X and Z span one placement tile; Y spans the configured cloud slab.
    public static final int NVDF_X = 128;
    public static final int NVDF_Y = 32;
    public static final int NVDF_Z = 128;

    // World period and cluster size of the placement map. The SDF bake spans one such tile horizontally,
    // and the map is sampled at exactly this period.
    public static final int PLACEMENT_TILE_BLOCKS = 2048;
    public static final int PLACEMENT_CELL_BLOCKS = 512;

    static final int SHAPE_WORDS = SHAPE_DIM * SHAPE_DIM * SHAPE_DIM;
    static final int DETAIL_WORDS = DETAIL_DIM * DETAIL_DIM * DETAIL_DIM;
    static final int WEATHER_WORDS = WEATHER_DIM * WEATHER_DIM;
    static final int PLACEMENT_WORDS = PLACEMENT_DIM * PLACEMENT_DIM;
    static final int NVDF_BASE_WORDS = SHAPE_WORDS + DETAIL_WORDS + WEATHER_WORDS + PLACEMENT_WORDS;
    static final int NVDF_WORDS = NVDF_X * NVDF_Y * NVDF_Z;
    static final int TOTAL_WORDS = NVDF_BASE_WORDS + NVDF_WORDS;
    private static final long BYTE_SIZE = (long) TOTAL_WORDS * 4L;

    // Bake parameters. The coverage is the midpoint of the slider range, so the shader can grow and
    // shrink the same bodies with a level-set offset instead of rebaking.
    static final float NVDF_NOMINAL_COVERAGE = 0.5f;
    static final float NVDF_FIELD_BAND_BLOCKS = 40.0f;
    // Runtime sampling constants mirrored in clouds.slang: the level-set shift per unit of live
    // coverage, and how far into the body the density profile ramps.
    static final float NVDF_COVERAGE_OFFSET_BLOCKS = 160.0f;
    static final float NVDF_PROFILE_DEPTH_BLOCKS = 40.0f;
    private static final float NVDF_EMPTY_BLOCKS = 1000.0f;
    private static final float NVDF_BILLOW_SIZE_BLOCKS = 256.0f;
    private static final float NVDF_BILLOW_STRENGTH = 0.75f;
    private static final float NVDF_BILLOW_GAIN = 0.5f;

    private static final int SHAPE_SEED = 0x1a2b3c4d;
    private static final int DETAIL_SEED = 0x51ed270b;
    private static final int WEATHER_SEED = 0x6d2b79f5;

    // The twelve edge directions of a cube, the standard Perlin gradient set.
    private static final int[][] GRADIENTS = {
        {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0},
        {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
        {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1}
    };

    private final RtBuffer buffer;

    private RtCloudNoise(RtBuffer buffer) {
        this.buffer = buffer;
    }

    public long address() {
        return buffer.deviceAddress;
    }

    public void destroy() {
        buffer.destroy();
    }

    public static RtCloudNoise create(RtContext ctx) {
        long started = System.nanoTime();
        int[] shape = buildShape();
        int[] detail = buildDetail();
        int[] weather = buildWeather();
        int[] placement = buildPlacement();
        // The SDF's vertical domain is the configured slab. RtCloudNoise is built with the world
        // pipeline, so a thickness change takes a world reload to reach the bake, as the other noise
        // fields do.
        float thickness = CausticaConfig.Rt.Clouds.THICKNESS.value();
        long nvdfStarted = System.nanoTime();
        float[] nvdf = buildNvdf(placement, thickness);
        long nvdfMillis = (System.nanoTime() - nvdfStarted) / 1_000_000L;
        CausticaMod.LOGGER.info("Baked cloud body SDF {}x{}x{} ({} blocks) in {} ms",
                NVDF_X, NVDF_Y, NVDF_Z, thickness, nvdfMillis);
        int[] nvdfWords = new int[NVDF_WORDS];
        for (int i = 0; i < NVDF_WORDS; i++) {
            nvdfWords[i] = Float.floatToRawIntBits(nvdf[i]);
        }

        RtBuffer upload = ctx.createUploadBuffer(BYTE_SIZE, "cloud noise upload");
        RtBuffer noise = null;
        try {
            RtBuffer created = ctx.createBuffer(BYTE_SIZE, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    false, "cloud noise");
            noise = created;
            ByteBuffer mapped = MemoryUtil.memByteBuffer(upload.mapped, (int) BYTE_SIZE)
                    .order(ByteOrder.nativeOrder());
            IntBuffer words = mapped.asIntBuffer();
            words.position(0);
            words.put(shape);
            words.position(SHAPE_WORDS);
            words.put(detail);
            words.position(SHAPE_WORDS + DETAIL_WORDS);
            words.put(weather);
            words.position(SHAPE_WORDS + DETAIL_WORDS + WEATHER_WORDS);
            words.put(placement);
            words.position(NVDF_BASE_WORDS);
            words.put(nvdfWords);
            upload.flush();

            ctx.submitSync(cmd -> {
                try (MemoryStack stack = MemoryStack.stackPush()) {
                    VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack);
                    region.get(0).srcOffset(0L).dstOffset(0L).size(BYTE_SIZE);
                    VK10.vkCmdCopyBuffer(cmd, upload.handle, created.handle, region);

                    // The copy is on the transfer stage; the trace reads these words from the ray-tracing
                    // stage in a later submission. All-commands on the destination is the conservative
                    // side of that handoff and costs nothing for a one-time upload.
                    VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack);
                    barrier.get(0).sType$Default()
                            .srcAccessMask(VK10.VK_ACCESS_TRANSFER_WRITE_BIT)
                            .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT);
                    VK10.vkCmdPipelineBarrier(cmd, VK10.VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, barrier, null, null);
                }
            });
        } catch (Throwable t) {
            if (noise != null) {
                noise.destroy();
            }
            throw t;
        } finally {
            upload.destroy();
        }
        long elapsedMs = (System.nanoTime() - started) / 1_000_000L;
        CausticaMod.LOGGER.info("Generated cloud noise and body SDF ({} KiB) in {} ms",
                BYTE_SIZE / 1024L, elapsedMs);
        return new RtCloudNoise(noise);
    }

    static int[] buildShape() {
        int[] words = new int[SHAPE_WORDS];
        int size = SHAPE_DIM;
        IntStream.range(0, size).parallel().forEach(z -> {
            double wz = (double) z / size;
            for (int y = 0; y < size; y++) {
                double wy = (double) y / size;
                for (int x = 0; x < size; x++) {
                    double wx = (double) x / size;
                    double perlin = perlinFbm(wx, wy, wz, 4, SHAPE_SEED, 4);
                    double w4 = worley(wx, wy, wz, 4, SHAPE_SEED + 1);
                    double w8 = worley(wx, wy, wz, 8, SHAPE_SEED + 2);
                    double w16 = worley(wx, wy, wz, 16, SHAPE_SEED + 3);
                    double worleyFbm = w4 * 0.625 + w8 * 0.25 + w16 * 0.125;
                    double base = clamp01(remap(perlin, worleyFbm - 1.0, 1.0, 0.0, 1.0));
                    words[(z * size + y) * size + x] = pack(base, w4, w8, w16);
                }
            }
        });
        // Perlin's range is not the byte range: raw values cluster near the middle and would make the
        // shader's coverage threshold select almost the whole sky. Stretching each channel to its own
        // observed range gives the threshold a usable distribution without changing the field's shape.
        for (int channel = 0; channel < 4; channel++) {
            normalizeChannel(words, channel);
        }
        return words;
    }

    static int[] buildDetail() {
        int[] words = new int[DETAIL_WORDS];
        int size = DETAIL_DIM;
        IntStream.range(0, size).parallel().forEach(z -> {
            double pz = (z + 0.5) / size;
            for (int y = 0; y < size; y++) {
                double py = (y + 0.5) / size;
                for (int x = 0; x < size; x++) {
                    double px = (x + 0.5) / size;

                    // Billows: inverted Worley FBM at four cells per volume plus two octaves.
                    double billow = 1.0 - numosWorleyFbm3D(px * 4.0, py * 4.0, pz * 4.0, 3,
                            5.3 + DETAIL_SEED * 1.0e-6, 4.0, 4.0, 4.0) / 0.875;
                    // Wisps: inverted alligator FBM, squeezed vertically so the strands shear.
                    double wispy = Math.pow(clamp01(1.0 - numosAlligatorFbm3D(
                            px * 6.0, py * 12.0, pz * 6.0, 3, 23.1 + DETAIL_SEED * 1.0e-6,
                            6.0, 12.0, 6.0)), 2.0);
                    // A cheap periodic distortion pair. NUMOS uses the curl of a potential; a single
                    // low-frequency Perlin pair is enough to break the detail lookup's regularity
                    // without another twelve FBM evaluations per voxel.
                    double warpX = perlinFbm(px, py, pz, 2, DETAIL_SEED + 11, 3);
                    double warpZ = perlinFbm(px, py, pz, 3, DETAIL_SEED + 12, 3);

                    words[(z * size + y) * size + x] = pack(billow, wispy, warpX, warpZ);
                }
            }
        });
        return words;
    }

    static int[] buildPlacement() {
        int[] words = new int[PLACEMENT_WORDS];
        int size = PLACEMENT_DIM;
        // NUMOS blends the two nearest integer cell counts so cloudCellSize can vary continuously.
        // CausticaLM bakes one fixed pair, so this reduces to a single cluster field.
        double cells = Math.clamp((double) PLACEMENT_TILE_BLOCKS / PLACEMENT_CELL_BLOCKS, 1.0, 32.0);
        IntStream.range(0, size).parallel().forEach(y -> {
            double v = (y + 0.5) / size;
            for (int x = 0; x < size; x++) {
                double u = (x + 0.5) / size;
                double cluster = clusterField2D(u, v, cells);
                double topJitter = topJitter2D(u, v, cells);
                double baseLift = fbm2DPeriodic(u * 3.0, v * 3.0, 2, 3.0, 113.7);
                words[y * size + x] = pack(cluster, topJitter, baseLift, 1.0);
            }
        });
        return words;
    }

    static int[] buildWeather() {
        int[] words = new int[WEATHER_WORDS];
        int size = WEATHER_DIM;
        IntStream.range(0, size).parallel().forEach(y -> {
            double wy = (double) y / size;
            for (int x = 0; x < size; x++) {
                double wx = (double) x / size;
                double value = perlinFbm(wx, wy, 0.37, 4, WEATHER_SEED, 5);
                // Stretch the distribution away from its mean so open sky and solid decks both appear
                // instead of a uniform haze.
                value = clamp01(0.5 + (value - 0.5) * 1.6);
                words[y * size + x] = pack(value, value, value, 1.0);
            }
        });
        normalizeChannel(words, 0);
        // Keep the duplicated lanes in step with red so a dumped buffer reads consistently.
        for (int i = 0; i < words.length; i++) {
            int value = words[i] & 0xff;
            words[i] = value | (value << 8) | (value << 16) | (value << 24);
        }
        return words;
    }

    // ---- Cloud-body SDF bake ---------------------------------------------------------------------
    //
    // Occupancy -> boundary seeds -> jump flooding -> signed resolve, the CPU twin of NUMOS'
    // cloud_nvdf_* passes. The field is baked at NVDF_NOMINAL_COVERAGE; the density shader shifts the
    // iso-level for the live coverage instead of rebaking. Voxel centres map to the placement tile's
    // UV and to the configured slab's height fraction.

    static float[] buildNvdf(int[] placementWords, float thicknessBlocks) {
        double slab = Math.max(thicknessBlocks, 1.0);
        float[] encoded = new float[NVDF_WORDS];
        IntStream.range(0, NVDF_Z).parallel().forEach(z -> {
            double v = (z + 0.5) / NVDF_Z;
            double[] placement = new double[3];
            for (int y = 0; y < NVDF_Y; y++) {
                double hf = (y + 0.5) / NVDF_Y;
                int row = (z * NVDF_Y + y) * NVDF_X;
                for (int x = 0; x < NVDF_X; x++) {
                    double u = (x + 0.5) / NVDF_X;
                    samplePlacement(placementWords, u, v, placement);
                    double field = bodyFieldBlocks(u, hf, v, placement, slab);
                    encoded[row + x] = (float) clamp01(
                            0.5 + field / (2.0 * NVDF_FIELD_BAND_BLOCKS));
                }
            }
        });

        int[] seeds = nvdfSeedBoundary(encoded);
        seeds = nvdfJumpFlood(seeds, thicknessBlocks);
        return nvdfResolve(encoded, seeds, thicknessBlocks);
    }

    // Bilinear placement tap at a tile UV. Texel centres sit at (i + 0.5) / dim, so the lookup is
    // uv * dim - 0.5 and both axes wrap onto the tile torus.
    private static void samplePlacement(int[] words, double u, double v, double[] out) {
        double px = u * PLACEMENT_DIM - 0.5;
        double py = v * PLACEMENT_DIM - 0.5;
        int cx = floor(px);
        int cy = floor(py);
        double fx = px - cx;
        double fy = py - cy;
        int x0 = wrapIndex(cx, PLACEMENT_DIM);
        int y0 = wrapIndex(cy, PLACEMENT_DIM);
        int x1 = wrapIndex(cx + 1, PLACEMENT_DIM);
        int y1 = wrapIndex(cy + 1, PLACEMENT_DIM);
        for (int channel = 0; channel < 3; channel++) {
            int shift = channel * 8;
            double v00 = (words[y0 * PLACEMENT_DIM + x0] >>> shift) & 0xff;
            double v10 = (words[y0 * PLACEMENT_DIM + x1] >>> shift) & 0xff;
            double v01 = (words[y1 * PLACEMENT_DIM + x0] >>> shift) & 0xff;
            double v11 = (words[y1 * PLACEMENT_DIM + x1] >>> shift) & 0xff;
            out[channel] = lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy) * (1.0 / 255.0);
        }
    }

    // Signed cloud-body field in blocks, positive inside. Base and top are per-column height cuts and
    // the side is the cluster waterline; the Worley billows displace the sides and top only, so bases
    // stay flat cuts as in NUMOS' cumulus body field.
    private static double bodyFieldBlocks(double u, double hf, double v, double[] placement,
                                          double slab) {
        double baseFrac = placement[2] * 0.20;
        double topFrac = clamp01(0.55 + placement[1] * 0.40);
        double baseDistance = (hf - baseFrac) * slab;
        double topDistance = (topFrac - hf) * slab;
        double sideDistance = (placement[0] - (1.0 - NVDF_NOMINAL_COVERAGE))
                * 0.5 * PLACEMENT_CELL_BLOCKS;

        double t = clamp01((hf - baseFrac) / Math.max(topFrac - baseFrac, 1.0e-3));
        double growth = lerp(0.25, 1.0, smoothstep(0.0, 0.4, t));
        double displacement = NVDF_BILLOW_STRENGTH * NVDF_BILLOW_SIZE_BLOCKS
                * billow(u, hf, v, slab) * growth;
        return Math.min(baseDistance, Math.min(sideDistance, topDistance) + displacement);
    }

    // NUMOS' body billow: four octaves of (0.5 - Worley F1) at half the size each, normalised. The
    // lattice is periodic and isotropic in blocks, so the lumps do not stretch with the slab.
    private static double billow(double u, double hf, double v, double slab) {
        double cycles = Math.max(1.0,
                Math.round(PLACEMENT_TILE_BLOCKS / NVDF_BILLOW_SIZE_BLOCKS));
        double px = u * cycles;
        double py = hf * slab / PLACEMENT_TILE_BLOCKS * cycles;
        double pz = v * cycles;
        double period = cycles;
        double value = 0.0;
        double amplitude = 1.0;
        double norm = 0.0;
        for (int i = 0; i < 4; i++) {
            value += amplitude * (0.5 - numosWorleyNoise3D(px, py, pz, period, period, period,
                    41.3 + 17.9 * i));
            norm += amplitude;
            amplitude *= NVDF_BILLOW_GAIN;
            px *= 2.0;
            py *= 2.0;
            pz *= 2.0;
            period *= 2.0;
        }
        return value / Math.max(norm, 1.0e-4);
    }

    // Boundary seeds: an occupied voxel with at least one empty 6-neighbour. XZ neighbours wrap; Y
    // outside the slab is empty, so a body touching the slab floor or ceiling ends there.
    private static int[] nvdfSeedBoundary(float[] encoded) {
        int[] seeds = new int[NVDF_WORDS];
        java.util.Arrays.fill(seeds, -1);
        IntStream.range(0, NVDF_Z).parallel().forEach(z -> {
            for (int y = 0; y < NVDF_Y; y++) {
                int row = (z * NVDF_Y + y) * NVDF_X;
                for (int x = 0; x < NVDF_X; x++) {
                    if (encoded[row + x] <= 0.5f) {
                        continue;
                    }
                    boolean boundary = !nvdfOccupied(encoded, x, y, z + 1)
                            || !nvdfOccupied(encoded, x, y, z - 1)
                            || !nvdfOccupied(encoded, x, y + 1, z)
                            || !nvdfOccupied(encoded, x, y - 1, z)
                            || !nvdfOccupied(encoded, x + 1, y, z)
                            || !nvdfOccupied(encoded, x - 1, y, z);
                    if (boundary) {
                        seeds[row + x] = packSeed(x, y, z);
                    }
                }
            }
        });
        return seeds;
    }

    // Jump flooding with NUMOS' schedule (64, 32, 16, 8, 4, 2, 1, plus one extra 1). The 3x3x3
    // neighbourhood wraps in XZ and stops at the slab in Y. Distances are compared in blocks, with the
    // horizontal and vertical voxel sizes kept separate so a non-cubic grid still finds the nearest
    // seed by physical distance.
    private static int[] nvdfJumpFlood(int[] seedInit, float thicknessBlocks) {
        float voxelXZ = (float) PLACEMENT_TILE_BLOCKS / NVDF_X;
        float voxelY = Math.max(thicknessBlocks, 1.0f) / NVDF_Y;
        int[] jumps = {64, 32, 16, 8, 4, 2, 1, 1};
        int[] seeds = seedInit;
        int[] scratch = new int[NVDF_WORDS];
        for (int jump : jumps) {
            final int step = jump;
            final int[] source = seeds;
            final int[] destination = scratch;
            IntStream.range(0, NVDF_Z).parallel().forEach(z -> {
                for (int y = 0; y < NVDF_Y; y++) {
                    int row = (z * NVDF_Y + y) * NVDF_X;
                    for (int x = 0; x < NVDF_X; x++) {
                        int index = row + x;
                        int bestSeed = source[index];
                        float bestDistance = bestSeed < 0 ? Float.MAX_VALUE
                                : nvdfDistanceSq(x, y, z, bestSeed, voxelXZ, voxelY);
                        for (int dz = -1; dz <= 1; dz++) {
                            int nz = wrapIndex(z + dz * step, NVDF_Z);
                            for (int dy = -1; dy <= 1; dy++) {
                                int ny = y + dy * step;
                                if (ny < 0 || ny >= NVDF_Y) {
                                    continue;
                                }
                                int neighbourRow = (nz * NVDF_Y + ny) * NVDF_X;
                                for (int dx = -1; dx <= 1; dx++) {
                                    if (dx == 0 && dy == 0 && dz == 0) {
                                        continue;
                                    }
                                    int nx = wrapIndex(x + dx * step, NVDF_X);
                                    int candidate = source[neighbourRow + nx];
                                    if (candidate < 0) {
                                        continue;
                                    }
                                    float distance = nvdfDistanceSq(x, y, z, candidate,
                                            voxelXZ, voxelY);
                                    if (distance < bestDistance) {
                                        bestDistance = distance;
                                        bestSeed = candidate;
                                    }
                                }
                            }
                        }
                        destination[index] = bestSeed;
                    }
                }
            });
            seeds = destination;
            scratch = source;
        }
        return seeds;
    }

    // Signed resolve in blocks: the sign comes from occupancy, the magnitude from the nearest boundary
    // seed. Near the surface the encoded body field's continuous distance takes over, which removes
    // the voxel-centre staircase from the profile ramp. A voxel with no seed is empty air.
    private static float[] nvdfResolve(float[] encoded, int[] seeds, float thicknessBlocks) {
        float band = NVDF_FIELD_BAND_BLOCKS;
        float voxelXZ = (float) PLACEMENT_TILE_BLOCKS / NVDF_X;
        float voxelY = Math.max(thicknessBlocks, 1.0f) / NVDF_Y;
        float[] sdf = new float[NVDF_WORDS];
        IntStream.range(0, NVDF_Z).parallel().forEach(z -> {
            for (int y = 0; y < NVDF_Y; y++) {
                int row = (z * NVDF_Y + y) * NVDF_X;
                for (int x = 0; x < NVDF_X; x++) {
                    int index = row + x;
                    int seed = seeds[index];
                    if (seed < 0) {
                        sdf[index] = NVDF_EMPTY_BLOCKS;
                        continue;
                    }
                    float distance = (float) Math.sqrt(
                            nvdfDistanceSq(x, y, z, seed, voxelXZ, voxelY));
                    boolean inside = encoded[index] > 0.5f;
                    float fieldDistance = -(encoded[index] - 0.5f) * (2.0f * band);
                    float jfaDistance = inside ? -distance : distance;
                    float weight = (float) smoothstep(0.5 * band, 0.9 * band, distance);
                    sdf[index] = fieldDistance + (jfaDistance - fieldDistance) * weight;
                }
            }
        });
        return sdf;
    }

    private static boolean nvdfOccupied(float[] encoded, int x, int y, int z) {
        if (y < 0 || y >= NVDF_Y) {
            return false;
        }
        int index = (wrapIndex(z, NVDF_Z) * NVDF_Y + y) * NVDF_X + wrapIndex(x, NVDF_X);
        return encoded[index] > 0.5f;
    }

    private static int packSeed(int x, int y, int z) {
        return x | (y << 8) | (z << 16);
    }

    // Squared distance in blocks between two voxel centres, wrapping XZ and keeping the slab's
    // vertical scale. Seeds pack their coordinates as x | y<<8 | z<<16.
    private static float nvdfDistanceSq(int x, int y, int z, int seed,
                                        float voxelXZ, float voxelY) {
        int sx = seed & 0xff;
        int sy = (seed >>> 8) & 0x1f;
        int sz = (seed >>> 16) & 0xff;
        int dx = Math.abs(x - sx);
        dx = Math.min(dx, NVDF_X - dx);
        int dz = Math.abs(z - sz);
        dz = Math.min(dz, NVDF_Z - dz);
        float fx = dx * voxelXZ;
        float fy = (y - sy) * voxelY;
        float fz = dz * voxelXZ;
        return fx * fx + fy * fy + fz * fz;
    }

    private static int wrapIndex(int value, int size) {
        int wrapped = value % size;
        return wrapped < 0 ? wrapped + size : wrapped;
    }

    private static double smoothstep(double edge0, double edge1, double value) {
        double t = clamp01((value - edge0) / (edge1 - edge0));
        return t * t * (3.0 - 2.0 * t);
    }

    // ---- NUMOS-derived placement and turbulence helpers -----------------------------------------
    //
    // These are direct numeric ports of the periodic noise used by NUMOS' cloud placement map and
    // cumulus turbulence bakers. All lattices wrap at integer periods, which is what makes the
    // sampled fields tile at the shader's world periods.

    private static double numosHash31(double x, double y, double z) {
        int qx = (int) (x * 1597.0);
        int qy = (int) (y * 2549.0);
        int qz = (int) (z * 3571.0);
        int n = (qx ^ qy ^ qz) * 0x9e3779b1;
        n = (n >>> 16) ^ n;
        return Integer.toUnsignedLong(n) / 4294967296.0;
    }

    private static double numosWorleyFeature(double cx, double cy, double cz, double seed, int axis) {
        double h;
        if (axis == 0) {
            h = cx * 127.1 + cy * 311.7 + cz * 74.7;
        } else if (axis == 1) {
            h = cx * 269.5 + cy * 183.3 + cz * 246.1;
        } else {
            h = cx * 113.5 + cy * 271.9 + cz * 124.6;
        }
        return frac(Math.sin(h + seed) * 43758.5453);
    }

    private static double numosWorleyNoise3D(double x, double y, double z,
                                             double periodX, double periodY, double periodZ,
                                             double seed) {
        double xf = floor(x);
        double yf = floor(y);
        double zf = floor(z);
        double px = x - xf;
        double py = y - yf;
        double pz = z - zf;
        double minDistSq = 4.0;
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    double cx = wrap(xf + dx, periodX);
                    double cy = wrap(yf + dy, periodY);
                    double cz = wrap(zf + dz, periodZ);
                    double fx = numosWorleyFeature(cx, cy, cz, seed, 0);
                    double fy = numosWorleyFeature(cx, cy, cz, seed, 1);
                    double fz = numosWorleyFeature(cx, cy, cz, seed, 2);
                    double ox = dx + fx - px;
                    double oy = dy + fy - py;
                    double oz = dz + fz - pz;
                    minDistSq = Math.min(minDistSq, ox * ox + oy * oy + oz * oz);
                }
            }
        }
        return clamp01(Math.sqrt(minDistSq));
    }

    private static double numosWorleyFbm3D(double x, double y, double z, int octaves, double seed,
                                           double periodX, double periodY, double periodZ) {
        double value = 0.0;
        double amplitude = 0.5;
        for (int i = 0; i < octaves && i < 4; i++) {
            value += amplitude * numosWorleyNoise3D(x, y, z, periodX, periodY, periodZ,
                    seed + i * 19.07);
            x *= 2.0;
            y *= 2.0;
            z *= 2.0;
            periodX *= 2.0;
            periodY *= 2.0;
            periodZ *= 2.0;
            amplitude *= 0.5;
        }
        // Amplitude sum is 0.875; the billow channel divides by exactly that, as NUMOS does.
        return value;
    }

    private static double numosAlligator3D(double x, double y, double z, double seed,
                                           double periodX, double periodY, double periodZ) {
        double xf = floor(x);
        double yf = floor(y);
        double zf = floor(z);
        double px = x - xf;
        double py = y - yf;
        double pz = z - zf;
        double m1 = 0.0;
        double m2 = 0.0;
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    double cx = wrap(xf + dx, periodX);
                    double cy = wrap(yf + dy, periodY);
                    double cz = wrap(zf + dz, periodZ);
                    double fx = numosHash31(cx + seed + 11.317, cy + seed, cz + seed);
                    double fy = numosHash31(cx + seed, cy + seed + 47.113, cz + seed);
                    double fz = numosHash31(cx + seed, cy + seed, cz + seed + 73.951);
                    double amp = 0.25 + 0.75 * numosHash31(cx + seed + 5.771,
                            cy + seed + 9.043, cz + seed + 1.229);
                    double ox = dx + fx - px;
                    double oy = dy + fy - py;
                    double oz = dz + fz - pz;
                    double d = Math.sqrt(ox * ox + oy * oy + oz * oz);
                    double k = amp * Math.pow(clamp01(1.0 - d * 0.7071), 1.6);
                    m2 = Math.max(m2, Math.min(m1, k));
                    m1 = Math.max(m1, k);
                }
            }
        }
        return clamp01((m1 - m2) * 1.25);
    }

    private static double numosAlligatorFbm3D(double x, double y, double z, int octaves, double seed,
                                              double periodX, double periodY, double periodZ) {
        double value = 0.0;
        double amplitude = 0.5;
        double total = 0.0;
        for (int i = 0; i < octaves; i++) {
            value += amplitude * numosAlligator3D(x, y, z, seed + i * 13.7,
                    periodX, periodY, periodZ);
            total += amplitude;
            x *= 2.0;
            y *= 2.0;
            z *= 2.0;
            periodX *= 2.0;
            periodY *= 2.0;
            periodZ *= 2.0;
            amplitude *= 0.5;
        }
        return value / Math.max(total, 1.0e-4);
    }

    // NUMOS' periodic 2D cloud placement noise.
    private static double numosLatticeHash2D(double cx, double cy, double seed) {
        int n = (int) (cx) * 0x9e3779b9 + (int) (cy) * 0x85ebca6b
                + (int) (seed * 3571.0) * 0xc2b2ae35;
        n ^= n >>> 16;
        n *= 0x7feb352d;
        n ^= n >>> 15;
        n *= 0x846ca68b;
        n ^= n >>> 16;
        return Integer.toUnsignedLong(n) / 4294967296.0;
    }

    static double valueNoise2DPeriodic(double x, double y, double period, double seed) {
        double cellX = floor(x);
        double cellY = floor(y);
        double fx = x - cellX;
        double fy = y - cellY;
        fx = fx * fx * (3.0 - 2.0 * fx);
        fy = fy * fy * (3.0 - 2.0 * fy);
        double c00x = wrap(cellX, period);
        double c00y = wrap(cellY, period);
        double c10x = wrap(cellX + 1.0, period);
        double c10y = c00y;
        double c01x = c00x;
        double c01y = wrap(cellY + 1.0, period);
        double c11x = c10x;
        double c11y = c01y;
        double v00 = numosLatticeHash2D(c00x, c00y, seed);
        double v10 = numosLatticeHash2D(c10x, c10y, seed);
        double v01 = numosLatticeHash2D(c01x, c01y, seed);
        double v11 = numosLatticeHash2D(c11x, c11y, seed);
        return lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
    }

    private static double fbm2DPeriodic(double x, double y, int octaves, double period, double seed) {
        double sum = 0.0;
        double amplitude = 0.5;
        double total = 0.0;
        double freq = 1.0;
        for (int i = 0; i < octaves; i++) {
            sum += amplitude * valueNoise2DPeriodic(x * freq, y * freq, period * freq,
                    seed + i * 13.7);
            total += amplitude;
            amplitude *= 0.5;
            freq *= 2.0;
        }
        return sum / Math.max(total, 1.0e-4);
    }

    static double worleyNoise2DPeriodic(double x, double y, double period, double seed) {
        double cellX = floor(x);
        double cellY = floor(y);
        double fx = x - cellX;
        double fy = y - cellY;
        double minDistSq = 8.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                double nx = dx;
                double ny = dy;
                double wrappedX = wrap(cellX + nx, period);
                double wrappedY = wrap(cellY + ny, period);
                double fpx = numosLatticeHash2D(wrappedX, wrappedY, seed);
                double fpy = numosLatticeHash2D(wrappedX, wrappedY, seed + 31.7);
                double ox = nx + fpx - fx;
                double oy = ny + fpy - fy;
                minDistSq = Math.min(minDistSq, ox * ox + oy * oy);
            }
        }
        return clamp01(Math.sqrt(minDistSq));
    }

    private static double clusterField2D(double u, double v, double cells) {
        double w = worleyNoise2DPeriodic(u * cells, v * cells, cells, 11.3);
        double clusterCore = 1.0 - w;
        double shapeFbm = fbm2DPeriodic(u * cells, v * cells, 3, cells, 53.1);
        return clamp01(0.5 + (clusterCore - 0.45) * 0.95 + (shapeFbm - 0.5) * 0.65);
    }

    private static double topJitter2D(double u, double v, double cells) {
        double cellsJ = Math.max(Math.round(cells * 0.5), 2.0);
        return fbm2DPeriodic(u * cellsJ, v * cellsJ, 2, cellsJ, 71.9);
    }

    private static double wrap(double value, double period) {
        return value - period * Math.floor(value / period);
    }

    private static double frac(double value) {
        return value - Math.floor(value);
    }

    private static void normalizeChannel(int[] words, int channel) {
        int shift = channel * 8;
        int min = 255;
        int max = 0;
        for (int word : words) {
            int value = (word >>> shift) & 0xff;
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        if (max - min < 8) {
            return;
        }
        float scale = 255.0f / (max - min);
        int clear = ~(0xff << shift);
        for (int i = 0; i < words.length; i++) {
            int word = words[i];
            int value = (word >>> shift) & 0xff;
            int normalized = Math.round((value - min) * scale);
            words[i] = (word & clear) | (Math.clamp(normalized, 0, 255) << shift);
        }
    }

    private static double perlinFbm(double x, double y, double z, int basePeriod, int seed, int octaves) {
        double sum = 0.0;
        double amplitude = 0.5;
        double total = 0.0;
        for (int octave = 0; octave < octaves; octave++) {
            int period = basePeriod << octave;
            sum += amplitude * perlin(x * period, y * period, z * period, period, seed + octave);
            total += amplitude;
            amplitude *= 0.5;
        }
        // Perlin's practical range is about [-0.7, 0.7]; map it onto [0,1] and let the caller clamp.
        return 0.5 + 0.75 * sum / total;
    }

    static double perlin(double x, double y, double z, int period, int seed) {
        int xi = floor(x);
        int yi = floor(y);
        int zi = floor(z);
        double xf = x - xi;
        double yf = y - yi;
        double zf = z - zi;

        int mask = period - 1;
        int x0 = xi & mask;
        int y0 = yi & mask;
        int z0 = zi & mask;
        int x1 = (xi + 1) & mask;
        int y1 = (yi + 1) & mask;
        int z1 = (zi + 1) & mask;

        double u = fade(xf);
        double v = fade(yf);
        double w = fade(zf);

        double n000 = gradDot(x0, y0, z0, xf, yf, zf, seed);
        double n100 = gradDot(x1, y0, z0, xf - 1.0, yf, zf, seed);
        double n010 = gradDot(x0, y1, z0, xf, yf - 1.0, zf, seed);
        double n110 = gradDot(x1, y1, z0, xf - 1.0, yf - 1.0, zf, seed);
        double n001 = gradDot(x0, y0, z1, xf, yf, zf - 1.0, seed);
        double n101 = gradDot(x1, y0, z1, xf - 1.0, yf, zf - 1.0, seed);
        double n011 = gradDot(x0, y1, z1, xf, yf - 1.0, zf - 1.0, seed);
        double n111 = gradDot(x1, y1, z1, xf - 1.0, yf - 1.0, zf - 1.0, seed);

        double nx00 = lerp(n000, n100, u);
        double nx10 = lerp(n010, n110, u);
        double nx01 = lerp(n001, n101, u);
        double nx11 = lerp(n011, n111, u);
        return lerp(lerp(nx00, nx10, v), lerp(nx01, nx11, v), w);
    }

    private static double gradDot(int x, int y, int z, double dx, double dy, double dz, int seed) {
        int hash = hash(x, y, z, seed);
        int[] gradient = GRADIENTS[(hash & 0x7fffffff) % GRADIENTS.length];
        return gradient[0] * dx + gradient[1] * dy + gradient[2] * dz;
    }

    /** F1 Worley distance, normalised so a cell centre is one and a cell boundary is zero. */
    static double worley(double x, double y, double z, int period, int seed) {
        int xi = floor(x);
        int yi = floor(y);
        int zi = floor(z);
        int mask = period - 1;
        double nearest = Double.MAX_VALUE;
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int cx = (xi + dx) & mask;
                    int cy = (yi + dy) & mask;
                    int cz = (zi + dz) & mask;
                    int hash = hash(cx, cy, cz, seed);
                    double jx = ((hash & 1023) + 0.5) * (1.0 / 1024.0);
                    double jy = (((hash >>> 10) & 1023) + 0.5) * (1.0 / 1024.0);
                    double jz = (((hash >>> 20) & 1023) + 0.5) * (1.0 / 1024.0);
                    double px = xi + dx + jx - x;
                    double py = yi + dy + jy - y;
                    double pz = zi + dz + jz - z;
                    nearest = Math.min(nearest, px * px + py * py + pz * pz);
                }
            }
        }
        return 1.0 - Math.min(1.0, Math.sqrt(nearest) / 0.75);
    }

    private static int hash(int x, int y, int z, int seed) {
        int h = seed;
        h = h * 0x27d4eb2d + x * 0x165667b1;
        h = (h ^ (h >>> 13)) * 0x2c1b3c6d;
        h += y * 0x9e3779b1;
        h = (h ^ (h >>> 16)) * 0x85ebca6b;
        h += z * 0x297a2d39;
        h ^= h >>> 15;
        return h;
    }

    private static int floor(double value) {
        int i = (int) value;
        return value < i ? i - 1 : i;
    }

    private static double fade(double t) {
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    }

    private static double lerp(double a, double b, double t) {
        return a + (b - a) * t;
    }

    private static double remap(double value, double low, double high, double outLow, double outHigh) {
        return outLow + (value - low) / (high - low) * (outHigh - outLow);
    }

    private static double clamp01(double value) {
        return Math.clamp(value, 0.0, 1.0);
    }

    private static int pack(double r, double g, double b, double a) {
        return channel(r) | (channel(g) << 8) | (channel(b) << 16) | (channel(a) << 24);
    }

    private static int channel(double value) {
        return (int) Math.round(clamp01(value) * 255.0);
    }
}
