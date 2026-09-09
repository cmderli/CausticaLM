# rayquery-bench

Micro-benchmark that answers one question: **on this device, is a full rayQuery
path tracer viable, or is the renderer better off as a hybrid raster + rayQuery
design?**

It builds a Minecraft-like acceleration structure — one BLAS per 16³ section,
two geometries per BLAS (an opaque solid bucket and an alpha-tested cutout
bucket), one TLAS instance per section with `instanceCustomIndex` carrying the
section index — then measures sustained `rayQuery` throughput for a set of ray
mixes using GPU timestamp queries.

This is the phase-0 experiment for porting Caustica's world pass to a device that
exposes `VK_KHR_ray_query` but not `VK_KHR_ray_tracing_pipeline`.

## What it measures

| run | rays per pixel | what it represents |
|---|---|---|
| `primary-only` | ~1 | camera ray + guide capture |
| `primary+shadow-opaque` | ~2 | `RAY_FLAG_FORCE_OPAQUE` sun shadow (no cutout) |
| `primary+shadow-candidate` | ~2 | candidate-loop shadow with alpha tests (the hybrid shape) |
| `path-1bounce` | ~4 | path tracer with one continuation |
| `path-2bounce` | ~6 | path tracer with two continuations |
| `primary-random` | ~1 | incoherent rays (how much SER-style reordering would help) |
| `path-2bounce-random` | ~6 | incoherent path tracer |

Ray counts are counted in the shader (per-workgroup shared-memory counter, one
global atomic per workgroup), so the reported MRays/s is measured, not derived.

## Build

Both scripts also compile and validate the shader with `glslangValidator` +
`spirv-val`. Requires glslang and spirv-tools.

### Desktop

```bash
./build_desktop.sh
./rayquery-bench --list
```

Needs Vulkan headers and a loader. If the headers are missing they are cloned
into `.deps/` (gitignored).

### Android (arm64)

```bash
export ANDROID_NDK_HOME=/path/to/ndk
./build_android.sh
adb devices
./run_android.sh --tlas-rebuild --incoherent
```

The NDK sysroot already ships the Vulkan headers and the `libvulkan` stub, so no
Vulkan SDK is needed. `run_android.sh` pushes to `/data/local/tmp/rayquery-bench`
and runs headless — no surface, no swapchain, no app scaffolding.

## Options

```
--width N         render width (default 960)
--height N        render height (default 540)
--sections N      number of 16^3 sections / BLAS (default 1024)
--tris N          triangles per section (default 1024)
--frames N        timed frames per config (default 30)
--warmup N        warmup frames (default 5)
--maxT F          ray tmax in blocks (default 10000)
--modes LIST      comma-separated mode filter, e.g. --modes 2,4
--spirv PATH      path to bench.comp.spv (default: beside the binary)
--tlas-rebuild    also measure a per-frame TLAS rebuild
--incoherent      also measure uniform random ray directions
--list            print device capabilities and exit
```

Defaults describe a ~1M-triangle render distance region (1024 sections × 1024
triangles) at 960×540, which is a 0.5× render scale of a 1080p-class phone
display. Budget roughly 130–200 MB of VRAM for that configuration; lower
`--sections` / `--tris` on a constrained device.

## First run on a device

Start small. Building 1024 BLASes covering ~1M triangles is the slow phase on
mobile and can take tens of seconds; the ray tracing itself is fast by
comparison.

```bash
./run_android.sh --list                                   # device caps only, instant
./run_android.sh --sections 256 --tris 512 --frames 10    # smoke test
./run_android.sh --tlas-rebuild --incoherent              # the real measurement
```

stdout is unbuffered and every phase prints progress (`geometry: 256/1024
sections`, `BLAS build done in ...`), so a silent run means the process died, not
that it is busy. `run_android.sh` forwards no timeout: if a run needs to be
stopped, kill it on the device:

```bash
adb shell pkill -f rayquery-bench
```

## Reading the output

```
=== device ===                 apiVersion, driverVersion, timestampPeriod, limits
=== building scene ===         section/triangle/memory totals, BLAS + TLAS build cost
=== throughput ... ===         one line per run: rays/frame, ms/frame, MRays/s
=== verdict at 960x540 ===     achievable fps for the full-PT and hybrid mixes
```

The verdict converts a measured rate into fps:

```
fps = MRays/s × 1e6 / (pixels × rays-per-pixel)
```

Decision rule of thumb, at the render resolution you actually intend to use:

- **full PT fps ≥ 30** → the path tracer is viable as-is (expect the real number
  to be lower: this kernel's shading is intentionally minimal).
- **full PT < 30 but hybrid ≥ 60** → hybrid raster + rayQuery (shadows/AO/GI at
  low res) is the design to pursue.
- **hybrid < 60** → rayQuery is not the lever; stay on raster and revisit later.

Also watch:

- `primary-random` vs `primary-only`: a large gap means ray coherence matters, so
  a shared-memory reorder (the SER substitute) is worth implementing.
- `primary+shadow-opaque` vs `primary+shadow-candidate`: the cost of the
  candidate loop, i.e. what alpha-tested foliage really costs on this device.
- BLAS build time × sections streamed per second: that is the terrain streaming
  budget, which competes with the ray budget for the same GPU.
- `--tlas-rebuild`: Caustica rebuilds the TLAS every frame; if that alone eats a
  large share of the frame, the TLAS strategy needs to change first.

## Caveats

- **Shading is minimal.** Per hit: one UV fetch, one texture sample on cutout
  hits, a few dozen FLOPs. Real material evaluation is more expensive, so the
  implied fps is an **upper bound**.
- **No temporal accumulation.** Each frame traces its own rays; a real renderer
  would amortize over several frames. This measures raw throughput, which is what
  the decision needs.
- **Synthetic geometry.** Height-field terrain with two buckets, not real chunk
  meshes. Triangle density and instance counts are realistic; the distribution is
  not.
- **Timestamps may be unavailable** on some drivers (`timestampValidBits == 0`).
  The benchmark then falls back to wall-clock around submit+fence and marks the
  source column `wall`, which includes host overhead.
- **The shader targets SPIR-V 1.5** (`--target-env vulkan1.2`). If the device
  reports a Vulkan 1.1 `apiVersion`, recompile with `--target-env vulkan1.1` and
  check that `spirv-val` still accepts it — this is also the check that decides
  whether Caustica's `-profile spirv_1_5` shaders can load at all.

## Status

Validated and executed. The shader passes `glslangValidator` + `spirv-val`
(`RayQueryKHR`, no `position_fetch` dependency), the host compiles clean with
`-Wall -Wextra`, and the Android build has been run on a real device.

### Measured: Adreno 830 (Snapdragon 8 Elite), Vulkan 1.3.284, driver 0800.75.1

Scene: 1100 sections / 1.13 M triangles, 960×540 (0.5× of a 1080p display),
GPU timestamps, dispatch only (no per-frame TLAS rebuild):

| config | rays/px | ms/frame | MRays/s |
|---|---|---|---|
| `primary-only` | 1.00 | 3.55 | 145.9 |
| `primary+shadow-opaque` | 1.69 | 5.66 | 155.2 |
| `primary+shadow-candidate` | 1.69 | 6.40 | 137.3 |
| `path-1bounce` | 2.80 | 18.4 | 79.2 |
| `path-2bounce` | 3.49 | 27.7 | 65.2 |
| `primary-random` | 1.00 | 9.60 | 54.0 |
| `path-2bounce-random` | 2.76 | 38.0 | 37.6 |

Derived observations from those runs:

- **Full path tracing is borderline, hybrid is comfortable.** 2-bounce PT costs
  27.7 ms/frame (36 fps) with minimal shading; the hybrid mix costs 6.4 ms
  (156 fps). Real material evaluation, denoising, and the display chain are not
  included in either number.
- **Cost scales ~linearly with pixels.** `path-2bounce` at 480×270 is 7.96 ms
  (126 fps), at 1280×720 it is 48.1 ms (21 fps); throughput stays 57–67 MRays/s.
  Rendering at 0.25× and upscaling buys roughly 4×.
- **Ray coherence is the biggest single lever.** Uniform random directions drop
  primary throughput from 145.9 to 54.0 MRays/s (2.7×), so a shared-memory
  reorder (the SER substitute) is worth implementing.
- **The candidate loop costs ~13 %.** Alpha-tested cutout rays (137.3 MRays/s)
  are slower than force-opaque shadows (155.2 MRays/s) at the same ray count.
- **A per-frame TLAS rebuild costs ~0.7 ms** for 1100 instances (4.27 → 3.55 ms
  on `primary-only` when `--tlas-rebuild` is removed), i.e. ~4 % of a 60 fps
  budget.

Scene size matters as much as the ray mix: the same `path-2bounce` config runs at
200 MRays/s against 294 sections and 65 MRays/s against 1100, so always measure
at a realistic scene size.
