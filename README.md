# CausticaLM

CausticaLM is an experimental ray-traced renderer for Minecraft 26.2's Vulkan backend.
It replaces the vanilla world view with hardware ray tracing and NVIDIA DLSS
features while keeping Minecraft's familiar UI and gameplay intact.

CausticaLM is early software. Expect bugs, missing visual cases, and frequent
changes while the renderer is being built.

![CausticaLM ray-traced Minecraft scene](docs/gallery/2026-07-09_21.25.14.jpg)

## Links

- [Discord](https://discord.gg/SeWCjyKu2)
- [Modrinth](https://modrinth.com/mod/caustica)
- [CurseForge](https://www.curseforge.com/minecraft/mc-mods/caustica/preview)
- [Gallery](docs/gallery.md)

## Features

- Vulkan hardware path-traced world rendering
- Vendor-neutral reconstruction: DLSS Ray Reconstruction on NVIDIA, a built-in denoiser elsewhere
- DLSS Frame Generation support (experimental)
- HDR output
- Dynamic entity rendering in the ray-traced scene
- LabPBR-style material support
- OMM (Opacity Micro-Map) + SER (Shader Execution Reordering) optimizations

## Requirements

- **Vulkan graphics backend enabled**
- A GPU and driver with Vulkan ray tracing support
- NVIDIA RTX GPU and supported driver for DLSS features
- HDR-capable display and OS HDR mode for HDR output
- On Linux, an HDR-capable Wayland compositor and a native Wayland session for HDR output
- Install LabPBR resource pack like [SPBR](https://modrinth.com/resourcepack/spbr) for better visuals

Reconstruction is selectable in Video Settings. `auto` uses DLSS Ray Reconstruction where NGX can run,
then the NRD denoiser when its native library is present, then the built-in denoiser. The built-in denoiser
has no temporal history, so it is softer and noisier than the temporal denoisers at the same sample count.
Turning reconstruction off traces at native resolution and shows the raw path-traced image.

NRD (NVIDIA Real-time Denoisers) is a vendor-neutral compute denoiser used on devices that cannot run DLSS.
It is reached through the `caustica_nrd` native shim, which is built from `native/nrd_shim` and is not part
of the mod jar by default: build it, then put `libcaustica_nrd.so` (or `caustica_nrd.dll`) in `run/natives/`
or point `-Dcaustica.nrd.path` at it. `native/nrd_shim/build_cross.sh` builds the same shim for arm64 Linux
and arm64 Android, and `gradle bundleNrdNatives` packages whichever builds exist. Without it the built-in
denoiser is used instead. NRD is proprietary NVIDIA software; see `THIRD_PARTY_NOTICES.md`.

The built-in denoiser and the NRD backend resolve to display resolution with a bilinear filter unless an
AMD FidelityFX (FSR) Vulkan runtime is present, in which case the Super Resolution Method option can select
it for temporal upscaling. `amd_fidelityfx_vk.dll` is the runtime on Windows; the FidelityFX SDK's Vulkan
backend is a Windows build, so Linux has no `libamd_fidelityfx_vk.so` unless one is built from the SDK
sources and placed in `run/natives/`, the game directory, or `-Dcaustica.fsr.path`. Without it the method
falls back to bilinear and says so in the log.

## Building

```sh
# Mod jar. NRD natives are bundled when native/nrd_shim has been built (see its README).
./gradlew build -x bundleNgxNatives

# Full build with the NVIDIA DLSS/NGX natives, needs the DLSS SDK:
DLSS_SDK=/path/to/DLSS ./gradlew build
```

The jar lands in `build/libs/`. Skipping `bundleNgxNatives` only leaves the DLSS runtime out of the jar:
DLSS features report themselves unavailable and reconstruction falls back to NRD or the built-in denoiser.

## Installation

1. Install Fabric Loader for Minecraft `26.2`.
2. Install Fabric API.
3. Put the CausticaLM jar in your Minecraft `mods` folder.
4. Launch the game with the Vulkan graphics backend.
5. Open Video Settings to adjust CausticaLM's renderer options.

## Usage Notes

- CausticaLM is client-side only.
- DLSS Ray Reconstruction and Frame Generation require supported NVIDIA
  hardware and drivers.
- On Linux if Minecraft crashes on startup with stack overflow errors, try adding `-Xss2M` to the Java args to increase the stack size.
- Use Java args to improve performance. Minecraft Launcher default:
  `-XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC`
- Frame Generation is experimental and needs to be enabled by modifying the configuration file.
- HDR output requires an HDR swapchain and a correctly configured HDR display.
- When HDR is enabled on Linux, CausticaLM selects GLFW's native Wayland backend automatically. X11/XWayland surfaces generally do not expose the required HDR10/PQ format.
- If Minecraft falls back to OpenGL after a crash, re-enable the Vulkan backend
  before using CausticaLM again.

## Compatibility

CausticaLM takes over the world renderer, so other mods that heavily modify world
rendering, shader pipelines, post-processing, or the Vulkan backend may conflict.
UI-only mods are more likely to work.

## Status

CausticaLM is not a finished renderer yet. Current work focuses on visual
correctness, world coverage, stability, and making the SDR/HDR presentation
paths behave consistently.

## License

CausticaLM's project-owned source code and documentation are licensed under the
GNU Lesser General Public License v3.0 or later. See [LICENSE.md](LICENSE.md),
[COPYING](COPYING), and [COPYING.LESSER](COPYING.LESSER).

Release artifacts may bundle NVIDIA DLSS/NGX SDK components under NVIDIA's own
license terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## TODO List

- [ ] Nether/End sky, weather, volumetric fog/clouds
- [ ] LOD
- [ ] ReSTIR
