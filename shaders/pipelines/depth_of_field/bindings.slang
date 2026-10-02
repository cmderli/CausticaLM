// Descriptor contract for the depth-of-field pipeline: the single place these numbers are written down, and the
// source of the Java layout and the generated RtBindings.
//
// Each stage declares its OWN bindings locally rather than importing these. Slang emits one SPIR-V module per
// entry point and carries only the globals that entry point actually reaches, so a module that imports the whole
// set gets a RENUMBERED layout: a stage reaching three of ten bindings emits them as 0, 1, 2 and binds its
// resources to the wrong descriptors. Repeating the attributes in each stage is what keeps the numbers stable.
//
//   0 sceneImage        Sampler2D  the reconstructed image: the gather reads it, the prefilter filters it
//   1 gDepth           Sampler2D  render-resolution reversed-Z depth, for the circles of confusion
//   2 gMotion          Sampler2D  render-resolution motion guide, for the history's reprojection
//   3 gatheredImage    Storage    the half-resolution target the gather writes and the prefilter fills
//   4 historyImage     Sampler2D  the previous frame's gather result, and the blur stages' source
//   5 gatheredSampler  Sampler2D  the gather result the upscale reads (what binding 3 held last frame)
//   6 upsampleImage    Storage    the display-resolution image the upscale writes
//   7 sharpSource      Sampler2D  the same reconstructed image at DISPLAY resolution, for the upscale's fallback
//
// Binding 0 and binding 7 are the same scene at two resolutions, and the upscale needs both. The gather reads
// the reconstruction output, whose size depends on the backend, while the upscale writes the frame: a pixel it
// hands back untouched has to come from an image at the resolution it is writing, or an in-focus pixel is a
// magnified render texel beside a display-resolution neighbour and the step between them is a line. That is why
// the artifact only appeared under DLSS -- under DLAA the two are the same size and the step is nothing.
//
// The gather reads the SHARP source at binding 0 and does not filter it: the disc is carried by the sample
// radius, and a tap that reads a prefiltered image stands for more area than it samples, which is blur the disc
// did not ask for. A prefiltered source is the alternative -- sampling a mip pyramid at a level chosen from the
// disc radius, as for instance Blender's EEVEE gather does -- and it buys smoothness at the cost of detail the
// disc did not ask to lose.
//
// Binding 3 is a storage image and binding 5 is the sampled image of the same kind of resource on purpose: the
// gather writes the frame it is producing while the upscale reads the one the gather finished, and a binding
// carries one descriptor type. A descriptor set may not bind one image both ways either, so the two stages are
// always given different images.
