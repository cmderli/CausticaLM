// Motion blur inputs: the reconstructed display-resolution scene plus the render-resolution guides that say
// where each pixel's surface was during the shutter interval.
//
// The scene is scene-linear ACEScg, so streaks of bright emitters carry their real radiance and the display
// transform sees the integrated image. Motion is the renderer's own guide: render-pixel velocity with
// "previous - current" sign, which is the direction a surface came from.
[[vk::binding(0, 0)]] public Sampler2D sceneImage;
[[vk::binding(1, 0)]] public Sampler2D gMotion;
[[vk::binding(2, 0)]] public Sampler2D gDepth;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> outputImage;
