[[vk::binding(0, 0)]] [format("rgba16f")] public RWTexture2D<float4> colorImage;
[[vk::binding(1, 0)]] public RWStructuredBuffer<uint, Std430DataLayout> histBins;
[[vk::binding(2, 0)]] [format("r32f")] public RWTexture2D<float> depthImage;
[[vk::binding(3, 0)]] [format("rgba16f")] public RWTexture2D<float4> albedoImage;
// Single-element scratch for the auto-focus distance, in raw ray distance. Cleared here, measured by the
// autofocus pass, and read by exposure_resolve, which publishes it in the exposure state.
[[vk::binding(4, 0)]] public RWStructuredBuffer<float, Std430DataLayout> focusScratch;

