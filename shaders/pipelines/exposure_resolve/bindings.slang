import display_common;

[[vk::binding(0, 0)]] public StructuredBuffer<uint, Std430DataLayout> histBins;
[[vk::binding(1, 0)]] [format("r32f")] public RWTexture2D<float> exposureImage;
[[vk::binding(2, 0)]] public RWStructuredBuffer<ExposureState, Std430DataLayout> stateBuf;
// Autofocus scratch produced by the histogram pass over the same frame: lane 0 is the centre-weighted
// distance sum, lane 1 its weight. Publishing it here keeps the whole controller readable from one buffer.
[[vk::binding(3, 0)]] public StructuredBuffer<float, Std430DataLayout> focusScratch;
