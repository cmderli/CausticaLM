package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import dev.comfyfluffy.caustica.rt.gen.DenoisePushData;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.comfyfluffy.caustica.rt.RtContext.check;
import static dev.comfyfluffy.caustica.rt.pipeline.RtBindings.*;

/**
 * Edge-aware spatial denoiser for the render-resolution path-traced image: a small a-trous filter chain
 * whose taps follow the guide buffers, followed by a bilinear resolve into the display-resolution output.
 *
 * <p>It exists for devices that cannot run a vendor reconstruction feature. It is not a replacement for
 * one: with no temporal history there is nothing to accumulate against, so it trades some sharpness for
 * variance and leaves mirror-like surfaces untouched rather than blurring a reflection that has no
 * neighbourhood to average with.
 *
 * <p>Each iteration is a separate dispatch over the same shader with a doubling tap spacing, and the four
 * steps get their own descriptor set — binding the src/dst pair per step keeps the shader free of dynamic
 * storage-image indexing, which would need {@code shaderStorageImageArrayDynamicIndexing}, a feature the
 * device bring-up does not ask for.
 */
public final class RtDenoisePipeline {
    private static final String SHADER = "/caustica/shaders/pipelines/denoise/main.comp.spv";
    /** A-trous iterations; tap spacing doubles each time (1, 2, 4 render texels). */
    public static final int ITERATIONS = 3;
    /** Steps are the iterations plus the resolve, which writes the display-res output. */
    private static final int RESOLVE_SET = ITERATIONS;
    private static final int SET_COUNT = ITERATIONS + 1;

    private static final int MODE_ATROUS = 0;
    private static final int MODE_RESOLVE = 1;

    // Filter tolerances. Wide enough to remove one-sample-per-pixel Monte-Carlo variance, tight enough
    // that a depth, normal, albedo or roughness discontinuity still stops the taps.
    private static final float DEPTH_SIGMA = 0.10f;
    private static final float NORMAL_POWER = 24.0f;
    private static final float ALBEDO_SIGMA = 0.10f;
    private static final float LUMA_SIGMA = 0.30f;
    private static final float MIRROR_ROUGHNESS = 0.10f;
    private static final float FIREFLY_CLAMP = 8.0f;

    private final RtContext ctx;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long[] descriptorSets;
    private final long pipelineLayout;
    private final long pipeline;
    private final long sampler;

    private final long[] boundSources = new long[SET_COUNT];
    private final long[] boundTargets = new long[SET_COUNT];
    private long boundDepthView;
    private long boundNormalView;
    private long boundAlbedoView;
    private long boundSpecAlbedoView;
    private boolean destroyed;

    private RtDenoisePipeline(RtContext ctx, long descriptorSetLayout, long descriptorPool,
                              long[] descriptorSets, long pipelineLayout, long pipeline, long sampler) {
        this.ctx = ctx;
        this.descriptorSetLayout = descriptorSetLayout;
        this.descriptorPool = descriptorPool;
        this.descriptorSets = descriptorSets;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
        this.sampler = sampler;
    }

    public static RtDenoisePipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer bindings =
                    VkDescriptorSetLayoutBinding.calloc(DENOISE_BINDING_COUNT, stack);
            bindings.get(DENOISE_SOURCE).binding(DENOISE_SOURCE)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(DENOISE_OUTPUT).binding(DENOISE_OUTPUT)
                    .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1)
                    .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            for (int binding : new int[] {DENOISE_DEPTH, DENOISE_NORMAL, DENOISE_ALBEDO, DENOISE_SPEC_ALBEDO}) {
                bindings.get(binding).binding(binding)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1)
                        .stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }

            LongBuffer handle = stack.mallocLong(1);
            VkDescriptorSetLayoutCreateInfo layoutInfo =
                    VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(bindings);
            check(VK10.vkCreateDescriptorSetLayout(vk, layoutInfo, null, handle),
                    "vkCreateDescriptorSetLayout(rt denoise)");
            long descriptorSetLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                    descriptorSetLayout, "denoise descriptor set layout");

            VkDescriptorPoolSize.Buffer poolSize = VkDescriptorPoolSize.calloc(2, stack);
            poolSize.get(0).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(SET_COUNT);
            poolSize.get(1).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(SET_COUNT * 5);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(SET_COUNT).pPoolSizes(poolSize);
            check(VK10.vkCreateDescriptorPool(vk, poolInfo, null, handle),
                    "vkCreateDescriptorPool(rt denoise)");
            long descriptorPool = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, descriptorPool, "denoise descriptor pool");

            LongBuffer setLayouts = stack.mallocLong(SET_COUNT);
            for (int i = 0; i < SET_COUNT; i++) {
                setLayouts.put(i, descriptorSetLayout);
            }
            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(descriptorPool).pSetLayouts(setLayouts);
            LongBuffer setHandles = stack.mallocLong(SET_COUNT);
            check(VK10.vkAllocateDescriptorSets(vk, allocateInfo, setHandles),
                    "vkAllocateDescriptorSets(rt denoise)");
            long[] descriptorSets = new long[SET_COUNT];
            for (int i = 0; i < SET_COUNT; i++) {
                descriptorSets[i] = setHandles.get(i);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET,
                        descriptorSets[i], "denoise descriptor set " + i);
            }

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0).size(DenoisePushData.BYTE_SIZE);
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(vk, pipelineLayoutInfo, null, handle),
                    "vkCreatePipelineLayout(rt denoise)");
            long pipelineLayout = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipelineLayout, "denoise pipeline layout");

            long module = loadModule(vk, stack);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, "denoise shader module");
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer pipelineInfo = VkComputePipelineCreateInfo.calloc(1, stack);
            pipelineInfo.get(0).sType$Default().stage(stage).layout(pipelineLayout);
            LongBuffer pipelineHandle = stack.mallocLong(1);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, pipelineInfo, null, pipelineHandle),
                    "vkCreateComputePipelines(rt denoise)");
            VK10.vkDestroyShaderModule(vk, module, null);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipelineHandle.get(0), "denoise compute pipeline");

            // Nearest-ish behaviour comes from sampling exactly at texel centres; the linear filter is only
            // used by the resolve's bilinear fetch, where it is what performs the upscale.
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            check(VK10.vkCreateSampler(vk, samplerInfo, null, handle), "vkCreateSampler(rt denoise)");
            long sampler = handle.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "denoise linear sampler");

            return new RtDenoisePipeline(ctx, descriptorSetLayout, descriptorPool, descriptorSets,
                    pipelineLayout, pipelineHandle.get(0), sampler);
        }
    }

    /**
     * Point the four steps at this frame's images. {@code ping} and {@code pong} are the render-res
     * scratch pair the iterations alternate through; {@code out} is the display-res resolve target.
     */
    public void setImages(RtImage noisy, RtImage ping, RtImage pong, RtImage out,
                          RtImage depth, RtImage normal, RtImage diffuseAlbedo, RtImage specularAlbedo) {
        long[] sources = {noisy.view, ping.view, pong.view, ping.view};
        long[] targets = {ping.view, pong.view, ping.view, out.view};
        boolean unchanged = boundDepthView == depth.view && boundNormalView == normal.view
                && boundAlbedoView == diffuseAlbedo.view && boundSpecAlbedoView == specularAlbedo.view;
        for (int i = 0; unchanged && i < SET_COUNT; i++) {
            unchanged = boundSources[i] == sources[i] && boundTargets[i] == targets[i];
        }
        if (unchanged) {
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(SET_COUNT * DENOISE_BINDING_COUNT, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(SET_COUNT * DENOISE_BINDING_COUNT, stack);
            int index = 0;
            for (int set = 0; set < SET_COUNT; set++) {
                index = writeSet(stack, images, writes, index, descriptorSets[set], sources[set], targets[set],
                        depth.view, normal.view, diffuseAlbedo.view, specularAlbedo.view);
                boundSources[set] = sources[set];
                boundTargets[set] = targets[set];
            }
            VkWriteDescriptorSet.Buffer used = VkWriteDescriptorSet.create(writes.address(), index);
            VK10.vkUpdateDescriptorSets(ctx.vk(), used, null);
        }
        boundDepthView = depth.view;
        boundNormalView = normal.view;
        boundAlbedoView = diffuseAlbedo.view;
        boundSpecAlbedoView = specularAlbedo.view;
    }

    private int writeSet(MemoryStack stack, VkDescriptorImageInfo.Buffer images,
                         VkWriteDescriptorSet.Buffer writes, int index, long set,
                         long sourceView, long targetView, long depthView, long normalView,
                         long albedoView, long specAlbedoView) {
        index = writeSampled(stack, images, writes, index, set, DENOISE_SOURCE, sourceView);
        images.get(index).imageView(targetView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        writes.get(index).sType$Default().dstSet(set).dstBinding(DENOISE_OUTPUT)
                .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
        index++;
        index = writeSampled(stack, images, writes, index, set, DENOISE_DEPTH, depthView);
        index = writeSampled(stack, images, writes, index, set, DENOISE_NORMAL, normalView);
        index = writeSampled(stack, images, writes, index, set, DENOISE_ALBEDO, albedoView);
        return writeSampled(stack, images, writes, index, set, DENOISE_SPEC_ALBEDO, specAlbedoView);
    }

    private int writeSampled(MemoryStack stack, VkDescriptorImageInfo.Buffer images,
                             VkWriteDescriptorSet.Buffer writes, int index, long set,
                             int binding, long view) {
        images.get(index).imageView(view).sampler(sampler).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
        writes.get(index).sType$Default().dstSet(set).dstBinding(binding)
                .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                .pImageInfo(VkDescriptorImageInfo.create(images.address(index), 1));
        return index + 1;
    }

    /**
     * Record the a-trous iterations only, leaving the denoised render-res result in the ping-pong pair's
     * {@code ping} image (the target of the last iteration). An upscaler that reprojects and resolves — FSR
     * — consumes that image instead of the built-in resolve. A memory barrier separates every step, each of
     * which reads exactly what the previous one wrote.
     */
    public void recordIterations(VkCommandBuffer cmd, int renderWidth, int renderHeight,
                                 int displayWidth, int displayHeight, float projM22, float projM32) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "scene denoise")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            for (int iteration = 0; iteration < ITERATIONS; iteration++) {
                recordStep(cmd, stack, descriptorSets[iteration], MODE_ATROUS, 1 << iteration,
                        renderWidth, renderHeight, displayWidth, displayHeight, projM22, projM32);
            }
        }
    }

    /**
     * Record only the bilinear resolve. An upscaler inside the pipeline (FSR) needs the denoised
     * render-res image without the built-in resolve, and a backend that produced its own render-res image
     * (NRD) needs the resolve on its own.
     */
    public void recordResolve(VkCommandBuffer cmd, int renderWidth, int renderHeight,
                              int displayWidth, int displayHeight, float projM22, float projM32) {
        try (MemoryStack stack = MemoryStack.stackPush();
             RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "scene resolve")) {
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            recordStep(cmd, stack, descriptorSets[RESOLVE_SET], MODE_RESOLVE, 1,
                    renderWidth, renderHeight, displayWidth, displayHeight, projM22, projM32);
        }
    }

    /**
     * Record the a-trous iterations followed by the bilinear resolve. {@code projM22}/{@code projM32} are
     * this frame's projection terms, used to turn the reversed-Z depth guide into view distances.
     */
    public void record(VkCommandBuffer cmd, int renderWidth, int renderHeight,
                       int displayWidth, int displayHeight, float projM22, float projM32) {
        recordIterations(cmd, renderWidth, renderHeight, displayWidth, displayHeight, projM22, projM32);
        recordResolve(cmd, renderWidth, renderHeight, displayWidth, displayHeight, projM22, projM32);
    }

    private void recordStep(VkCommandBuffer cmd, MemoryStack stack, long set, int mode, int step,
                            int renderWidth, int renderHeight, int displayWidth, int displayHeight,
                            float projM22, float projM32) {
        VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelineLayout, 0, stack.longs(set), null);
        ByteBuffer push = stack.malloc(DenoisePushData.BYTE_SIZE);
        new DenoisePushData(mode, step, renderWidth, renderHeight, displayWidth, displayHeight,
                projM22, projM32, DEPTH_SIGMA, NORMAL_POWER, ALBEDO_SIGMA, LUMA_SIGMA,
                MIRROR_ROUGHNESS, FIREFLY_CLAMP).write(push);
        VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
        int width = mode == MODE_RESOLVE ? displayWidth : renderWidth;
        int height = mode == MODE_RESOLVE ? displayHeight : renderHeight;
        VK10.vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, pipeline, null);
        VK10.vkDestroySampler(vk, sampler, null);
        VK10.vkDestroyPipelineLayout(vk, pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, descriptorSetLayout, null);
        destroyed = true;
    }

    private static long loadModule(VkDevice vk, MemoryStack stack) {
        byte[] bytes;
        try (InputStream input = RtDenoisePipeline.class.getResourceAsStream(SHADER)) {
            if (input == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER);
            }
            bytes = input.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, moduleInfo, null, module), "vkCreateShaderModule(denoise)");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
