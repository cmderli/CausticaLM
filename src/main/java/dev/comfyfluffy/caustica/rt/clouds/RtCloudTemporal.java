package dev.comfyfluffy.caustica.rt.clouds;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
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
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * Sky/cloud temporal accumulation. Reads the trace color, hardware depth and motion guides, and a
 * history image; writes a filtered color image and next frame's history. Geometry pixels copy through,
 * sky pixels reproject and blend. The history is ping-ponged by the caller.
 */
public final class RtCloudTemporal {
    private static final String SHADER_RESOURCE = "/caustica/shaders/pipelines/world/cloud_temporal.comp.spv";
    private static final int PUSH_BYTES = 48;
    private static final int GROUP_SIZE = 16;
    private static final int CURRENT = 0;
    private static final int DEPTH = 1;
    private static final int MOTION = 2;
    private static final int HISTORY = 3;
    private static final int OUTPUT = 4;
    private static final int HISTORY_OUTPUT = 5;
    private static final int BINDING_COUNT = 6;

    private final RtContext ctx;
    private final long sampler;
    private final long descriptorSetLayout;
    private final long descriptorPool;
    private final long[] descriptorSets;
    private final long pipelineLayout;
    private final long pipeline;

    private RtCloudTemporal(RtContext ctx, long sampler, long dsl, long pool, long[] sets, long layout,
                            long pipeline) {
        this.ctx = ctx;
        this.sampler = sampler;
        this.descriptorSetLayout = dsl;
        this.descriptorPool = pool;
        this.descriptorSets = sets;
        this.pipelineLayout = layout;
        this.pipeline = pipeline;
    }

    public static RtCloudTemporal create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR).minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f).maxLod(0.0f);
            LongBuffer handle = stack.mallocLong(1);
            check(VK10.vkCreateSampler(ctx.vk(), samplerInfo, null, handle),
                    "vkCreateSampler(cloud temporal)");
            long sampler = handle.get(0);

            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(BINDING_COUNT, stack);
            for (int i = CURRENT; i <= HISTORY; i++) {
                bindings.get(i).binding(i).descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            }
            bindings.get(OUTPUT).binding(OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            bindings.get(HISTORY_OUTPUT).binding(HISTORY_OUTPUT).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
            VkDescriptorSetLayoutCreateInfo dslInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings);
            check(VK10.vkCreateDescriptorSetLayout(ctx.vk(), dslInfo, null, handle),
                    "vkCreateDescriptorSetLayout(cloud temporal)");
            long dsl = handle.get(0);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(8);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(4);
            VkDescriptorPoolCreateInfo poolInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(2).pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(ctx.vk(), poolInfo, null, handle),
                    "vkCreateDescriptorPool(cloud temporal)");
            long pool = handle.get(0);

            VkDescriptorSetAllocateInfo allocateInfo = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(pool).pSetLayouts(stack.longs(dsl, dsl));
            LongBuffer setHandles = stack.mallocLong(2);
            check(VK10.vkAllocateDescriptorSets(ctx.vk(), allocateInfo, setHandles),
                    "vkAllocateDescriptorSets(cloud temporal)");
            long[] sets = {setHandles.get(0), setHandles.get(1)};

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
            pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(PUSH_BYTES);
            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(dsl)).pPushConstantRanges(pushRange);
            check(VK10.vkCreatePipelineLayout(ctx.vk(), layoutInfo, null, handle),
                    "vkCreatePipelineLayout(cloud temporal)");
            long layout = handle.get(0);

            long module = loadModule(ctx, stack);
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
            info.get(0).sType$Default().stage(stage).layout(layout);
            check(VK10.vkCreateComputePipelines(ctx.vk(), VK10.VK_NULL_HANDLE, info, null, handle),
                    "vkCreateComputePipelines(cloud temporal)");
            long pipeline = handle.get(0);
            VK10.vkDestroyShaderModule(ctx.vk(), module, null);

            RtCloudTemporal result = new RtCloudTemporal(ctx, sampler, dsl, pool, sets, layout, pipeline);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline, "cloud temporal pipeline");
            return result;
        }
    }

    /**
     * Writes the two history-parity descriptor sets once per image set. Set 0 reads history A and writes
     * B; set 1 reads B and writes A. Keeping both sets immutable after this call avoids rewriting a
     * descriptor set that a previous frame may still be executing.
     */
    public void configure(long currentView, long depthView, long motionView, long historyAView,
                          long historyBView, long outputView) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorImageInfo.Buffer images = VkDescriptorImageInfo.calloc(BINDING_COUNT, stack);
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(BINDING_COUNT * 2, stack);
            int write = 0;
            for (int parity = 0; parity < 2; parity++) {
                images.clear();
                long readHistory = parity == 0 ? historyAView : historyBView;
                long writeHistory = parity == 0 ? historyBView : historyAView;
                for (int i = CURRENT; i <= HISTORY; i++) {
                    long view = switch (i) {
                        case CURRENT -> currentView;
                        case DEPTH -> depthView;
                        case MOTION -> motionView;
                        default -> readHistory;
                    };
                    images.get(i).imageView(view).sampler(sampler)
                            .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                    writes.get(write++).sType$Default().dstSet(descriptorSets[parity]).dstBinding(i)
                            .descriptorCount(1)
                            .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                            .pImageInfo(VkDescriptorImageInfo.create(images.address(i), 1));
                }
                images.get(OUTPUT).imageView(outputView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(write++).sType$Default().dstSet(descriptorSets[parity]).dstBinding(OUTPUT)
                        .descriptorCount(1).descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(OUTPUT), 1));
                images.get(HISTORY_OUTPUT).imageView(writeHistory)
                        .imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);
                writes.get(write++).sType$Default().dstSet(descriptorSets[parity])
                        .dstBinding(HISTORY_OUTPUT).descriptorCount(1)
                        .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.create(images.address(HISTORY_OUTPUT), 1));
            }
            VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);
        }
    }

    public void dispatch(VkCommandBuffer cmd, MemoryStack stack, int width, int height,
                         boolean historyValid, int historyIndex, long worldPushAddr, long noiseAddr,
                         float windDeltaX, float windDeltaZ) {
        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                stack.longs(descriptorSets[historyIndex]), null);
        ByteBuffer push = stack.malloc(PUSH_BYTES).order(java.nio.ByteOrder.nativeOrder());
        push.putFloat(width);
        push.putFloat(height);
        push.putInt(historyValid ? 1 : 0);
        push.putInt(0);
        push.putLong(worldPushAddr);
        push.putLong(noiseAddr);
        push.putFloat(windDeltaX);
        push.putFloat(windDeltaZ);
        push.putFloat(0.0f);
        push.putFloat(0.0f);
        push.flip();
        VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
        VK10.vkCmdDispatch(cmd, (width + GROUP_SIZE - 1) / GROUP_SIZE,
                (height + GROUP_SIZE - 1) / GROUP_SIZE, 1);
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    public void destroy() {
        VK10.vkDestroyPipeline(ctx.vk(), pipeline, null);
        VK10.vkDestroyPipelineLayout(ctx.vk(), pipelineLayout, null);
        VK10.vkDestroyDescriptorPool(ctx.vk(), descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(ctx.vk(), descriptorSetLayout, null);
        VK10.vkDestroySampler(ctx.vk(), sampler, null);
    }

    private static long loadModule(RtContext ctx, MemoryStack stack) {
        byte[] bytes;
        try (InputStream in = RtCloudTemporal.class.getResourceAsStream(SHADER_RESOURCE)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_RESOURCE);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_RESOURCE, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType$Default().pCode(code);
            LongBuffer module = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(ctx.vk(), moduleInfo, null, module),
                    "vkCreateShaderModule(" + SHADER_RESOURCE + ")");
            return module.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
