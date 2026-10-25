package dev.comfyfluffy.caustica.rt.clouds;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * The cloud shadow-map bake. One compute dispatch fills a 256 x 256 tile-space transmittance map from
 * the baked cloud-body SDF; terrain NEE reads the map instead of marching the density again.
 *
 * <p>The map is a BDA buffer, so the pass needs no descriptor set: the shader receives the frame's
 * WorldPush slot, the noise buffer and the map address in a small push block. The map is read in the
 * same submission after a barrier.
 */
public final class RtCloudShadow {
    public static final int MAP_SIZE = 256;
    private static final int PUSH_BYTES = 48;
    private static final int GROUP_SIZE = 8;
    private static final String SHADER_RESOURCE = "/caustica/shaders/pipelines/world/cloud_shadow.comp.spv";

    private final RtContext ctx;
    private final RtBuffer map;
    private final long pipelineLayout;
    private final long pipeline;

    private RtCloudShadow(RtContext ctx, RtBuffer map, long pipelineLayout, long pipeline) {
        this.ctx = ctx;
        this.map = map;
        this.pipelineLayout = pipelineLayout;
        this.pipeline = pipeline;
    }

    public static RtCloudShadow create(RtContext ctx) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            RtBuffer map = ctx.createBuffer((long) MAP_SIZE * MAP_SIZE * 4L,
                    VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "cloud shadow map");
            try {
                VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
                pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                        .offset(0).size(PUSH_BYTES);
                VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                        .sType$Default().pPushConstantRanges(pushRange);
                LongBuffer handle = stack.mallocLong(1);
                check(VK10.vkCreatePipelineLayout(ctx.vk(), layoutInfo, null, handle),
                        "vkCreatePipelineLayout(cloud shadow)");
                long pipelineLayout = handle.get(0);
                RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipelineLayout,
                        "cloud shadow pipeline layout");
                long pipeline = createPipeline(ctx, stack, pipelineLayout);
                return new RtCloudShadow(ctx, map, pipelineLayout, pipeline);
            } catch (Throwable t) {
                map.destroy();
                throw t;
            }
        }
    }

    public long address() {
        return map.deviceAddress;
    }

    /** Records the shadow bake. The caller must barrier before the trace reads the map. */
    public void record(VkCommandBuffer cmd, MemoryStack stack, long worldPushAddr, long noiseAddr,
                       float thickness, float coverage, float densityScale, float mapTile) {
        ByteBuffer push = stack.malloc(PUSH_BYTES).order(ByteOrder.nativeOrder());
        push.putLong(worldPushAddr);
        push.putLong(noiseAddr);
        push.putLong(map.deviceAddress);
        push.putFloat(thickness);
        push.putFloat(coverage);
        push.putFloat(densityScale);
        push.putFloat(mapTile);
        push.putFloat(0.0f);
        push.putFloat(0.0f);
        push.flip();

        VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        VK10.vkCmdPushConstants(cmd, pipelineLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
        VK10.vkCmdDispatch(cmd, (MAP_SIZE + GROUP_SIZE - 1) / GROUP_SIZE,
                (MAP_SIZE + GROUP_SIZE - 1) / GROUP_SIZE, 1);
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
    }

    public void destroy() {
        VK10.vkDestroyPipeline(ctx.vk(), pipeline, null);
        VK10.vkDestroyPipelineLayout(ctx.vk(), pipelineLayout, null);
        map.destroy();
    }

    private static long createPipeline(RtContext ctx, MemoryStack stack, long layout) {
        long module = loadModule(ctx, stack);
        VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                .sType$Default().stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT)
                .module(module).pName(stack.UTF8("main"));
        VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
        info.get(0).sType$Default().stage(stage).layout(layout);
        LongBuffer handle = stack.mallocLong(1);
        check(VK10.vkCreateComputePipelines(ctx.vk(), VK10.VK_NULL_HANDLE, info, null, handle),
                "vkCreateComputePipelines(cloud shadow)");
        VK10.vkDestroyShaderModule(ctx.vk(), module, null);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, handle.get(0), "cloud shadow pipeline");
        return handle.get(0);
    }

    private static long loadModule(RtContext ctx, MemoryStack stack) {
        byte[] bytes;
        try (InputStream input = RtCloudShadow.class.getResourceAsStream(SHADER_RESOURCE)) {
            if (input == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_RESOURCE);
            }
            bytes = input.readAllBytes();
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
