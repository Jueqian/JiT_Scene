# src/models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from omegaconf import DictConfig # 用于 Hydra 配置

# 假设 PointCloudDecoder 是您自定义的模块
# ⚠️ 您需要自行实现 VFEModule 和 PointCloudDecoder
# from .vfe_module import VFEModule 
# from .pointcloud_decoder import PointCloudDecoder 


from PIL import Image
import torch

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载预训练模型和调度器
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)  # type: ignore



# 设置采样步数
scheduler.set_timesteps(50)

# 准备随机噪声输入
sample_size = model.config.sample_size  # 256   # type: ignore
noise = torch.randn((1, 3, sample_size, sample_size), device=device)
input_tensor = noise

# DDPM 采样循环
for t in scheduler.timesteps:
    with torch.no_grad():
        # UNet 输出
        noisy_residual = model(input_tensor, t).sample
        # Scheduler 更新
        prev_noisy_sample = scheduler.step(noisy_residual, t, input_tensor).prev_sample     # type: ignore
        input_tensor = prev_noisy_sample

# 后处理为图片
image = (input_tensor / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]  # NHWC
image = Image.fromarray((image * 255).round().astype("uint8"))

# 保存图片
image.save("ddpm_sample.png")
print("Sample saved as ddpm_sample.png")










# class VAEPointcloudCompressor(nn.Module):
#     """
#     基于 VAE 的 LiDAR 点云压缩网络。
#     集成了 Voxel Feature Extraction (VFE), 2D U-Net Encoder (使用 diffusers), 
#     VAE 编码/解码 和 Point Cloud Decoder。
#     """
#     def __init__(self, cfg: DictConfig):
#         super().__init__()
        
#         # 1. Voxel Feature Extraction (VFE) and BEV Projection
#         # ⚠️ 占位符：需要实现 VFE 模块，它将点云转换为 BEV 特征图。
#         # 假设 cfg.vfe_params 包含了 VFE 模块所需的参数
#         # self.vfe = VFEModule(cfg.vfe_params)
#         # 暂时使用一个简单的占位符，模拟 VFE 输出 BEV 图的通道数
#         self.vfe_output_channels = cfg.model.vfe.output_channels 

#         # 2. 2D U-Net Encoder (使用 diffusers 库)
#         # 我们使用 UNet2DModel 作为主干，并将其视为一个强大的特征提取器。
#         # 我们将从 UNet 的中间层或最后一层卷积块的输出中提取特征。
        
#         unet_config = cfg.model.unet
#         self.unet_encoder = UNet2DModel(
#             sample_size=unet_config.sample_size, # e.g., 64 (BEV map resolution)
#             in_channels=self.vfe_output_channels,
#             out_channels=unet_config.latent_feature_channels, # U-Net 编码器最终输出的特征通道数
#             # 传递其他 diffusers 参数
#             layers_per_block=unet_config.layers_per_block,
#             block_out_channels=unet_config.block_out_channels,
#             down_block_types=unet_config.down_block_types,
#             up_block_types=[], # 仅使用 Encoder 部分，但 diffusers 库的 UNet2DModel 默认包含 Decoder
#                                # 实际使用时，需要自定义提取 Encoder 输出的逻辑
#         )
#         # 实际操作中，为了避免使用 UNet2DModel 附带的 Decoder 部分，
#         # 您可能需要使用 diffusers 库中的更底层组件，例如 DownBlock2D。
        
#         # 3. VAE Latent Space 编码层 (将 U-Net 输出映射到 μ 和 log_sigma^2)
#         # 假设 U-Net 编码器最终输出一个形状为 (B, C_feat, H/8, W/8) 的特征图。
#         self.latent_feature_size = unet_config.latent_feature_channels * \
#                                    (unet_config.sample_size // (2**len(unet_config.block_out_channels)))**2
        
#         self.fc_mu = nn.Linear(self.latent_feature_size, cfg.model.vae.latent_dim)
#         self.fc_logvar = nn.Linear(self.latent_feature_size, cfg.model.vae.latent_dim)
        
#         self.latent_dim = cfg.model.vae.latent_dim
        
#         # 4. VAE Decoder Head (将 Z 映射回特征图)
#         self.fc_decoder = nn.Linear(self.latent_dim, self.latent_feature_size)
        
#         # 5. Point Cloud Decoder
#         # ⚠️ 占位符：需要实现将特征图重建回点云的模块。
#         # self.pointcloud_decoder = PointCloudDecoder(cfg.decoder_params)


#     def encode(self, x):
#         """
#         编码阶段: VFE -> U-Net Encoder -> Latent Params
#         x: LiDAR 原始点云数据 (B, N, D)
#         """
#         # 1. VFE 和 BEV 投影
#         # bev_map = self.vfe(x) 
#         # 暂时使用一个简单的占位符
#         bev_map = torch.randn(x.size(0), self.vfe_output_channels, 64, 64).to(x.device) 
        
#         # 2. U-Net Encoder (特征提取)
#         # ⚠️ 关键步骤：如何从 UNet2DModel 中提取编码器的最后一层特征。
#         # 实际中，您可能需要修改 UNet2DModel 的 forward 或使用其更底层的 DownBlock2D。
        
#         # 简化的占位符: 假设 unet_encoder 的输出是编码特征
#         unet_output = self.unet_encoder(bev_map, return_dict=True)
#         # 为了演示，我们假设提取到最后的特征图并将其展平
#         encoded_features = unet_output.sample # ⚠️ 这是一个不准确的简化！
#                                               # 在 VAE 语境下，应该提取编码器块的输出。
        
#         # 展平特征图
#         flattened_features = encoded_features.view(encoded_features.size(0), -1)
        
#         # 3. Latent Space 参数
#         mu = self.fc_mu(flattened_features)
#         logvar = self.fc_logvar(flattened_features)
        
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         """重参数化技巧"""
#         if self.training:
#             std = torch.exp(0.5 * logvar)
#             eps = torch.randn_like(std)
#             return mu + eps * std
#         else:
#             return mu

#     def decode(self, z, original_shape):
#         """
#         解码阶段: Latent Z -> Feature Map -> Point Cloud
#         z: 潜在向量 (B, latent_dim)
#         original_shape: 用于重塑特征图的形状信息
#         """
#         # 1. 潜在向量映射回特征图空间
#         feature_map_flat = self.fc_decoder(z)
#         # 重塑回 BEV 特征图的形状 (B, C_feat, H, W)
#         C, H, W = original_shape
#         feature_map = feature_map_flat.view(z.size(0), C, H, W)

#         # 2. Point Cloud Decoder (从 BEV 特征图重建点云)
#         # reconstructed_pc = self.pointcloud_decoder(feature_map) 
#         # 暂时使用一个简单的占位符
#         reconstructed_pc = torch.randn(z.size(0), 1024, 3).to(z.device)

#         return reconstructed_pc

#     def forward(self, x):
#         """
#         完整的 VAE 前向传播
#         x: LiDAR 原始点云数据
#         """
#         # 1. 编码
#         mu, logvar = self.encode(x)
        
#         # 2. 重参数化
#         z = self.reparameterize(mu, logvar)
        
#         # 3. 解码
#         # ⚠️ 需要根据您的 U-Net 编码器输出尺寸来计算解码器所需的特征图形状
#         # 暂时使用占位符形状 (32, 8, 8)
#         feature_map_shape = (
#             self.unet_encoder.config.out_channels, 
#             self.unet_encoder.config.sample_size // (2**len(self.unet_encoder.config.block_out_channels)),
#             self.unet_encoder.config.sample_size // (2**len(self.unet_encoder.config.block_out_channels))
#         )
#         reconstructed_pc = self.decode(z, feature_map_shape)

#         return reconstructed_pc, mu, logvar
    
    