# SPDX-FileCopyrightText: 2025 Mobile Robotics Lab, Technical University of Munich
# SPDX-FileCopyrightText: 2025 Jiaxin Wei
# SPDX-License-Identifier: Apache-2.0


from io import BytesIO
import random
from typing import List
from omegaconf import OmegaConf
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from marigold.marigold_gsfixer_pipeline import MarigoldGSFixerPipeline
from src.trainer.base_trainer import BaseTrainer


class MarigoldGSFixerTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldGSFixerPipeline,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
        dual_input: bool = False,
    ):
        super().__init__(cfg,
                         model,
                         train_dataloader,
                         device,
                         out_dir_ckpt,
                         out_dir_eval,
                         out_dir_vis,
                         accumulation_steps,
                         val_dataloaders,
                         vis_dataloaders,
                         dual_input,
                        )
        # ===== ABLATION[A2] Random composite corruption; false uses legacy corruption. =====
        self.corruption_aug_cfg = getattr(cfg, "corruption_aug", None)
        self.corruption_aug_enabled = bool(
            getattr(self.corruption_aug_cfg, "enabled", False)
        ) if self.corruption_aug_cfg is not None else False

    def _legacy_corrupt(self, image: torch.Tensor) -> torch.Tensor:
        image = self.corrupt_transforms(image)
        image = image + torch.randn_like(image) * 0.2
        return torch.clamp(image, -1.0, 1.0)

    @staticmethod
    def _uniform_range(value, default):
        if value is None:
            value = default
        value = list(value)
        if len(value) != 2:
            raise ValueError(f"Expected a [min, max] range, got {value}")
        return random.uniform(float(value[0]), float(value[1]))

    def _apply_jpeg(self, image: torch.Tensor, quality: int) -> torch.Tensor:
        image_uint8 = ((image.detach().cpu() + 1.0) * 127.5).clamp(0, 255).byte()
        encoded = []
        for sample in image_uint8:
            pil = TF.to_pil_image(sample)
            buffer = BytesIO()
            pil.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            decoded = Image.open(buffer).convert("RGB")
            encoded.append(TF.pil_to_tensor(decoded))
        jpeg = torch.stack(encoded, dim=0).to(image.device, dtype=image.dtype)
        return jpeg / 127.5 - 1.0

    def _apply_color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        cfg = getattr(self.corruption_aug_cfg, "color_jitter", None)
        brightness = self._uniform_range(getattr(cfg, "brightness", [0.9, 1.1]), [0.9, 1.1])
        contrast = self._uniform_range(getattr(cfg, "contrast", [0.9, 1.1]), [0.9, 1.1])
        saturation = self._uniform_range(getattr(cfg, "saturation", [0.9, 1.1]), [0.9, 1.1])

        image = image * brightness
        mean = image.mean(dim=(2, 3), keepdim=True)
        image = (image - mean) * contrast + mean
        gray = (
            image[:, 0:1] * 0.2989
            + image[:, 1:2] * 0.5870
            + image[:, 2:3] * 0.1140
        )
        image = (image - gray) * saturation + gray
        return torch.clamp(image, -1.0, 1.0)

    def _apply_random_corruption(self, image: torch.Tensor, stream_name: str) -> torch.Tensor:
        cfg = self.corruption_aug_cfg

        blur_cfg = getattr(cfg, "blur", None)
        if bool(getattr(blur_cfg, "enabled", True)):
            kernel_choices = list(getattr(blur_cfg, "kernel_choices", [3, 5, 7, 9, 11, 13]))
            k = int(random.choice(kernel_choices))
            if k % 2 == 0:
                k += 1
            sigma = self._uniform_range(getattr(blur_cfg, "sigma", [0.1, 2.0]), [0.1, 2.0])
            image = TF.gaussian_blur(image, kernel_size=[k, k], sigma=[sigma, sigma])

        noise_cfg = getattr(cfg, "gaussian_noise", None)
        if bool(getattr(noise_cfg, "enabled", True)):
            std = self._uniform_range(getattr(noise_cfg, "std", [0.0, 0.3]), [0.0, 0.3])
            image = image + torch.randn_like(image) * std

        jpeg_cfg = getattr(cfg, "jpeg", None)
        if bool(getattr(jpeg_cfg, "enabled", True)) and random.random() < float(getattr(jpeg_cfg, "prob", 0.25)):
            quality = int(self._uniform_range(getattr(jpeg_cfg, "quality", [30, 90]), [30, 90]))
            image = self._apply_jpeg(image, quality)

        resize_cfg = getattr(cfg, "downsample", None)
        if bool(getattr(resize_cfg, "enabled", True)) and random.random() < float(getattr(resize_cfg, "prob", 0.25)):
            scale = self._uniform_range(getattr(resize_cfg, "scale", [0.5, 0.9]), [0.5, 0.9])
            h, w = image.shape[-2:]
            small_h = max(8, int(h * scale))
            small_w = max(8, int(w * scale))
            image = F.interpolate(image, size=(small_h, small_w), mode="bilinear", align_corners=False)
            image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)

        sp_cfg = getattr(cfg, "salt_pepper", None)
        if bool(getattr(sp_cfg, "enabled", True)) and random.random() < float(getattr(sp_cfg, "prob", 0.2)):
            amount = self._uniform_range(getattr(sp_cfg, "amount", [0.0005, 0.01]), [0.0005, 0.01])
            salt_mask = torch.rand_like(image[:, :1]) < amount * 0.5
            pepper_mask = torch.rand_like(image[:, :1]) < amount * 0.5
            image = torch.where(salt_mask.expand_as(image), torch.ones_like(image), image)
            image = torch.where(pepper_mask.expand_as(image), -torch.ones_like(image), image)

        color_cfg = getattr(cfg, "color_jitter", None)
        if bool(getattr(color_cfg, "enabled", True)) and random.random() < float(getattr(color_cfg, "prob", 0.3)):
            image = self._apply_color_jitter(image)

        return torch.clamp(image, -1.0, 1.0)

    def _load_train_data(self, batch):
        """Load data from batch for training from scratch."""
        rgb = batch["rgb_norm"].to(self.device)
        corrupted_rgb1 = batch["gs_norm"].to(self.device)
        # ===== ABLATION[A2] Composite degradation vs fixed blur + Gaussian noise. =====
        if self.corruption_aug_enabled:
            corrupted_rgb1 = self._apply_random_corruption(corrupted_rgb1, "gs")
        else:
            corrupted_rgb1 = self._legacy_corrupt(corrupted_rgb1)
        if self.dual_input:
            corrupted_rgb2 = batch["mesh_norm"].to(self.device)
            if self.corruption_aug_enabled:
                corrupted_rgb2 = self._apply_random_corruption(corrupted_rgb2, "mesh")
            else:
                corrupted_rgb2 = self._legacy_corrupt(corrupted_rgb2)
        else:
            corrupted_rgb2 = None

        return rgb, corrupted_rgb1, corrupted_rgb2

    def _load_val_data(self, batch):
        """Load validation/visualization data from batch."""
        rgb = batch["rgb_int"] / 255.0
        corrupted_rgb1 = batch["gs_int"] / 255.0
        corrupted_rgb1 = self.corrupt_transforms(corrupted_rgb1)
        corrupted_rgb1 += torch.randn_like(corrupted_rgb1) * 0.2
        corrupted_rgb1 = torch.clamp(corrupted_rgb1, 0.0, 1.0) * 255.0
        if self.dual_input:
            corrupted_rgb2 = batch["mesh_int"] / 255.0
            corrupted_rgb2 = self.corrupt_transforms(corrupted_rgb2)
            corrupted_rgb2 += torch.randn_like(corrupted_rgb2) * 0.2
            corrupted_rgb2 = torch.clamp(corrupted_rgb2, 0.0, 1.0) * 255.0
        else:
            corrupted_rgb2 = None

        return rgb, corrupted_rgb1, corrupted_rgb2
