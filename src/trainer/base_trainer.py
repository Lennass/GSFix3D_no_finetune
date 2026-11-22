# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------
# This file has been modified by Mobile Robotics Lab, TU Munich (2025).
# SPDX-FileCopyrightText: 2025 Jiaxin Wei
# SPDX-License-Identifier: Apache-2.0


from abc import abstractmethod
import copy
import logging
import os
import shutil
import torch
import torch.nn.functional as F
from datetime import datetime
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur, Compose
from tqdm import tqdm
from typing import List, Union

from marigold.marigold_gsfixer_pipeline import MarigoldGSFixerPipeline, MarigoldGSFixerOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence


class BaseTrainer:
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
        self.cfg: OmegaConf = cfg
        self.model: MarigoldGSFixerPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps
        self.dual_input = dual_input

        # Adapt input layers when training from scratch
        if self.dual_input:
            self.input_channels = 12  # mesh + gs
        else:
            self.input_channels = 8  # only gs
        # Track whether conv_in was rebuilt (channel expansion). When the base
        # already has the right in_channels (e.g. finetuning gsfixer-full), conv_in
        # is the pretrained one and must NOT be unfrozen under LoRA.
        self.conv_in_replaced = self.input_channels != self.model.unet.config["in_channels"]
        if self.conv_in_replaced:
            self._replace_unet_conv_in()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        # Attention backend. xformers' prebuilt flash-attention kernels are
        # Hopper-only and crash on Blackwell (RTX 5090, sm_120) with a CUDA
        # "invalid argument" from flash_fwd_launch_template.h — especially under
        # bf16, which routes attention through that flash path. Default to
        # PyTorch SDPA (AttnProcessor2_0), which supports Blackwell natively and
        # is also memory-efficient. Set speedup.use_xformers: true to opt back in.
        _speedup_cfg = getattr(self.cfg, "speedup", None)
        _use_xformers = bool(getattr(_speedup_cfg, "use_xformers", False)) if _speedup_cfg is not None else False
        if _use_xformers:
            try:
                self.model.unet.enable_xformers_memory_efficient_attention()
                logging.info("xformers memory-efficient attention enabled")
            except Exception as e:
                logging.warning(f"xformers enable failed, falling back to SDPA: {e}")
                _use_xformers = False
        if not _use_xformers:
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.model.unet.set_attn_processor(AttnProcessor2_0())
                logging.info("Using PyTorch SDPA attention (AttnProcessor2_0)")
            except Exception as e:
                logging.warning(f"Could not set SDPA attention processor: {e}")

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)

        # ===== ABLATION[A5] Expanded-layer LoRA; false means full-UNet training. =====
        lora_cfg = getattr(self.cfg, "lora", None)
        self.lora_enabled = bool(getattr(lora_cfg, "enabled", False)) if lora_cfg is not None else False
        self.lora_save_merged = bool(getattr(lora_cfg, "save_merged", True)) if lora_cfg is not None else True
        self._lora_peft_config = None
        self._lora_load_path = None
        if self.lora_enabled:
            self._setup_lora(lora_cfg)

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        trainable_params = [p for p in self.model.unet.parameters() if p.requires_grad]
        opt_cfg = self.cfg.optimizer
        opt_name = getattr(opt_cfg, "name", "Adam")
        # ===== ABLATION[A7] AdamW; false restores the original Adam optimizer. =====
        adamw_enabled = bool(getattr(opt_cfg, "adamw_enabled", opt_name == "AdamW"))
        if adamw_enabled:
            kwargs_cfg = getattr(opt_cfg, "adamw_kwargs", getattr(opt_cfg, "kwargs", {}))
            _opt_cls = AdamW
        else:
            kwargs_cfg = getattr(opt_cfg, "adam_kwargs", {})
            _opt_cls = Adam
        if OmegaConf.is_config(kwargs_cfg):
            opt_kwargs = OmegaConf.to_container(kwargs_cfg, resolve=True)
        else:
            opt_kwargs = dict(kwargs_cfg)
        self.optimizer = _opt_cls(trainable_params, lr=lr, **opt_kwargs)
        logging.info(f"Optimizer: {_opt_cls.__name__}({opt_kwargs})")
        if self.lora_enabled:
            n_trainable = sum(p.numel() for p in trainable_params)
            n_total = sum(p.numel() for p in self.model.unet.parameters())
            logging.info(
                f"LoRA enabled: trainable params {n_trainable:,} / {n_total:,} "
                f"({100.0 * n_trainable / max(n_total, 1):.2f}%)"
            )

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)
        # ===== ABLATION[A4] Decoded-x0 pixel-space auxiliary loss. =====
        self.pixel_loss_enabled = bool(getattr(self.cfg, "pixel_loss", {}).get("enabled", False))
        self.pixel_loss_weight = float(getattr(self.cfg, "pixel_loss", {}).get("weight", 0.0))
        self.pixel_loss_max_samples = int(getattr(self.cfg, "pixel_loss", {}).get("max_samples", 1))
        self.pixel_loss_latent_downsample = int(getattr(self.cfg, "pixel_loss", {}).get("latent_downsample", 1))
        self.pixel_loss_fn = None
        if self.pixel_loss_enabled:
            pixel_loss_cfg = self.cfg.pixel_loss
            self.pixel_loss_fn = get_loss(
                loss_name=pixel_loss_cfg.name,
                **pixel_loss_cfg.kwargs,
            )

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )

        logging.info(
            "DDPM training noise scheduler config is updated: "
            f"rescale_betas_zero_snr = {self.training_noise_scheduler.config.rescale_betas_zero_snr}, "
            f"timestep_spacing = {self.training_noise_scheduler.config.timestep_spacing}"
        )

        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Inference DDIM scheduler (used for validation)
        self.model.scheduler = DDIMScheduler.from_config(
            self.training_noise_scheduler.config,
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        train_metric_keys = ["loss", "latent_loss"]
        if self.pixel_loss_enabled:
            train_metric_keys.append("pixel_loss")
        self.train_metrics = MetricTracker(*train_metric_keys)
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])

        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal

        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."

        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

        self.corrupt_transforms = Compose([
            GaussianBlur(kernel_size=13),
        ])

        # Speedup options
        speedup_cfg = getattr(self.cfg, "speedup", None)
        self.use_bf16 = bool(getattr(speedup_cfg, "bf16", False)) if speedup_cfg is not None else False
        self.use_compile = bool(getattr(speedup_cfg, "torch_compile", False)) if speedup_cfg is not None else False
        self.timestep_max = int(getattr(speedup_cfg, "timestep_max", self.scheduler_timesteps)) if speedup_cfg is not None else self.scheduler_timesteps
        self.timestep_max = max(1, min(self.timestep_max, self.scheduler_timesteps))

        if self.use_compile:
            try:
                self.model.unet = torch.compile(
                    self.model.unet, mode="reduce-overhead", fullgraph=False, dynamic=True
                )
                logging.info("torch.compile enabled on UNet (reduce-overhead)")
            except Exception as e:
                self.use_compile = False
                logging.warning(f"torch.compile failed, fallback to eager: {e}")

        logging.info(
            f"Speedup: bf16={self.use_bf16}, torch_compile={self.use_compile}, "
            f"timestep_max={self.timestep_max}/{self.scheduler_timesteps}"
        )

    def _reattach_lora_after_merge(self):
        """DEPRECATED. No longer used: save_checkpoint now merges on a deep copy,
        so the live UNet keeps its LoRA wrappers and the optimizer is never reset.
        Kept for backward compatibility with external callers."""
        try:
            from peft import get_peft_model
        except ImportError:
            return

        if self._lora_peft_config is None:
            return

        new_unet = get_peft_model(self.model.unet, self._lora_peft_config)
        self.model.unet = new_unet

        # Re-mark conv_in as trainable only if it was rebuilt for channel expansion.
        # (peft wrapping freezes the base model again; keep finetune conv_in frozen.)
        if getattr(self, "conv_in_replaced", False):
            for p in self.model.unet.conv_in.parameters():
                p.requires_grad = True

        # Replace optimizer param refs with the new (zero-init) LoRA params + conv_in
        new_trainable = [p for p in self.model.unet.parameters() if p.requires_grad]
        for group in self.optimizer.param_groups:
            group["params"] = new_trainable
        self.optimizer.state.clear()

    def _setup_lora(self, lora_cfg):
        try:
            from peft import LoraConfig, get_peft_model, PeftModel
        except ImportError as e:
            raise ImportError(
                "LoRA is enabled but `peft` is not installed. Please `pip install peft`."
            ) from e

        rank = int(getattr(lora_cfg, "rank", 8))
        alpha = float(getattr(lora_cfg, "alpha", 16))
        dropout = float(getattr(lora_cfg, "dropout", 0.0))
        target_modules = list(getattr(
            lora_cfg,
            "target_modules",
            ["to_q", "to_k", "to_v", "to_out.0"],
        ))
        logging.info(f"LoRA targets = {target_modules}")
        load_path = getattr(lora_cfg, "load_path", None)

        # Freeze all UNet weights first; LoRA wrappers will mark adapters trainable.
        self.model.unet.requires_grad_(False)

        if load_path:
            logging.info(f"Loading existing LoRA adapter from: {load_path}")
            self.model.unet = PeftModel.from_pretrained(
                self.model.unet, load_path, is_trainable=True
            )
            self._lora_load_path = load_path
            self._lora_peft_config = self.model.unet.peft_config[
                next(iter(self.model.unet.peft_config))
            ]
        else:
            peft_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                target_modules=target_modules,
            )
            self.model.unet = get_peft_model(self.model.unet, peft_config)
            self._lora_peft_config = peft_config

        # conv_in stays trainable under LoRA ONLY when it was rebuilt for channel
        # expansion (training from scratch). When finetuning a base that already has
        # the right in_channels, conv_in is pretrained and we keep it frozen so LoRA
        # is the only thing learning (avoids overfitting conv_in on small data).
        if getattr(self, "conv_in_replaced", False):
            for p in self.model.unet.conv_in.parameters():
                p.requires_grad = True
            logging.info("conv_in was rebuilt -> kept trainable under LoRA")
        else:
            logging.info("conv_in is pretrained -> frozen under LoRA")

    def _get_conv_in(self):
        """Return the actual conv_in module, unwrapping PeftModel if needed."""
        unet = self.model.unet
        # PeftModel exposes the wrapped UNet via get_base_model()
        get_base = getattr(unet, "get_base_model", None)
        if callable(get_base):
            try:
                unet = get_base()
            except Exception:
                unet = self.model.unet
        return unet.conv_in

    def _save_conv_in(self, ckpt_dir):
        """Persist the (trained) conv_in weights separately.

        Under LoRA, conv_in is fully trainable but is NOT part of the LoRA
        adapter (not in target_modules, not registered as modules_to_save), so
        PeftModel.save_pretrained silently drops it. We save it ourselves so the
        trained input layer is never lost.
        """
        from safetensors.torch import save_file

        conv_in = self._get_conv_in()
        state = {
            "weight": conv_in.weight.detach().cpu(),
            "bias": conv_in.bias.detach().cpu(),
        }
        path = os.path.join(ckpt_dir, "conv_in.safetensors")
        save_file(state, path)
        logging.info(f"conv_in weights saved to: {path}")

    def _load_conv_in(self, ckpt_path):
        """Load conv_in weights saved by _save_conv_in, if present."""
        from safetensors.torch import load_file

        path = os.path.join(ckpt_path, "conv_in.safetensors")
        if not os.path.isfile(path):
            logging.warning(
                f"No conv_in.safetensors found at {path}; conv_in keeps current weights."
            )
            return
        state = load_file(path)
        conv_in = self._get_conv_in()
        with torch.no_grad():
            conv_in.weight.copy_(state["weight"].to(conv_in.weight.device, conv_in.weight.dtype))
            conv_in.bias.copy_(state["bias"].to(conv_in.bias.device, conv_in.bias.dtype))
        logging.info(f"conv_in weights loaded from: {path}")

    def _replace_unet_conv_in(self):

        # replace the first layer to accept 8 or 12 in_channels
        _weight = self.model.unet.conv_in.weight.clone()
        _bias = self.model.unet.conv_in.bias.clone()
        # If checkpoint already has >4 channels (e.g. loaded from a prior run),
        # extract only the first 4 (original latent channels) before rebuilding.
        if _weight.shape[1] > 4:
            _weight = _weight[:, :4, :, :].clone()
        if self.dual_input:
            _weight = _weight.repeat((1, 3, 1, 1))  # 4 → 12
            _weight *= 0.33
        else:
            _weight = _weight.repeat((1, 2, 1, 1))  # 4 → 8
            _weight *= 0.5

        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            self.input_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.model.unet.config["in_channels"] = self.input_channels
        logging.info("Unet config is updated")
        return
    
    @abstractmethod
    def _load_train_data(self, batch):
        raise NotImplementedError
    
    @abstractmethod
    def _load_val_data(self, batch):
        raise NotImplementedError

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rgb, corrupted_rgb1, corrupted_rgb2 = self._load_train_data(batch)

                batch_size = rgb.shape[0]

                with torch.no_grad():
                    # Encode gt image
                    gt_target_latent = self.encode_rgb(rgb)  # [B, 4, h, w]
                    # Encode corrupted gs image
                    corrupted_rgb1_latent = self.encode_rgb(corrupted_rgb1)  # [B, 4, h, w]
                    if self.dual_input:
                        # Encode corrupted mesh image
                        corrupted_rgb2_latent = self.encode_rgb(corrupted_rgb2)  # [B, 4, h, w]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.timestep_max,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        gt_target_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        gt_target_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    gt_target_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Concat condition and target latents
                if self.dual_input:
                    cat_latents = torch.cat(
                        [corrupted_rgb2_latent, corrupted_rgb1_latent, noisy_latents], dim=1
                    )  # [B, 12, h, w]
                else:
                    cat_latents = torch.cat(
                        [corrupted_rgb1_latent, noisy_latents], dim=1
                    )  # [B, 8, h, w]
                cat_latents = cat_latents.float()

                # Predict the noise residual
                if self.use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model_pred = self.model.unet(
                            cat_latents, timesteps, text_embed
                        ).sample  # [B, 4, h, w]
                else:
                    model_pred = self.model.unet(
                        cat_latents, timesteps, text_embed
                    ).sample  # [B, 4, h, w]
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                # Get the target for loss depending on the prediction type
                if "sample" == self.prediction_type:
                    target = gt_target_latent
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        gt_target_latent, noise, timesteps
                    )
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")

                latent_loss = self.loss(model_pred.float(), target.float())
                total_loss = latent_loss
                pixel_loss_value = None

                # ===== ABLATION[A4] Add pixel loss only when this switch is on. =====
                if self.pixel_loss_enabled:
                    if "sample" == self.prediction_type:
                        pred_x0_latent = model_pred
                    elif "epsilon" == self.prediction_type:
                        alpha_cumprod = self.training_noise_scheduler.alphas_cumprod.to(device)
                        alpha_t = alpha_cumprod[timesteps].view(-1, 1, 1, 1)
                        sqrt_alpha_t = torch.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
                        pred_x0_latent = (noisy_latents - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t
                    elif "v_prediction" == self.prediction_type:
                        alpha_cumprod = self.training_noise_scheduler.alphas_cumprod.to(device)
                        alpha_t = alpha_cumprod[timesteps].view(-1, 1, 1, 1)
                        sqrt_alpha_t = torch.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
                        pred_x0_latent = sqrt_alpha_t * noisy_latents - sqrt_one_minus_alpha_t * model_pred
                    else:
                        raise ValueError(f"Unknown prediction type {self.prediction_type}")

                    pred_latent_for_pixel = pred_x0_latent
                    gt_rgb_for_pixel = rgb

                    if self.pixel_loss_max_samples > 0 and pred_latent_for_pixel.shape[0] > self.pixel_loss_max_samples:
                        pred_latent_for_pixel = pred_latent_for_pixel[:self.pixel_loss_max_samples]
                        gt_rgb_for_pixel = gt_rgb_for_pixel[:self.pixel_loss_max_samples]

                    if self.pixel_loss_latent_downsample > 1:
                        pred_latent_for_pixel = F.interpolate(
                            pred_latent_for_pixel,
                            scale_factor=1.0 / self.pixel_loss_latent_downsample,
                            mode="bilinear",
                            align_corners=False,
                        )

                    pred_rgb = self.decode_rgb(pred_latent_for_pixel)
                    if pred_rgb.shape[-2:] != gt_rgb_for_pixel.shape[-2:]:
                        gt_rgb_for_pixel = F.interpolate(
                            gt_rgb_for_pixel,
                            size=pred_rgb.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    pixel_loss_value = self.pixel_loss_fn(pred_rgb.float(), gt_rgb_for_pixel.float())
                    total_loss = total_loss + self.pixel_loss_weight * pixel_loss_value

                loss = total_loss.mean()

                self.train_metrics.update("loss", loss.item())
                self.train_metrics.update("latent_loss", latent_loss.mean().item())
                if pixel_loss_value is not None:
                    self.train_metrics.update("pixel_loss", pixel_loss_value.mean().item())
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    # ===== ABLATION[A8] Optional global-norm gradient clipping. =====
                    clipping_cfg = getattr(self.cfg, "gradient_clipping", None)
                    if clipping_cfg is not None:
                        clipping_enabled = bool(getattr(clipping_cfg, "enabled", False))
                        max_grad_norm = float(getattr(clipping_cfg, "max_norm", 0.0))
                        if not clipping_enabled:
                            max_grad_norm = 0.0
                    else:
                        # Backward compatibility with checkpoints/configs created
                        # before the explicit ablation switch was introduced.
                        max_grad_norm = float(getattr(self.cfg, "max_grad_norm", 0.0))
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in self.optimizer.param_groups for p in group["params"]],
                            max_grad_norm,
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dict(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_rgb(self, image_in):
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        latent = self.model.encode_rgb(image_in)
        return latent
    
    def decode_rgb(self, image_latent_in):
        assert len(image_latent_in.shape) == 4 and image_latent_in.shape[1] == 4
        rgb = self.model.decode_rgb(image_latent_in)
        return rgb

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        was_training = self.model.unet.training
        self.model.unet.eval()
        try:
            for i, val_loader in enumerate(self.val_loaders):
                val_dataset_name = val_loader.dataset.disp_name
                val_metric_dict = self.validate_single_dataset(
                    data_loader=val_loader, metric_tracker=self.val_metrics
                )
                logging.info(
                    f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
                )
                tb_logger.log_dict(
                    {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                    global_step=self.effective_iter,
                )
                # save to file
                eval_text = eval_dict_to_text(
                    val_metrics=val_metric_dict,
                    dataset_name=val_dataset_name,
                    sample_list_path=val_loader.dataset.disp_name,
                )
                _save_to = os.path.join(
                    self.out_dir_eval,
                    f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
                )
                with open(_save_to, "w+") as f:
                    f.write(eval_text)

                # Update main eval metric
                if 0 == i:
                    main_eval_metric = val_metric_dict[self.main_val_metric]
                    if (
                        "minimize" == self.main_val_metric_goal
                        and main_eval_metric < self.best_metric
                        or "maximize" == self.main_val_metric_goal
                        and main_eval_metric > self.best_metric
                    ):
                        self.best_metric = main_eval_metric
                        logging.info(
                            f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                        )
                        # Save a timestamped checkpoint and overwrite the fixed "best" slot
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                        )
                        self.save_checkpoint(
                            ckpt_name="best", save_train_state=False
                        )
        finally:
            if was_training:
                self.model.unet.train()

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_gt, corrupted_rgb1, corrupted_rgb2 = self._load_val_data(batch)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict
            pipe_out: MarigoldGSFixerOutput = self.model(
                corrupted_rgb1,
                corrupted_rgb2,
                denoising_steps=self.cfg.validation.denoising_steps,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            rgb_pred_ts: torch.Tensor = pipe_out.fixed_rgb_ts

            # Evaluate
            sample_metric = []

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(rgb_pred_ts.unsqueeze(0), rgb_gt.cpu()).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            # Save as image
            if save_to_dir is not None:
                img_name = os.path.basename(batch["rgb_path"][0])
                png_save_path = os.path.join(save_to_dir, img_name)
                pipe_out.fixed_rgb.save(png_save_path)

        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        if self.lora_enabled:
            # Always save LoRA adapter for resuming/continued finetune
            lora_adapter_path = os.path.join(ckpt_dir, "lora")
            self.model.unet.save_pretrained(lora_adapter_path, safe_serialization=True)
            logging.info(f"LoRA adapter is saved to: {lora_adapter_path}")

            # conv_in is trainable but not captured by the LoRA adapter; save it
            # separately so the trained input layer can always be restored.
            self._save_conv_in(ckpt_dir)

            if self.lora_save_merged:
                # Export a clean UNet with LoRA merged for inference (no peft required).
                # Merge on a DEEP COPY so the live training model and optimizer state
                # are never touched — merging in place would unload the LoRA wrappers
                # and force an optimizer reset on every save, disrupting training.
                try:
                    unet_copy = copy.deepcopy(self.model.unet)
                    merged_unet = unet_copy.merge_and_unload()
                    merged_unet.save_pretrained(unet_path, safe_serialization=True)
                    del unet_copy, merged_unet
                    logging.info(f"Merged UNet is saved to: {unet_path}")
                except Exception as e:
                    logging.warning(f"LoRA merge_and_unload failed, fallback to raw save: {e}")
                    self.model.unet.save_pretrained(unet_path, safe_serialization=True)
            else:
                self.model.unet.save_pretrained(unet_path, safe_serialization=True)
                logging.info(f"UNet (with LoRA wrappers) is saved to: {unet_path}")
        else:
            self.model.unet.save_pretrained(unet_path, safe_serialization=True)
            logging.info(f"UNet is saved to: {unet_path}")


        # Save scheduler
        scheduelr_path = os.path.join(ckpt_dir, "scheduler")
        self.model.scheduler.save_pretrained(scheduelr_path)
        logging.info(f"Scheduler is saved to: {scheduelr_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")

        lora_dir = os.path.join(ckpt_path, "lora")
        if self.lora_enabled and os.path.isdir(lora_dir):
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError("LoRA checkpoint detected but peft not installed.") from e

            # Reset to base UNet by merging current adapters off, then load saved adapter.
            try:
                base_unet = self.model.unet.get_base_model()
            except Exception:
                base_unet = self.model.unet
            self.model.unet = PeftModel.from_pretrained(base_unet, lora_dir, is_trainable=True)
            self.model.unet.to(self.device)
            logging.info(f"LoRA adapter loaded from {lora_dir}")

            # Restore the separately-saved conv_in (adapter does not contain it).
            self._load_conv_in(ckpt_path)
            # Keep conv_in trainable on resume when it was rebuilt for channel expansion.
            if getattr(self, "conv_in_replaced", False):
                for p in self._get_conv_in().parameters():
                    p.requires_grad = True
        else:
            # Load UNet
            _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
            self.model.unet.load_state_dict(
                torch.load(_model_path, map_location=self.device)
            )
            self.model.unet.to(self.device)
            logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
