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
# This file has been added by the GSFix3D maintainers (2026).
# SPDX-License-Identifier: Apache-2.0
#
# Export a self-contained, inference-ready GSFixer pipeline from a LoRA training
# checkpoint. A LoRA checkpoint stores three separate pieces:
#   1. the SD2 base pipeline (4-channel UNet + VAE + text_encoder + ...)
#   2. the trained LoRA adapter (attention layers only)        -> <ckpt>/lora
#   3. the trained, channel-expanded conv_in (NOT in the LoRA) -> <ckpt>/conv_in.safetensors
#
# This script assembles all three into one merged 12-channel (or 8-channel)
# pipeline that `inference.py` can load directly via from_pretrained, with no
# peft dependency at inference time.


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import argparse
import logging

import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from safetensors.torch import load_file

from marigold import MarigoldGSFixerPipeline


def _expand_conv_in(pipe, conv_in_state, dual_input):
    """Rebuild conv_in to the expanded channel count and load trained weights."""
    weight = conv_in_state["weight"]
    bias = conv_in_state["bias"]
    in_channels = weight.shape[1]
    expected = 12 if dual_input else 8
    if in_channels != expected:
        raise ValueError(
            f"conv_in.safetensors has in_channels={in_channels}, "
            f"but --dual_input implies {expected}. Mismatch."
        )
    out_channels = weight.shape[0]
    new_conv_in = Conv2d(
        in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    new_conv_in.weight = Parameter(weight.clone())
    new_conv_in.bias = Parameter(bias.clone())
    pipe.unet.conv_in = new_conv_in
    pipe.unet.config["in_channels"] = in_channels
    logging.info(f"conv_in rebuilt to {in_channels} channels from trained weights")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Merge a LoRA training checkpoint into a standalone GSFixer pipeline."
    )
    parser.add_argument(
        "--base_ckpt",
        type=str,
        required=True,
        help="Path to the SD2 base pipeline used for pretraining (e.g. the stable-diffusion-2 dir).",
    )
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        required=True,
        help="Path to the training checkpoint dir containing `lora/` and `conv_in.safetensors` "
        "(e.g. .../checkpoint/best).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write the merged, inference-ready pipeline.",
    )
    parser.add_argument(
        "--dual_input",
        action="store_true",
        help="Set if the model was trained with mesh+gs (12 channels). Omit for gs-only (8 channels).",
    )
    args = parser.parse_args()

    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft is required to merge a LoRA checkpoint.") from e

    lora_dir = os.path.join(args.lora_ckpt, "lora")
    conv_in_path = os.path.join(args.lora_ckpt, "conv_in.safetensors")
    scheduler_dir = os.path.join(args.lora_ckpt, "scheduler")

    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"LoRA adapter dir not found: {lora_dir}")
    if not os.path.isfile(conv_in_path):
        raise FileNotFoundError(
            f"conv_in.safetensors not found at {conv_in_path}. "
            f"This checkpoint predates the conv_in-saving fix and cannot be exported."
        )

    # 1. Load the SD2 base pipeline (4-channel UNet).
    logging.info(f"Loading base pipeline from {args.base_ckpt}")
    pipe: MarigoldGSFixerPipeline = MarigoldGSFixerPipeline.from_pretrained(args.base_ckpt)

    # 2. Rebuild conv_in to the expanded channel count with the TRAINED weights.
    conv_in_state = load_file(conv_in_path)
    _expand_conv_in(pipe, conv_in_state, args.dual_input)

    # 3. Attach the LoRA adapter to the (now 12-channel) UNet and merge it in.
    logging.info(f"Loading and merging LoRA adapter from {lora_dir}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir, is_trainable=False)
    pipe.unet = pipe.unet.merge_and_unload()
    logging.info("LoRA adapter merged into UNet")

    # 4. Prefer the trained scheduler if the checkpoint shipped one.
    if os.path.isdir(scheduler_dir):
        try:
            from diffusers import DDIMScheduler
            pipe.scheduler = DDIMScheduler.from_pretrained(scheduler_dir)
            logging.info(f"Scheduler loaded from {scheduler_dir}")
        except Exception as e:
            logging.warning(f"Could not load checkpoint scheduler, keeping base scheduler: {e}")

    # 5. Save the self-contained pipeline.
    os.makedirs(args.output_dir, exist_ok=True)
    pipe.save_pretrained(args.output_dir)
    logging.info(f"Merged pipeline saved to: {args.output_dir}")

    # Sanity check: conv_in must report the expanded channel count.
    saved_ic = pipe.unet.config["in_channels"]
    logging.info(f"Exported UNet in_channels = {saved_ic}")
    assert saved_ic == (12 if args.dual_input else 8), "Exported in_channels mismatch!"
    logging.info("Export complete. You can now run inference.py with --checkpoint set to output_dir.")


if "__main__" == __name__:
    main()
