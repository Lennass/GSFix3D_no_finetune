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


import cv2
import os
import copy
import random
import torch
import numpy as np
from PIL import Image
from enum import Enum
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize


class DatasetMode(Enum):
    TRAIN = "train"
    EVAL = "evaluate"
    VIS = "vis"


def default_rgb_transform(x):
    return x / 255.0 * 2 - 1


class BaseDataset(Dataset):
    """
    Base dataset providing shared utilities and common fields used by subclasses.

    Features:
      - Initialization arguments: mode (DatasetMode), dataset_dir, disp_name,
        dual_input, resize_to_hw, rgb_transform, and optional filename_ls_path.
      - Loads a semantic-mask dataset ("paint-by-inpaint/PIPE_Masks") for augmentations.
      - Helper methods:
          _read_image(path) -> np.ndarray: open image and return HxWxC numpy array.
          _prepare_image(image) -> (torch.IntTensor, torch.FloatTensor): convert uint8
            image to an int tensor and a normalized float tensor in [-1, 1].
          _training_preprocess(rasters): resize rasters when resize_to_hw is set.
          _sample_mask(split, width, height): sample a sparse semantic mask and a
            blurred version resized to (width, height) (ensures foreground < 10%).
      - Subclasses (BaseTrainDataset, BaseFinetuneDataset) implement dataset-specific
        loading, indexing and augmentation logic.
    """

    def __init__(
        self,
        mode: DatasetMode,
        dataset_dir: str,
        disp_name: str,
        dual_input: bool = False,
        resize_to_hw=None,
        rgb_transform=default_rgb_transform,
        filename_ls_path: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.dataset_dir = dataset_dir
        assert os.path.exists(self.dataset_dir), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.dual_input = dual_input

        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform

        # optional filenames list (used by BaseTrainDataset)
        self.filename_ls_path = filename_ls_path
        if self.filename_ls_path is not None:
            with open(self.filename_ls_path, "r") as f:
                self.filenames = [s.split() for s in f.readlines()]

        # Load semantic masks for augmentation
        data_files = {"train": "data/train-*", "test": "data/test-*"}
        self.dataset_masks = load_dataset("paint-by-inpaint/PIPE_Masks", data_files=data_files)

    def _read_image(self, image_path) -> np.ndarray:
        """
        Open image from filesystem.
        Returns numpy array HxWxC.
        """
        image = Image.open(image_path)
        return np.asarray(image)
    
    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to torch tensor.
        """
        image = np.transpose(image, (2, 0, 1)).astype(int)  # [C, H, W]
        image_norm = image / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        image = torch.from_numpy(image).int()
        image_norm = torch.from_numpy(image_norm).float()
        return image, image_norm   
    
    def _training_preprocess(self, rasters):
        """
        Shared preprocessing.
        """
        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.BILINEAR
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}
        return rasters
    
    def _sample_mask(self, split, width, height):
        while True:
            idx = random.randint(0, len(self.dataset_masks[split]) - 1)
            m = self.dataset_masks[split][idx]["mask"]
            m = np.asarray(m.convert("L"))
            ratio = np.sum(m > 0) / (m.shape[0] * m.shape[1])
            if ratio < 0.1:
                m = np.repeat(m[..., np.newaxis], 3, axis=-1) / 255.0
                m = cv2.resize(m, (width, height))
                blurred_m = cv2.GaussianBlur(m, (51, 51), sigmaX=10, sigmaY=10)
                return m, blurred_m


class BaseTrainDataset(BaseDataset):
    def __init__(
        self,
        mode: DatasetMode,
        dataset_dir: str,
        disp_name: str,
        dual_input: bool = False,
        resize_to_hw=None,
        rgb_transform=default_rgb_transform,  #  [0, 255] -> [-1, 1],
        filename_ls_path: str = None,
        hole_aug: dict = None,
        **kwargs,
    ) -> None:

        if filename_ls_path is None:
            raise ValueError("filename_ls_path must be provided for BaseTrainDataset.")

        super().__init__(
            mode=mode,
            filename_ls_path=filename_ls_path,
            dataset_dir=dataset_dir,
            disp_name=disp_name,
            dual_input=dual_input,
            resize_to_hw=resize_to_hw,
            rgb_transform=rgb_transform,
        )

        # Stronger, GS-artifact-like hole augmentation for pretraining. When
        # disabled (default), the legacy PIPE semantic-mask holes are used. When
        # enabled, we additionally punch large, dark, irregular holes into the
        # rendered gs/mesh inputs so the model learns to inpaint real GS black
        # holes zero-shot. Only affects pretraining (BaseTrainDataset); finetune
        # keeps mask aug off (see BaseFinetuneDataset).
        # ===== ABLATION[A3] Read the GS black-hole augmentation switch. =====
        hole_aug = hole_aug or {}
        self.hole_aug_enabled = bool(hole_aug.get("enabled", False))
        self.hole_aug_prob = float(hole_aug.get("prob", 0.5))
        self.hole_aug_max_holes = int(hole_aug.get("max_holes", 3))
        self.hole_aug_size_frac = list(hole_aug.get("size_frac", [0.05, 0.30]))
        self.hole_aug_darkness = float(hole_aug.get("darkness", 0.0))
        self.hole_aug_feather = float(hole_aug.get("feather", 0.4))

    def __len__(self):
        return len(self.filenames)

    def _punch_gs_holes(self, image: np.ndarray) -> np.ndarray:
        """Punch large, dark, irregular holes into an HxWxC uint8 image to mimic
        real 3DGS black holes / missing geometry. Returns a new array.

        Holes are built from a few overlapping ellipses at random positions, then
        blurred at the edges (feather) so the border is soft rather than a clean
        cut — closer to how GS artifacts fade out. `darkness` is the value the
        hole is pushed toward (0 = black).
        """
        h, w = image.shape[:2]
        n_holes = random.randint(1, max(1, self.hole_aug_max_holes))
        mask = np.zeros((h, w), dtype=np.float32)
        min_frac, max_frac = self.hole_aug_size_frac
        for _ in range(n_holes):
            frac = random.uniform(min_frac, max_frac)
            radius = max(4, int(frac * min(h, w)))
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            # Irregular: a couple of overlapping ellipses with random axes/angle.
            for _ in range(random.randint(1, 3)):
                ax = max(3, int(radius * random.uniform(0.5, 1.2)))
                ay = max(3, int(radius * random.uniform(0.5, 1.2)))
                angle = random.uniform(0, 360)
                ox = cx + random.randint(-radius // 2, radius // 2)
                oy = cy + random.randint(-radius // 2, radius // 2)
                cv2.ellipse(mask, (ox, oy), (ax, ay), angle, 0, 360, 1.0, -1)
        # Feather the edges.
        if self.hole_aug_feather > 0:
            k = max(3, int(self.hole_aug_feather * radius) | 1)  # odd kernel
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0.0, 1.0)[..., np.newaxis]  # HxWx1
        target = np.full_like(image, int(self.hole_aug_darkness * 255), dtype=np.float32)
        out = image.astype(np.float32) * (1.0 - mask) + target * mask
        return out.astype(image.dtype)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path = self._get_data_path(index=index)
        rasters = {}
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))
        other = {"index": index, "rgb_path": os.path.join(self.dataset_dir, rgb_rel_path)}
        return rasters, other

    def _get_data_path(self, index):
        filename_line = self.filenames[index]
        rgb_rel_path = filename_line[0]
        return rgb_rel_path

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_image(os.path.join(self.dataset_dir, rgb_rel_path))

        # Random semantic mask augmentation
        split = "train" if DatasetMode.TRAIN == self.mode else "test"
        if random.random() < 0.3:
            mask, blurred_mask = self._sample_mask(split, rgb.shape[1], rgb.shape[0])
            gs_image = rgb * (1 - blurred_mask)
            if self.dual_input:
                mesh_image = rgb * (1 - mask)
        else:
            gs_image = copy.deepcopy(rgb)
            if self.dual_input:
                mesh_image = copy.deepcopy(rgb)

        # Further random mask for gs_image
        if random.random() < 0.3:
            _, blurred_mask = self._sample_mask(split, rgb.shape[1], rgb.shape[0])
            gs_image = gs_image * (1 - blurred_mask)

        # Stronger GS-artifact-like holes (large / dark / irregular). Applied
        # independently per input stream so gs and mesh get different holes.
        # ===== ABLATION[A3] Apply irregular black holes only when enabled. =====
        if self.hole_aug_enabled and DatasetMode.TRAIN == self.mode:
            if random.random() < self.hole_aug_prob:
                gs_image = self._punch_gs_holes(np.asarray(gs_image).astype(np.uint8))
            if self.dual_input and random.random() < self.hole_aug_prob:
                mesh_image = self._punch_gs_holes(np.asarray(mesh_image).astype(np.uint8))

        rgb, rgb_norm = self._prepare_image(rgb)
        gs_image, gs_norm = self._prepare_image(gs_image)
        outputs = {
            "rgb_int": rgb,
            "rgb_norm": rgb_norm,
            "gs_int": gs_image,
            "gs_norm": gs_norm,
        }
        
        if self.dual_input:
            mesh_image, mesh_norm = self._prepare_image(mesh_image)
            outputs.update({"mesh_int": mesh_image,
                            "mesh_norm": mesh_norm})

        return outputs


class BaseFinetuneDataset(BaseDataset):
    def __init__(
        self,
        mode: DatasetMode,
        dataset_dir: str,
        disp_name: str,
        dual_input: bool = False,
        resize_to_hw=None,
        rgb_transform=default_rgb_transform,  #  [0, 255] -> [-1, 1],
        subset_method: str = "random",
        subset_seed: int = 2025,
        mask_aug_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            dataset_dir=dataset_dir,
            disp_name=disp_name,
            dual_input=dual_input,
            resize_to_hw=resize_to_hw,
            rgb_transform=rgb_transform,
        )

        # Probability of injecting a random semantic mask "hole" into the rendered
        # GS/mesh inputs. Useful as regularization when pretraining on large data,
        # but on single-scene finetuning it creates fake holes unrelated to the real
        # GS black holes and pollutes inpainting -> default off (0.0) for finetune.
        self.mask_aug_prob = float(mask_aug_prob)

        # Load dataset file paths for finetuning
        def _list_images_in_dir(d):
            allowed_exts = {".jpg", ".jpeg", ".png"}
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Directory not found: {d}")
            files = [os.path.join(d, f) for f in os.listdir(d)
                if os.path.isfile(os.path.join(d, f)) and os.path.splitext(f)[1].lower() in allowed_exts
            ]
            return sorted(files)

        self.gt_rgb_path = _list_images_in_dir(os.path.join(self.dataset_dir, "rgb"))
        self.rendered_gs_path = _list_images_in_dir(os.path.join(self.dataset_dir, "rendered_gs"))

        assert len(self.gt_rgb_path) == len(self.rendered_gs_path), "The number of images doesn't match!"

        if self.dual_input:
            self.rendered_mesh_path = _list_images_in_dir(os.path.join(self.dataset_dir, "rendered_mesh"))
            assert len(self.gt_rgb_path) == len(self.rendered_mesh_path), "The number of images doesn't match!"
        
        n_samples = len(self.gt_rgb_path)
        self.subset_method = subset_method
        self.subset_seed = subset_seed

        # Split the scene's frames into a held-out eval/vis subset and a training
        # subset that EXCLUDES it. Previously TRAIN used every frame while EVAL/VIS
        # randomly sampled from the same full set, so the eval frames were always a
        # subset of the training frames (data leakage). For per-scene finetuning we
        # must measure novel-view (held-out) quality, so train and eval are now
        # disjoint. The eval subset is the same regardless of mode (fixed by
        # subset_seed); TRAIN takes the complement.
        eval_ratio = 0.1
        eval_num = max(1, int(eval_ratio * n_samples))
        if self.subset_method == "head":
            eval_indices = set(range(eval_num))
        elif self.subset_method == "random":
            subset_rng = random.Random(self.subset_seed)
            eval_indices = set(subset_rng.sample(range(n_samples), eval_num))
        else:
            raise ValueError(f"Unknown subset_method: {self.subset_method}")

        if self.mode == DatasetMode.TRAIN:
            keep_indices = [i for i in range(n_samples) if i not in eval_indices]
        elif self.mode == DatasetMode.EVAL:
            keep_indices = sorted(eval_indices)
        elif self.mode == DatasetMode.VIS:
            # VIS is a small preview drawn from the held-out eval frames (never
            # from training frames), so visualizations reflect novel views too.
            vis_num = max(1, int(0.1 * len(eval_indices)))
            keep_indices = sorted(eval_indices)[:vis_num]
        else:
            keep_indices = list(range(n_samples))

        self.gt_rgb_path = [self.gt_rgb_path[i] for i in keep_indices]
        self.rendered_gs_path = [self.rendered_gs_path[i] for i in keep_indices]
        if self.dual_input:
            self.rendered_mesh_path = [self.rendered_mesh_path[i] for i in keep_indices]

        self.n_samples = len(self.gt_rgb_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        gt_rgb_file = self.gt_rgb_path[index]   
        rendered_gs_file = self.rendered_gs_path[index]

        gt_rgb = self._read_image(gt_rgb_file)
        rendered_gs = self._read_image(rendered_gs_file)
        
        if self.dual_input:
            rendered_mesh_file = self.rendered_mesh_path[index]
            rendered_mesh = self._read_image(rendered_mesh_file)
        
        # Random semantic mask augmentation
        if random.random() < self.mask_aug_prob:
            mask, blurred_mask = self._sample_mask("test", gt_rgb.shape[1], gt_rgb.shape[0])
            rendered_gs = rendered_gs * (1 - blurred_mask)
            if self.dual_input:
                rendered_mesh = rendered_mesh * (1 - mask)

        # Further random mask for gs_image
        if random.random() < self.mask_aug_prob:
            _, blurred_mask = self._sample_mask("test", gt_rgb.shape[1], gt_rgb.shape[0])
            rendered_gs = rendered_gs * (1 - blurred_mask)

        gt_rgb, gt_rgb_norm = self._prepare_image(gt_rgb)
        rendered_gs, rendered_gs_norm = self._prepare_image(rendered_gs)  
        outputs = {
            "gt_rgb": gt_rgb,
            "gt_rgb_norm": gt_rgb_norm,
            "rendered_gs": rendered_gs,
            "rendered_gs_norm": rendered_gs_norm,
        }

        if self.dual_input:
            rendered_mesh, rendered_mesh_norm = self._prepare_image(rendered_mesh)
            outputs.update({
                "rendered_mesh": rendered_mesh,
                "rendered_mesh_norm": rendered_mesh_norm,
            })  

        if DatasetMode.TRAIN == self.mode:
            outputs = self._training_preprocess(outputs)
            
        outputs.update({"index": index, "rgb_path": gt_rgb_file})
        
        return outputs
