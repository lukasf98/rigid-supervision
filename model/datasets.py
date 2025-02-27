from torch.utils.data import Dataset
import os
import json
import itk
import torch
import numpy as np
from multiprocessing import Manager


class NLSTDataset(Dataset):
    """
    A PyTorch Dataset for the NLST data.

    This dataset loads images, labels, and displacement fields from the provided directories.
    It also supports caching using a shared multiprocessing Manager if desired.
    """
    def __init__(self, data_dir, rigid_data_dir, ts_data_dir, lv_data_dir,
                 stage='train', use_cache=True, shared_cache=True):
        data_json = os.path.join(data_dir, "NLST_dataset.json")
        with open(data_json, "r") as file:
            data = json.load(file)

        # Build a list of file dictionaries for the training pairs.
        train_files = []
        for pair in data["training_paired_images"]:
            nam_fixed = os.path.basename(pair["fixed"]).split(".")[0]
            nam_moving = os.path.basename(pair["moving"]).split(".")[0]
            # Extract secondary name components for rigid/ts data.
            nam_fixed2 = os.path.basename(pair["fixed"]).split(".")[0].split("_")[1]
            nam_moving2 = os.path.basename(pair["moving"]).split(".")[0].split("_")[1]
            train_files.append({
                "fixed_image": os.path.join(data_dir, "imagesTr", f"{nam_fixed}_0000_0000.nii.gz"),
                "moving_image": os.path.join(data_dir, "imagesTr", f"{nam_moving}_0000_0000.nii.gz"),
                "fixed_label": os.path.join(data_dir, "masksTr", f"{nam_fixed}.nii.gz"),
                "moving_label": os.path.join(data_dir, "masksTr", f"{nam_moving}.nii.gz"),
                "fixed_keypoints": os.path.join(data_dir, "keypointsTr", f"{nam_fixed}.csv"),
                "moving_keypoints": os.path.join(data_dir, "keypointsTr", f"{nam_moving}.csv"),
                "fixed_rigid_label": os.path.join(
                    rigid_data_dir, f"{nam_moving2}_0001_to_0000", "0000_rigid_masks_combined.nii.gz"
                ),
                "moving_rigid_label": os.path.join(
                    rigid_data_dir, f"{nam_moving2}_0001_to_0000", "0001_rigid_masks_combined.nii.gz"
                ),
                "fixed_lv_label": os.path.join(ts_data_dir, f"{nam_fixed}.nii"),
                "moving_lv_label": os.path.join(ts_data_dir, f"{nam_moving}.nii"),
                "rigid_00001_to_0000_label": os.path.join(
                    rigid_data_dir, f"{nam_moving2}_0001_to_0000", "rigid_target_mask.npy"
                ),
                "rigid_00000_to_0001_label": os.path.join(
                    rigid_data_dir, nam_moving2, "rigid_target_mask.npy"
                ),
                "rigid_00001_to_0000_ddf": os.path.join(
                    rigid_data_dir, f"{nam_moving2}_0001_to_0000", "rigid_mask.npy"
                ),
                "rigid_00000_to_0001_ddf": os.path.join(
                    rigid_data_dir, nam_moving2, "rigid_mask.npy"
                ),
                "def_00001_to_0000": os.path.join(
                    lv_data_dir, f"{nam_moving2}_0001_to_0000", "displacement_field_itk.nii.gz"
                ),
                "def_00000_to_0001": os.path.join(
                    lv_data_dir, f"{nam_moving2}_0000_to_0001", "displacement_field_itk.nii.gz"
                ),
            })

        # Split data into training, validation, and test sets.
        split_idx1 = int(len(train_files) * 0.8)
        split_idx2 = int(len(train_files) * 0.9)
        train_files, val_files, _ = (
            train_files[:split_idx1],
            train_files[split_idx1:split_idx2],
            train_files[split_idx2:],
        )
        self.data_dicts = train_files if stage == 'train' else val_files

        self.use_cache = use_cache
        if self.use_cache:
            self._cache = Manager().dict() if shared_cache else {}
        else:
            self._cache = None

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        print(f"Adding {idx} to cache")
        data_item = self.data_dicts[idx]

        # Load ITK images.
        fixed_image = self.load_itk_image(data_item["fixed_image"])
        moving_image = self.load_itk_image(data_item["moving_image"])
        fixed_label = self.load_itk_image(data_item["fixed_label"])
        moving_label = self.load_itk_image(data_item["moving_label"])
        fixed_rigid_label = self.load_itk_image(data_item["fixed_rigid_label"])
        moving_rigid_label = self.load_itk_image(data_item["moving_rigid_label"])
        fixed_lv_label = self.load_itk_image(data_item["fixed_lv_label"])
        moving_lv_label = self.load_itk_image(data_item["moving_lv_label"])

        # Load displacement fields and rigid labels.
        rigid_00001_to_0000_ddf = np.load(data_item["rigid_00001_to_0000_ddf"])
        rigid_00000_to_0001_ddf = np.load(data_item["rigid_00000_to_0001_ddf"])
        rigid_00001_to_0000_label = np.load(data_item["rigid_00001_to_0000_label"])
        rigid_00000_to_0001_label = np.load(data_item["rigid_00000_to_0001_label"])

        def_00001_to_0000_ddf = itk.imread(data_item["def_00001_to_0000"])
        def_00000_to_0001_ddf = itk.imread(data_item["def_00000_to_0001"])
        ref_image = itk.imread(data_item["fixed_image"])

        # Scale image intensity to [0, 1].
        fixed_image = self.scale_intensity_range(fixed_image, -1200, 400, 0.0, 1.0, True)
        moving_image = self.scale_intensity_range(moving_image, -1200, 400, 0.0, 1.0, True)

        # Add channel dimension.
        fixed_image = fixed_image[None, ...]
        moving_image = moving_image[None, ...]
        fixed_label = fixed_label[None, ...]
        moving_label = moving_label[None, ...]
        fixed_rigid_label = fixed_rigid_label[None, ...]
        moving_rigid_label = moving_rigid_label[None, ...]
        fixed_lv_label = fixed_lv_label[None, ...]
        moving_lv_label = moving_lv_label[None, ...]

        sample = {
            "fixed_image": torch.tensor(fixed_image, dtype=torch.float32),
            "moving_image": torch.tensor(moving_image, dtype=torch.float32),
            "fixed_label": torch.tensor(fixed_label, dtype=torch.float32),
            "moving_label": torch.tensor(moving_label, dtype=torch.float32),
            "fixed_keypoints": data_item["fixed_keypoints"],
            "moving_keypoints": data_item["moving_keypoints"],
            "rigid_00001_to_0000_ddf": torch.tensor(rigid_00001_to_0000_ddf, dtype=torch.float32),
            "rigid_00000_to_0001_ddf": torch.tensor(rigid_00000_to_0001_ddf, dtype=torch.float32),
            "rigid_00001_to_0000_label": torch.tensor(rigid_00001_to_0000_label, dtype=torch.float32),
            "rigid_00000_to_0001_label": torch.tensor(rigid_00000_to_0001_label, dtype=torch.float32),
            "def_00001_to_0000_ddf": self.itk_to_monai_ddf(ref_image, def_00001_to_0000_ddf),
            "def_00000_to_0001_ddf": self.itk_to_monai_ddf(ref_image, def_00000_to_0001_ddf),
            "fixed_path": data_item["fixed_image"],
            "fixed_rigid_label": torch.tensor(fixed_rigid_label, dtype=torch.float32),
            "moving_rigid_label": torch.tensor(moving_rigid_label, dtype=torch.float32),
            "fixed_lv_label": torch.tensor(fixed_lv_label, dtype=torch.float32),
            "moving_lv_label": torch.tensor(moving_lv_label, dtype=torch.float32),
        }

        if self.use_cache:
            self._cache[idx] = sample

        return sample

    @staticmethod
    def scale_intensity_range(img, a_min, a_max, b_min, b_max, clip):
        """
        Scale the intensity range of an image.
        """
        img = (img - a_min) / (a_max - a_min)
        img = img * (b_max - b_min) + b_min
        if clip:
            img = np.clip(img, b_min, b_max)
        return img

    @staticmethod
    def load_itk_image(file_path):
        """
        Load an ITK image and return a numpy array view.
        """
        image = itk.imread(file_path)
        return itk.array_view_from_image(image)

    @staticmethod
    def itk_to_monai_ddf(image, ddf_itk):
        """
        Convert an ITK displacement field to a MONAI displacement field.
        """
        ddf_np = itk.array_view_from_image(ddf_itk)
        ndim = image.ndim

        # Correct for image direction.
        direction = np.asarray(image.GetDirection(), dtype=np.float64)
        ddf_np = np.einsum("ij,...j->...i", direction, ddf_np, dtype=np.float64).astype(np.float32)

        # Correct for image spacing.
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        ddf_np /= np.array(spacing, ndmin=ndim + 1)

        # Permute dimensions: x, y, z -> z, x, y.
        ddf_np = ddf_np[..., ::-1].copy()

        # Convert to tensor and adjust channel order.
        ddf = torch.tensor(ddf_np, dtype=torch.float32)
        ddf = ddf.permute(3, 0, 1, 2)

        return ddf
