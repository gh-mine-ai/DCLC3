from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os
import pandas as pd
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import argparse


# Existing Dataset Classes
class myDataset(Dataset):
    def __init__(self, csvfile_dir, augment_factor=1, transform=False):
        self.data = pd.read_csv(csvfile_dir)
        self.augment_factor = augment_factor
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data) * self.augment_factor if self.transform else len(self.data)

    def __getitem__(self, idx):
        base_idx = idx // self.augment_factor
        imageid = self.data.iloc[base_idx, 0]
        image_path = f"../data/dataset/training_patches/{imageid}"
        mask_path = image_path.replace("training_patches", "training_noisy_labels")

        try:
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
        except Exception as e:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)

        image = self.to_tensor(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            image, mask = augment_image_with_mask(image, mask)

        return image, mask


# Dataset class for Building datasets with accurate masks
class AccBuildingDataset(Dataset):
    def __init__(self, images, masks, augment_factor=1, transform=False, building_threshold=0.1):
        self.images = images
        self.masks = masks
        self.augment_factor = augment_factor
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.building_threshold = building_threshold
        self.valid_indices = self._filter_images()

    def _filter_images(self):
        valid_indices = []
        for idx, (image_path, mask_path) in enumerate(zip(self.images, self.masks)):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            building_area = (mask > 0).sum() / mask.size
            if building_area >= self.building_threshold:
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices) * 16 * self.augment_factor

    def __getitem__(self, idx):
        effective_index = (idx // self.augment_factor) % (len(self.valid_indices) * 16)
        image_idx = self.valid_indices[effective_index // 16]
        image_path = self.images[image_idx]
        mask_path = self.masks[image_idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        patch_idx = effective_index % 16
        x_offset = (patch_idx % 4) * 256
        y_offset = (patch_idx // 4) * 256
        image_patch = image[y_offset:y_offset + 256, x_offset:x_offset + 256]
        mask_patch = mask[y_offset:y_offset + 256, x_offset:x_offset + 256]

        image_patch = self.to_tensor(image_patch)
        mask_patch = torch.from_numpy(mask_patch).float() / 255.0
        mask_patch = mask_patch.unsqueeze(0)

        if self.transform:
            image_patch, mask_patch = augment_image_with_mask(image_patch, mask_patch)

        return image_patch, mask_patch


# Inference Dataset Class
class myInferDataset(Dataset):
    def __init__(self, images_list, masks_list, transform=False):
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.images_list = images_list
        self.masks_list = masks_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        mask_path = image_path.replace("training_patches", "training_noisy_labels")
        image_id = os.path.basename(image_path)

        try:
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise FileNotFoundError(f"File not found: {image_path} or {mask_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)

        if self.transform:
            image = self.to_tensor(image)
            mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        else:
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask, image_id


# Example augmentation function
def augment_image_with_mask(image, mask):
    return image, mask


## Lightning datamodule
class BuildingDataModule(LightningDataModule):
    def __init__(
        self,
        data_type = None,
        data_dir = None,
        transforms = None,
        augment_factor=1, 
        building_threshold=0.1,
        batch_size = 64,
        num_workers = 0,
        pin_memory= False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms
        self.data_type = data_type
        self.data_dir = data_dir

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.data_type == "acc_building":
                all_images = glob(f"{self.data_dir}/*.jpg")
                all_masks = [i.replace("Images" , "Masks").replace("jpg","tif") for i in all_images]
                # 80% in training and 20% in val
                n_train = int(len(all_images) * 0.8)
                # train dataset
                train_images = all_images[:n_train]
                train_masks = all_masks[:n_train]
                self.data_train =  AccBuildingDataset(
                    images=train_images,
                    masks=train_masks,
                    augment_factor=self.hparams.augment_factor, 
                    transform=self.transforms, 
                    building_threshold=self.hparams.building_threshold,
                )
                # val dataset
                val_images = all_images[n_train:]
                val_masks = all_masks[n_train:]
                self.data_val =  AccBuildingDataset(
                    images=val_images,
                    masks=val_masks,
                    augment_factor=self.hparams.augment_factor, 
                    transform=None, 
                    building_threshold=self.hparams.building_threshold,
                )
            else:
                raise Exception("Unknow dataset type!")
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        pass


if __name__ == "__main__":
    _ = BuildingDataModule()

