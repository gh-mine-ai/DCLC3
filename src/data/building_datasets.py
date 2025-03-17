from typing import Any, Dict, Optional, Tuple

from torchvision import transforms
from PIL import Image, ImageFilter
import random

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
import os
import pandas as pd
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import argparse




# Existing Dataset Classes
class ActualDataset(Dataset):
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
        image_path = f"data/dataset/training_patches/{imageid}"
        mask_path = image_path.replace("training_patches", "training_noisy_labels")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        image = self.to_tensor(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            image, mask = augment_image_with_mask(image, mask)
        # image = image.resize(224,224)
        # mask = mask.resize(224,224)
        return image, mask


# Example augmentation function

def augment_image_with_mask(image, mask):
    seed = np.random.randint(2147483647)
    
    spatial_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2)),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])
    
    photometric_transforms = transforms.Compose([
        transforms.ColorJitter(brightness = 0.4,saturation=0.3, hue=0.2,contrast =0.5),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.5)),
        transforms.Lambda(lambda img: transforms.F.adjust_brightness(img, 0.7)),
        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.005),
        # transforms.Lambda(lambda img: img * torch.tensor([0.75, 1.00, 0.75], device=img.device).view(3, 1, 1)),
        transforms.Lambda(lambda img: img + torch.tensor([0.00, 0.08, 0.00], device=img.device).view(3, 1, 1))
    ])
    
    torch.manual_seed(seed)
    image = spatial_transforms(image)
    torch.manual_seed(seed)
    mask = spatial_transforms(mask)

    image = photometric_transforms(image)


    return image, mask
    

# Dataset class for Building datasets with accurate masks

class AccBuildingDataset(Dataset):
    def __init__(self, images, masks, augment_factor=1, transform=True, building_threshold=0.15 ):
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


class NewNewDataset(Dataset):
    def __init__(self,images,masks,transform = False ):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)*196

    def __getitem__(self,idx):
        image_idx = idx // 196
        image_path = self.images[image_idx]
        mask_path = self.masks[image_idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        image = np.pad(image, ((66, 66), (40, 40), (0, 0)), mode='constant', constant_values=0)
        mask = np.pad(mask, ((66, 66), (40, 40), (0, 0)), mode='constant', constant_values=0)
        mask = np.where(mask[:,:,2] == 238 , 1,0)
        
        patch_idx = idx % 196
        x_offset = (patch_idx % 14) * 256
        y_offset = (patch_idx // 14) * 256
        image_patch = image[y_offset:y_offset + 256, x_offset:x_offset + 256]
        mask_patch = mask[y_offset:y_offset + 256, x_offset:x_offset + 256]
    
        image_patch = self.to_tensor(image_patch)
        mask_patch = torch.from_numpy(mask_patch).float()
        mask_patch = mask_patch.unsqueeze(0)

        if self.transform:
            image_patch, mask_patch = augment_image_with_mask(image_patch, mask_patch )

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

        return image, mask,image_id



## Lightning datamodule
class BuildingDataModule(LightningDataModule):
    def __init__(
        self,
        data_type = None,
        data_dir = None,
        transforms = False,
        augment_factor=1, 
        building_threshold=0.1,
        batch_size = 64,
        num_workers = 1,
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
                    transform = None, 
                    building_threshold=self.hparams.building_threshold,
                )


            elif self.data_type == "new_acc_building":
                all_images = glob(f"{self.data_dir}/*.tif")
                all_masks = [i.replace("Images" , "Masks") for i in all_images]
                # 80% in training and 20% in val
                n_train = int(len(all_images) * 0.8)
                # train dataset
                train_images = all_images[:n_train]
                train_masks = all_masks[:n_train]
                self.data_train =  NewNewDataset(
                    images=train_images,
                    masks=train_masks,
                    transform=self.transforms, 
                )
                # val dataset
                val_images = all_images[n_train:]
                val_masks = all_masks[n_train:]
                self.data_val =  NewNewDataset(
                    images=val_images,
                    masks=val_masks,
                    transform = False, 
                )

            elif self.data_type == "concat":
                
                data1_dir = "data/dataset/New_dataset/Images"
                images1 = glob(f"{data1_dir}/*.jpg")
                masks1 = [i.replace("Images" , "Masks").replace("jpg","tif") for i in images1]
        
                n_train1 = int(len(images1) * 0.8)
                
                train_images1 = images1[:n_train1]
                train_masks1 = masks1[:n_train1]
                
                train_dataset1 = AccBuildingDataset(
                    images=train_images1,
                    masks=train_masks1,
                    augment_factor=self.hparams.augment_factor, 
                    transform=self.transforms, 
                    building_threshold=self.hparams.building_threshold)

                val_masks1 = masks1[n_train1:]
                val_images1 =[i.replace("New_dataset","Concat_dim_val_images").replace("Masks","Val_images1").replace(".tif",".jpg")
                            for i in val_masks1]
                

                val_dataset1 = AccBuildingDataset(
                    images=val_images1,
                    masks=val_masks1,
                    augment_factor=self.hparams.augment_factor, 
                    transform=False,
                    building_threshold=self.hparams.building_threshold)

                
                data2_dir = "data/dataset/New_New_dataset/Images"
                images2 = glob(f"{data2_dir}/*.tif")
                masks2 = [i.replace("Images" , "Masks") for i in images2]

                n_train2 = int(len(images2) * 0.8)
                
                train_images2 = images2[:n_train2]
                train_masks2 = masks2[:n_train2]
                
                train_dataset2 = NewNewDataset(
                    images=train_images2,
                    masks=train_masks2,
                    transform=self.transforms)

                val_masks2 = masks2[n_train2:]
                val_images2 = [i.replace("New_New_dataset","Concat_dim_val_images").replace("Masks","Val_images2").replace(".tif",".jpg")
                               for i in val_masks2]
   
                val_dataset2 = NewNewDataset(
                    images=val_images2,
                    masks=val_masks2,
                    transform=False,
                    )
                
                train_dataset3 = ActualDataset(csvfile_dir = "data/train_shuffled.csv",
                                              transform = self.transforms,
                                              augment_factor=2)
                val_dataset3 = ActualDataset(csvfile_dir = "data/val_shuffled.csv",
                                              transform = False,
                                              augment_factor=2)
                
                train_list = [train_dataset1 , train_dataset2, train_dataset3]
                val_list =   [val_dataset1,    val_dataset2,   val_dataset3]
                self.data_train = ConcatDataset(train_list)
                self.data_val = ConcatDataset(val_list)

                # self.data_train = train_dataset3
                # self.data_val = val_dataset3
                
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

