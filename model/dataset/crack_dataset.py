import copy
import torchvision.transforms as transforms
import random, pickle
import torch
import cv2
from ..preprocessing.Discrete_Fourier_transform import DFT, img_range
import os


class elpv(torch.utils.data.Dataset):
        def __init__(self, path, mode='train', types=['mono', 'poly']):

            self.mode = mode
            self.type = types
            self.path = path
            self.infos_path = os.path.join(path, f"elpv_infos_{types}_train.pkl") \
                if mode == 'train' else os.path.join(path, f"elpv_infos_{types}_test.pkl")

            with open(self.infos_path, 'rb') as f:
                self.samples = pickle.load(f)

            self.to_tensor = transforms.ToTensor()

            self.transform_t_geometric = transforms.Compose([
                transforms.Resize((288, 288)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.2),
            ])

            self.transform_t_colour = transforms.Compose([
                transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.3))
            ])

            self.resized_v = transforms.Resize((288, 288))
            self.normalized_tv_img = transforms.Normalize(mean=[0.5968], std=[0.0977])
            self.normalized_tv_filter = transforms.Normalize(mean=[0.1], std=[0.0984])

        def __getitem__(self, index):
            img_path = os.path.join(self.path, self.samples[index]["Path"])
            lab = int(self.samples[index]["Class"])

            if self.samples[index]["Type"] != self.type:
                raise f"TypeMatchError: Having issue in {self.samples[index]}"

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            filter_image = img_range(DFT(image, offset=5, mode='HP'))

            if self.mode == "train":
                image = self.to_tensor(image)
                filter_image = self.to_tensor(filter_image)
                img_cat_filter = torch.cat([image, filter_image], dim=0)
                img_cat_filter_geo = self.transform_t_geometric(img_cat_filter)
                filter_image = img_cat_filter_geo[1].unsqueeze(0)
                image_colour_aug = self.transform_t_colour(img_cat_filter_geo[0].unsqueeze(0))
                image = self.normalized_tv_img(image_colour_aug)
                filter_image = self.normalized_tv_filter(filter_image)
            else:
                image = self.to_tensor(image)
                filter_image = self.to_tensor(filter_image)
                image = self.resized_v(image)
                filter_image = self.resized_v(filter_image)
                image = self.normalized_tv_img(image)
                filter_image = self.normalized_tv_filter(filter_image)

            label = torch.zeros(2)
            label[lab] = 1

            return image, filter_image, label, self.samples[index]["Path"]

        def __len__(self):
            return len(self.samples)
