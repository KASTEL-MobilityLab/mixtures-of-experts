"""
A2d2 dataset dataloader.
highway structure
./aev_data/highway
  /train_val
    /GT
      /20181204191844_label_frontcenter_000022904.png.png
    /Raw
      /20181204191844_camera_frontcenter_000022904.png
  /test

./aev_data/aev_sky_drivable/AEV_GT_ratio_max/
TODO: save train, val split as txt file
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import custom_transforms as tr
from dataloader.datasets.audi import Audi_Dataset_Labels
import matplotlib.pyplot as plt


def path_parsing(root, txt_file):
    """return absolute path of image file stored in txt"""
    files = []
    with open(txt_file, "r") as opened_file:
        for line in opened_file:
            name = line.strip().split("/")[-1]
            seq, label_camera, _, _ = name.split("_")
            seq = seq[:8] + "_" + seq[8:]
            filepath = os.path.join(
                *[root, seq, label_camera, "cam_front_center", name])
            files.append(filepath)
    return files


def get_gt_files(root, arguments, split):
    """return list of filename wrt split and expert"""
    gt_files = []

    if split in ["train", "val"]:
        if arguments["model"]=="moe" or "baseline" in arguments["model"]:
            # --- return all expert data subsets
            for dataset_name in ["highway", "urban"]:
                textfile_name = get_txtfilename(dataset_name, split)
                gt_files += path_parsing(root, textfile_name)
        else:
            # --- return single expert data subset
            textfile_name = get_txtfilename(arguments["train_dataset"], split)
            gt_files = path_parsing(root, textfile_name)

    elif split == "test":
        textfile_name = get_txtfilename(arguments["test_dataset"], split)
        gt_files += path_parsing(root, textfile_name)

    return gt_files


def get_txtfilename(dataset_name, split):  # TODO change PATH!
    textfile_name = ("./dataloader/audi_split_txt/" +
                     "_".join([dataset_name, split]) + "_gt.txt")
    return textfile_name


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
    :param rootdir is the root directory
    :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)
    ]


class AudiSplitSegmentation(data.Dataset):
    """Audi split segmentation"""
    NUM_CLASSES = 55

    def __init__(self,
                 arguments,
                 root="/disk/ml/datasets/aev/camera_lidar_semantic",
                 split="train"):
        self.root = root
        self.split = split
        self.params = arguments
        self.files_gt = {}
        self.files_gt[split] = get_gt_files(root, arguments, split)
        self.void_classes = []
        self.class_names = []
        self.valid_classes = []


        self.ignore_index = 255

        for label_description in Audi_Dataset_Labels:
            if (label_description.trainId == self.ignore_index
                    or label_description.trainId < 0):
                self.void_classes.append(label_description.trainId)
            else:
                self.valid_classes.append(label_description.trainId)
                self.class_names.append(label_description.name)

        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files_gt[split]:
            raise Exception("No files for split=[%s] found in %s" %
                            (split, self.root))

        print("Found %d %s images" % (len(self.files_gt[split]), split))

    def __len__(self):
        """Length of dataset"""
        return len(self.files_gt[self.split])

    def __getitem__(self, index):
        """Get an item from dataset"""

        lbl_path = self.files_gt[self.split][index].rstrip()
        img_path = lbl_path.replace("/label/",
                                    "/camera/").replace("_label_", "_camera_")

        _img = Image.open(img_path).convert("RGB")

        # convert label from RGB to "L" mode
        l_rgb = np.array(Image.open(lbl_path), dtype=np.uint8)
        l_id = np.zeros(l_rgb[:, :, 0].shape, dtype=np.uint8)
        for label in Audi_Dataset_Labels:
            mask = ((l_rgb[:, :, 0] == label.color[0])
                    & (l_rgb[:, :, 1] == label.color[1])
                    & (l_rgb[:, :, 2] == label.color[2]))
            l_id[mask] = label.trainId
        l_id = self.encode_segmap(l_id)
        _target = Image.fromarray(l_id)
        _sample = {"image": _img, "label": _target}
        if self.split == "train":
            try:
                if self.params["only_test_data_pertubations"] == True:
                    return self.transform_ts(_sample)
                else:
                    return self.transform_tr(_sample)
            except KeyError:
                return self.transform_tr(_sample)
            
        if self.split == "val":
            return self.transform_val(_sample)
        if self.split == "test":
            return self.transform_ts(_sample)

        return None, None

    def encode_segmap(self, mask):
        """Encode segmentation map"""
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]

        return mask

    def transform_tr(self, _sample):
        """Transform train"""
        composed_transforms = transforms.Compose([

            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(
                base_size=self.params["base_size"],
                crop_size=self.params["crop_size"],
                fill=255,
            ),

            tr.RandomGaussianBlur(),
            tr.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),

        ])
        return composed_transforms(_sample)

    def transform_val(self, _sample):
        """Transform validation"""
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.params["crop_size"]),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
        ])
        return composed_transforms(_sample)

    def transform_ts(self, _sample):
        """Transform test"""
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.params["crop_size"]),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor(),
        ])
        return composed_transforms(_sample)
