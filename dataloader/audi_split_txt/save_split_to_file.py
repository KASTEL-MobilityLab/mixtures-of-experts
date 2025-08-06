"""
Training reproducing:
save dataset split to text files
road type:
  highway_train_gt.txt with "highway/train_val/xx_label_frontcenter_xx.png"
  highway_val_gt.txt
  highway_test_gt.txt
  urban_train_gt.txt
  urban_val_gt.txt
  urban_test_gt.txt
  ambiguous_test_gt.txt
sky drivable:
  high_train_gt.txt
"""

import os
import random


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


def write_to_text_file(img_files, txt_name):
    """Write image to text file"""
    with open(txt_name, "w") as opened_file:
        for file in img_files:
            # relative path "highway/train_val/xx_label_frontcenter_xx.png"
            file = file.split(BASE)[-1]
            opened_file.write(file + "\n")


def save_split_road(subsets):
    """Save split road to text file"""
    for subset in subsets:
        if subset == "ambiguous":
            base_path = os.path.join(*[BASE, subset, "test", "GT"])
            img_files = recursive_glob(base_path, ".png")
            txt_name = "_".join([subset, "test", "gt"]) + ".txt"
            write_to_text_file(img_files, txt_name)
        else:
            base_path = os.path.join(*[BASE, subset, "test", "GT"])
            img_files = recursive_glob(base_path, ".png")
            txt_name = "_".join([subset, "test", "gt"]) + ".txt"
            write_to_text_file(img_files, txt_name)
            # random split train_val
            base_path = os.path.join(*[BASE, subset, "train_val", "GT"])
            img_files = recursive_glob(base_path, ".png")
            random.seed(1)
            random.shuffle(img_files)
            train_img_files = img_files[:6132]
            val_img_files = img_files[6132:6132 + 876]
            train_txt_name = "_".join([subset, "train", "gt"]) + ".txt"
            val_txt_name = "_".join([subset, "val", "gt"]) + ".txt"
            write_to_text_file(train_img_files, train_txt_name)
            write_to_text_file(val_img_files, val_txt_name)


if __name__ == "__main__":
    BASE = "."
    subsets_road = ["highway", "urban", "ambiguous"]
    # subsets_sky = ['max', 'min', 'mid']
    # splits = ['train_val', 'test']

    save_split_road(subsets_road)
