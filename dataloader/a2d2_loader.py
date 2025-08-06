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
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils import data

from dataloader.a2d2_labels import labels_aev_38_unique
from dataloader.transforms import Compose, FixedResize, Normalize, ToTensor, RandomHorizontallyFlip


def get_dataloader(_C, subsets, stage='train'):
    """get dataloader according to subset list"""
    # pylint: disable-msg=too-many-locals
    kwargs = {'num_workers': _C['SYSTEM']['num_workers'], 'pin_memory': _C['SYSTEM']['pin_memory']}
    img_size = (_C['DATASET']['img_height'], _C['DATASET']['img_width'])
    label_names, label_colors = {}, {}
    if stage == 'train':
        train_set_list, val_set_list = [], []
        for sub in subsets:
            print("Loading train ", sub)
            root = os.path.join(_C['DATASET']['root_dataset'], sub)
            train_sub = a2d2Loader(root, split='train', img_size=img_size)
            val_sub = a2d2Loader(root, split='val', img_size=img_size)
            train_set_list.append(train_sub)
            val_set_list.append(val_sub)
            label_names = val_sub.label_names
            label_colors = val_sub.label_colours
        train_set = ConcatDataset(train_set_list)
        val_set = ConcatDataset(val_set_list)
        train_loader = DataLoader(train_set, batch_size=_C['TRAIN']['batch_size'],
                                  shuffle=False, **kwargs)
        val_loader = DataLoader(val_set, batch_size=_C['VAL']['batch_size'],
                                shuffle=False, **kwargs)
        return train_loader, val_loader, label_names, label_colors
    else:
        test_set_list = []
        for sub in subsets:
            print("Loading ", sub)
            root = os.path.join(_C['DATASET']['root_dataset'], sub)
            test_sub = a2d2Loader(root, split='test', img_size=img_size)
            test_set_list.append(test_sub)
            label_names = test_sub.label_names
            label_colors = test_sub.label_colours
        test_set = ConcatDataset(test_set_list)
        test_loader = DataLoader(test_set, batch_size=_C['TEST']['batch_size'],
                                 shuffle=False, **kwargs)
        return test_loader, label_names, label_colors

    
    
def path_parsing(root, txt_file):
    """return absolute path of image file stored in txt"""
    files = []
    with open(txt_file, 'r') as f:
        for line in f:
            name = line.strip().split('/')[-1]
            seq, label_camera, _, _ = name.split('_')
            seq = seq[:8]+'_'+seq[8:]
            filepath = os.path.join(*[root, seq, label_camera, 'cam_front_center', name])
            files.append(filepath)
    return files


class a2d2Loader(data.Dataset):
    colors = [lb.color for lb in labels_aev_38_unique]
    class_names = [lb.name for lb in labels_aev_38_unique]

    label_colours = dict(zip(range(38), colors))
    label_names = dict(zip(range(38), class_names))

    def __init__(self, root, split="train", img_size=(480, 640), img_norm=True):
        self.root = os.path.dirname(root)
        self.subset = os.path.basename(root)
        self.split = split
        self.img_norm = img_norm
        self.n_classes = 38
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files_gt = {}
        txt_name = './dataloader/audi_split_txt/' + '_'.join([self.subset, split]) + '_gt.txt'
        # self.files_gt[split] = [os.path.join(self.root, i_id.strip()) for i_id in open(txt_name)]
        self.files_gt[split] = path_parsing(self.root, txt_name)

        # self.void_classes = [36, 37]
        # self.valid_classes = list(range(36))

        self.ignore_index = 250
        # self.class_map = {m[0]: m[1] for m in map_38_to_11}

        if not self.files_gt[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, root))

        print("Found %d %s images" % (len(self.files_gt[split]), split))

    def __len__(self):
        return len(self.files_gt[self.split])

    def __getitem__(self, index):
        lbl_path = self.files_gt[self.split][index].rstrip()
        # img_path = lbl_path.replace("GT", "Raw").replace("_label_", "_camera_")
        img_path = lbl_path.replace("/label/", "/camera/").replace("_label_", "_camera_")

        img = Image.open(img_path)

        # convert gt image from RGB to L mode
        lbl = Image.open(lbl_path)
        lbl = self.encode_segmap(lbl)

        img, lbl = self.transform(img, lbl)
        if self.split == 'test':
            return img, lbl, lbl_path
        else:
            return img, lbl

    def transform(self, img, lbl):
        """Transform combination"""
        if self.split == 'train':
            # Resize, Augment, Normalize, ToTensor
            composed_transform = Compose([
                FixedResize(size=self.img_size),
                RandomHorizontallyFlip(),
                Normalize(),
                ToTensor()
            ])
        else:
            composed_transform = Compose([
                FixedResize(size=self.img_size),
                Normalize(),
                ToTensor()
            ])
        return composed_transform(img, lbl)

    def decode_segmap(self, temp):
        """Decode from label to color"""
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, l_rgb):
        """Convert RGB to L mode; relabel 36 to 11 cls"""
        if isinstance(l_rgb, Image.Image):
            l_rgb = np.array(l_rgb, dtype=np.uint8)
        l_id = np.zeros(l_rgb[:, :, 0].shape, dtype=np.uint8)
        # check unique color in lbl:
        # labels on a2d2 subset(highway) already set to 11 classes, else use labels_aev_38
        # print(set(tuple(v) for m2d in l_rgb for v in m2d))
        for label in labels_aev_38_unique:
            # masked array to map the corresponding RGB color to the label ID
            mask = ((l_rgb[:, :, 0] == label.color[0]) & (l_rgb[:, :, 1] == label.color[1])
                    & (l_rgb[:, :, 2] == label.color[2]))
            l_id[mask] = label.id
        # print(set(v for m2d in l_id for v in m2d))
        # relabel 36 to 11 cls
        # for _voidc in self.void_classes:
        #     l_id[l_id == _voidc] = self.ignore_index
        # for _validc in self.valid_classes:
        #     l_id[l_id == _validc] = self.class_map[_validc]
        l_id = Image.fromarray(l_id, mode="L")
        return l_id


if __name__ == "__main__":
    local_path = ""
    dst = a2d2Loader(local_path)
    bs = 1
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        print(imgs.shape)
