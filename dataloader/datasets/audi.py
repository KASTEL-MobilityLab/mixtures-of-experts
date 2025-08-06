# External file copied from deeplabv3plus to avoid circular dependency
# pylint: disable=all
import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloader import custom_transforms as tr
from collections import namedtuple

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

Audi_Dataset_Labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'Car 1'                ,  0 ,       0 , 'void'            , 0       , False        , False        , [255, 0, 0] ),
    Label(  'Car 2'                ,  1 ,       1 , 'void'            , 0       , False        , False        , [200, 0, 0] ),
    Label(  'Car 3'                ,  2 ,       2 , 'void'            , 0       , False        , False        , [150, 0, 0] ),
    Label(  'Car 4'                ,  3 ,       3 , 'void'            , 0       , False        , False        , [128, 0, 0] ),
    Label(  'Bicycle 1'            ,  4 ,       4 , 'void'            , 0       , False        , False        , [182, 89, 6] ),
    Label(  'Bicycle 2'            ,  5 ,       5 , 'void'            , 0       , False        , False        , [150, 50, 4] ),
    Label(  'Bicycle 3'            ,  6 ,       6 , 'void'            , 0       , False        , False        , [90, 30, 1] ),
    Label(  'Bicycle 4'            ,  7 ,       7 , 'void'            , 0       , False        , False        , [90, 30, 30] ),
    Label(  'Pedestrian 1'         ,  8 ,       8 , 'void'            , 0       , False        , False        , [204, 153, 255] ),
    Label(  'Pedestrian 2'         ,  9 ,       9 , 'void'            , 0       , False        , False        , [189, 73, 155] ),
    Label(  'Pedestrian 3'         , 10 ,      10 , 'void'            , 0       , False        , False        , [239, 89, 191] ),
    Label(  'Truck 1'              , 11 ,      11 , 'void'            , 0       , False        , False        , [255, 128, 0] ),
    Label(  'Truck 2'              , 12 ,      12 , 'void'            , 0       , False        , False        , [200, 128, 0] ),
    Label(  'Truck 3'              , 13 ,      13 , 'void'            , 0       , False        , False        , [150, 128, 0] ),
    Label(  'Small vehicles 1'     , 14 ,      14 , 'void'            , 0       , False        , False        , [0, 255, 0] ),
    Label(  'Small vehicles 2'     , 15 ,      15 , 'void'            , 0       , False        , False        , [0, 200, 0] ),
    Label(  'Small vehicles 3'     , 16 ,      16 , 'void'            , 0       , False        , False        , [0, 150, 0] ),
    Label(  'Traffic signal 1'     , 17 ,      17 , 'void'            , 0       , False        , False        , [0, 128, 255] ),
    Label(  'Traffic signal 2'     , 18 ,      18 , 'void'            , 0       , False        , False        , [30, 28, 158] ),
    Label(  'Traffic signal 3'     , 19 ,      19 , 'void'            , 0       , False        , False        , [60, 28, 100] ),
    Label(  'Traffic sign 1'       , 20 ,      20 , 'void'            , 0       , False        , False        , [0, 255, 255] ),
    Label(  'Traffic sign 2'       , 21 ,      21 , 'void'            , 0       , False        , False        , [30, 220, 220] ),
    Label(  'Traffic sign 3'       , 22 ,      22 , 'void'            , 0       , False        , False        , [60, 157, 199] ),
    Label(  'Utility vehicle 1'    , 23 ,      23 , 'void'            , 0       , False        , False        , [255, 255, 0] ),
    Label(  'Utility vehicle 2'    , 24 ,      24 , 'void'            , 0       , False        , False        , [255, 255, 200] ),
    Label(  'Sidebars'             , 25 ,      25 , 'void'            , 0       , False        , False        , [233, 100, 0] ),
    Label(  'Speed bumper'         , 26 ,      26 , 'void'            , 0       , False        , False        , [110, 110, 0] ),
    Label(  'Curbstone'            , 27 ,      27 , 'void'            , 0       , False        , False        , [128, 128, 0] ),
    Label(  'Solid line'           , 28 ,      28 , 'void'            , 0       , False        , False        , [255, 193, 37] ),
    Label(  'Irrelevant signs'     , 29 ,      29 , 'void'            , 0       , False        , False        , [64, 0, 64] ),
    Label(  'Road blocks'          , 30 ,      30 , 'void'            , 0       , False        , False        , [185, 122, 87] ),
    Label(  'Tractor'              , 31 ,      31 , 'void'            , 0       , False        , False        , [0, 0, 100] ),
    Label(  'Non-drivable street'  , 32 ,      32 , 'void'            , 0       , False        , False        , [139, 99, 108] ),
    Label(  'Zebra crossing'       , 33 ,      33 , 'void'            , 0       , False        , False        , [210, 50, 115] ),
    Label(  'Obstacles / trash'    , 34 ,      34 , 'void'            , 0       , False        , False        , [255, 0, 128] ),
    Label(  'Poles'                , 35 ,      35 , 'void'            , 0       , False        , False        , [255, 246, 143] ),
    Label(  'RD restricted area'   , 36 ,      36 , 'void'            , 0       , False        , False        , [150, 0, 150] ),
    Label(  'Animals'              , 37 ,      37 , 'void'            , 0       , False        , False        , [204, 255, 153] ),
    Label(  'Grid structure'       , 38 ,      38 , 'void'            , 0       , False        , False        , [238, 162, 173] ),
    Label(  'Signal corpus'        , 39 ,      39 , 'void'            , 0       , False        , False        , [33, 44, 177] ),
    Label(  'Drivable cobblestone' , 40 ,      40 , 'void'            , 0       , False        , False        , [180, 50, 180] ),
    Label(  'Electronic traffic'   , 41 ,      41 , 'void'            , 0       , False        , False        , [255, 70, 185] ),
    Label(  'Slow drive area'      , 42 ,      42 , 'void'            , 0       , False        , False        , [238, 233, 191] ),
    Label(  'Nature object'        , 43 ,      43 , 'void'            , 0       , False        , False        , [147, 253, 194] ),
    Label(  'Parking area'         , 44 ,      44 , 'void'            , 0       , False        , False        , [150, 150, 200] ),
    Label(  'Sidewalk'             , 45 ,      45 , 'void'            , 0       , False        , False        , [180, 150, 200] ),
    Label(  'Ego car'              , 46 ,      46 , 'void'            , 0       , False        , False        , [72, 209, 204] ),
    Label(  'Painted driv. instr.' , 47 ,      47 , 'void'            , 0       , False        , False        , [200, 125, 210] ),
    Label(  'Traffic guide obj.'   , 48 ,      48 , 'void'            , 0       , False        , False        , [159, 121, 238] ),
    Label(  'Dashed line'          , 49 ,      49 , 'void'            , 0       , False        , False        , [128, 0, 255] ),
    Label(  'RD normal street'     , 50 ,      50 , 'void'            , 0       , False        , False        , [255, 0, 255] ),
    Label(  'Sky'                  , 51 ,      51 , 'void'            , 0       , False        , False        , [135, 206, 255] ),
    Label(  'Buildings'            , 52 ,      52 , 'void'            , 0       , False        , False        , [241, 230, 255] ),
    Label(  'Blurred area'         , 53 ,      53 , 'void'            , 0       , False        , False        , [96, 69, 143] ),
    Label(  'Rain dir'             , 54 ,      54 , 'void'            , 0       , False        , False        , [53, 46, 82] ),
]

_RAW_AUDI_DATASET_SPLITS = {
    "audi_train": ["20180807_145028",  "20180925_135056",  "20181107_132300",  "20181108_091945",  "20181204_154421",
                    "20180810_142822",  "20181008_095521",  "20181107_132730",  "20181108_103155",  "20181204_170238",
                    "20180925_101535",  "20181016_082154",  "20181107_133258",  "20181108_123750",  "20181204_191844",
                    "20180925_112730",  "20181016_095036",  "20181107_133445",  "20181108_141609", 
                    "20180925_124435",  "20181016_125231"],
    "audi_val": ["20181108_084007"],
    "audi_test": ["20181204_135952"]                 
}

def get_audi_labels():
    ignore_index = 255
    valid_classes_color = []
    unique_valid_classes_color = []
    for labelDescription in Audi_Dataset_Labels:
        if not (labelDescription.trainId == ignore_index or labelDescription.trainId < 0):
            valid_classes_color.append(labelDescription.color)
    
    for elem in valid_classes_color:
        if elem not in unique_valid_classes_color:
            unique_valid_classes_color.append(elem)
    return np.array(unique_valid_classes_color)

def get_audi_names():

    valid_colors = get_audi_labels()
    valid_classes_names = []

    for color in valid_colors:        
        for labelDescription in Audi_Dataset_Labels:
            if (color == labelDescription.color).all():
                valid_classes_names.append(labelDescription.name)
                break

    return np.asarray(valid_classes_names)


class AudiSegmentation(data.Dataset):
    NUM_CLASSES = 55

    def __init__(self, args, root="/disk/ml/datasets/aev/camera_lidar_semantic", split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = self.root

        for key, image_dirs in _RAW_AUDI_DATASET_SPLITS.items():
            if split in key:
                self.files[split] = []
                for image_dir in image_dirs:                
                    self.files[split].extend(self.recursive_glob(rootdir=os.path.join(self.images_base,image_dir,'camera'), suffix='.png'))

        self.void_classes = []
        self.class_names = []
        self.valid_classes = []

        self.ignore_index = 255

        for labelDescription in Audi_Dataset_Labels:
            if labelDescription.trainId == self.ignore_index or labelDescription.trainId < 0:
                self.void_classes.append(labelDescription.trainId)  
            else:
                self.valid_classes.append(labelDescription.trainId)
                self.class_names.append(labelDescription.name)
                            
        
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()

        lbl_path = img_path.replace("/camera/", "/label/", 1)
        lbl_path = lbl_path.replace("_camera_", "_labelTrainIds_", 1)

        if not (os.path.isfile(lbl_path)):
            print("Create Training ID (greyscale) label image: " + lbl_path)

            label_file = img_path.replace("/camera/", "/label/", 1)
            label_file = label_file.replace("_camera_", "_label_", 1)

            assert os.path.isfile(
                label_file
            ), "Label file for audi dataset not found."

            # Create label images mapped to training label ids
            labelImg = np.array(Image.open(label_file), dtype="uint8")
            labelImg = labelImg[:,:,:3]

            labelImgIDs = np.array(Image.new("L", (1920, 1208), 0), dtype="uint8")   

            for labelDescription in Audi_Dataset_Labels:
                labelImgIDs[np.where((labelImg==labelDescription.color).all(axis=2))] = labelDescription.trainId
                        
            labelImgIDs = Image.fromarray(labelImgIDs)
            labelImgIDs.save(lbl_path)  

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)

        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloader.dl_tool import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    audi_train = AudiSegmentation(args, split='train')

    dataloader = DataLoader(audi_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='audi')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

