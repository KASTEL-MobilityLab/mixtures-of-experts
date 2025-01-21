"""
Define some tools to processing data or folder operation.
"""


from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image

from dataloader.datasets.audi import get_audi_labels

def ensure_dir(path):
    """create path"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('EarlyStopping!')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def inplace_relu(m):
    """use inplace relu for efficient training."""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        
def cross_entropy2d(inp, target, weight=None, size_average=True):
    """loss function"""
    _, _, height, width = inp.size()
    _, height_t, width_t = target.size()

    if height != height_t and width != width_t:  # upsample labels
        inp = F.interpolate(inp, size=(height_t, width_t),
                            mode="bilinear", align_corners=False)

    loss = F.cross_entropy(
        inp, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def colorize(label_colors, pred):
    """Decode from label to color"""
    red = np.zeros_like(pred, dtype=np.uint8)
    green = np.zeros_like(pred, dtype=np.uint8)
    blue = np.zeros_like(pred, dtype=np.uint8)
#     for i, label_color in enumerate(label_colors):
    for i, label_color in label_colors.items():
        red[pred == i] = label_color[0]
        green[pred == i] = label_color[1]
        blue[pred == i] = label_color[2]

    rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype='uint8')
    rgb[:, :, 0] = red
    rgb[:, :, 1] = green
    rgb[:, :, 2] = blue
    # rgb = np.transpose(rgb, (2, 0, 1))
    return Image.fromarray(rgb, 'RGB')

def load_my_state_dict(model, state_dict):
    """custom function to load model when not all dict elements"""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('{} not in model_state'.format(name))
            continue
        else:
            own_state[name].copy_(param)

    return model


def save_stack_img(imgs, pred_save_path):
    """Stack image vertically and save"""
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    store_img = np.vstack([np.asarray(i.resize(min_shape)) for i in imgs])
    store_img = Image.fromarray(store_img)
    store_img.save(pred_save_path)
    
def prepare_image(image_given):
    """Transform images of a batch by return normalization and change dimensions to save images.
    """

    image = torch.clone(image_given)
    img = image.detach().cpu().numpy()
    
    for i,item in enumerate(img):
        item = np.transpose(item, (1, 2, 0))  
        #item *= (0.229, 0.224, 0.225)
        #item += (0.485, 0.456, 0.406)
        item = np.uint8(item*255)
        img[i] = np.transpose(item, (2, 0, 1))     

    return torch.from_numpy(img)


def calculate_and_return_segmentation_mask(image, model, label_colors):
    """Calculate the labels and return segmentation mask.
    """
    output = model(image).data.cpu().numpy()
    pred = np.argmax(output, axis=1).astype(np.uint8)
    red = np.zeros_like(pred, dtype=np.uint8)
    green = np.zeros_like(pred, dtype=np.uint8)
    blue = np.zeros_like(pred, dtype=np.uint8)
    for i, label_color in label_colors.items():
        red[pred == i] = label_color[0]
        green[pred == i] = label_color[1]
        blue[pred == i] = label_color[2]
    # adding another dimension for 3 rgb colors
    rgb = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 3), dtype='uint8')
    rgb[:, :, :, 0] = red
    rgb[:, :, :, 1] = green
    rgb[:, :, :, 2] = blue
    rgb = np.transpose(rgb, (0, 3, 1, 2))
    return torch.Tensor(rgb)


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 55
    label_colours = get_audi_labels()

    red = label_mask.copy()
    green = label_mask.copy()
    blue = label_mask.copy()
    for idx in range(0, n_classes):
        red[label_mask == idx] = label_colours[idx, 0]
        green[label_mask == idx] = label_colours[idx, 1]
        blue[label_mask == idx] = label_colours[idx, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = red / 255.0
    rgb[:, :, 1] = green / 255.0
    rgb[:, :, 2] = blue / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
        return rgb
    return rgb

def prepare_image(image_given):
    """Transform images of a batch by return normalization and change dimensions to save images.
    """

    image = torch.clone(image_given)
    img = image.detach().cpu().numpy()

    for i, item in enumerate(img):
        item = np.transpose(item, (1, 2, 0))  ##anderes Transform
        item *= (0.229, 0.224, 0.225)
        item += (0.485, 0.456, 0.406)
        item = np.uint8(item * 255)
        img[i] = np.transpose(item, (2, 0, 1))

    return torch.from_numpy(img)

def calculate_and_return_segmentation_mask(image, model):
    """Calculate the labels and return segmentation mask.
    """
    output = model(image).data.cpu().numpy()
    y1_argmax = np.argmax(output, axis=1).astype(np.uint8)
    segmap = np.empty_like(image.detach().cpu().numpy())

    return convert_to_img_and_return(y1_argmax, segmap)


def calculate_and_save_segmentation_mask(image, model, epoch, image_name, log_dir):
    """Calculate the labels.
    """
    output = model(image).data.cpu().numpy()
    y1_argmax = np.argmax(output, axis=1).astype(np.uint8)

    convert_to_img_and_save(y1_argmax, epoch, image_name, log_dir)
    
    
def save_image(image, epoch, image_name, log_dir):
    """Save images of a batch.
    """
    image_to_save = prepare_image(image)
    image_to_save = image_to_save.detach().cpu().numpy()
    for idx, item in enumerate(image_to_save):
        item = np.transpose(item, (1, 2, 0)).astype(np.uint8)
        plt.imsave(log_dir + str(image_name) + "/" + str(image_name) + "_" + str(epoch) + "_" + str(idx) + ".png", item)
        
def convert_to_img_and_save(pred, epoch, image_name, log_dir):
    """Decode segmentation mask and save the segmentation mask.
    """
    for idx, item in enumerate(pred):
        segmap = decode_segmap(item, dataset="audi_split")

        plt.imsave(log_dir + str(image_name) + "/" + str(image_name) + "_" + str(epoch) + "_" + str(idx) + ".png",
                   segmap)
        
def convert_to_img_and_return(pred, segmap):
    """Decode segmentation mask and save the segmentation mask.
    """
    for idx, item in enumerate(pred):
        segmap[idx] = np.transpose(decode_segmap(item, dataset="audi_split"), (2, 0, 1))

    return torch.from_numpy(segmap)


