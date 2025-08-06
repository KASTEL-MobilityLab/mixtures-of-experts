import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
import yaml
import wandb

from dataloader.a2d2_loader import get_dataloader
from models.frrn import FRRNet
from models.frrn_ensemble import Ensemble
from models.frrn_moe import MoE
from utils.metrics import Evaluator
from utils.train_utils import cross_entropy2d, ensure_dir, colorize, load_my_state_dict, save_stack_img, prepare_image, calculate_and_return_segmentation_mask
from utils.saver import Saver

class Validator():
    """Define model validator"""
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    def __init__(self, _config):
        self.params = params = _config
        params["gpu_ids"] = [0]
        params["output"] = params["DIR"]
        params["dataset"] = ""
        self.is_cuda = len(params["gpu_ids"]) > 0
        self.device = torch.device('cuda', params["gpu_ids"][0]) \
            if self.is_cuda else torch.device('cpu')
        saver = Saver(params)
        save_path = ensure_dir(os.path.join(saver.directory, 'eval_logs'))
        self.test_loader, self.label_names, self.label_colors = \
            get_dataloader(params, params['test_sets'], 'test')
        
        if 'moe' in params['MODEL']['expert'].lower():
            expert_names = list(s for s in params['DATASET']['dataset'].split(','))
            checkpoints = [os.path.join(*[params['DIR'],
                                          params['MODEL']['arch'] + '_' + expert_name,
                                          'model_best.pth'])
                           for expert_name in expert_names]
            # fully connect layer feature depend on scale factor
            l_feat = int((params['DATASET']['img_height']/16) * (params['DATASET']['img_width']/16))
            self.model = MoE(3, params['DATASET']['num_class'], 
                        l_feat,
                        checkpoints, 
                        params['MODEL']['gate'],
                        params['MODEL']['with_conv'], 
                        len(expert_names))
        elif 'ensemble' in params['MODEL']['expert'].lower():
            expert_names = list(s for s in params['DATASET']['dataset'].split(','))
            checkpoints = [os.path.join(*[params['DIR'],
                                          params['MODEL']['arch'] + '_' + expert_name,
                                          'model_best.pth'])
                           for expert_name in expert_names]
            # fully connect layer feature depend on scale factor
            self.model = Ensemble(3, 
                             params['DATASET']['num_class'], 
                             checkpoints, 
                             params['MODEL']['ensemble_type'])
        else:
            self.model = FRRNet(out_channels=params['DATASET']['num_class'])
            
        self.evaluator = Evaluator(params['DATASET']['num_class'])
        if self.is_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=params["gpu_ids"])
            self.model = self.model.to(self.device)
        if params['MODEL']['expert'] == "moe":
            saved_ckpt_path = os.path.join(*[params['DIR'], params['MODEL']['arch']+'_'+
                                         params['MODEL']['expert'] +'_'+ 
                                         #params['MODEL']['layer']+'_'+
                                         params['MODEL']['gate']
                                         ,params['VAL']['checkpoint']])
        else:
            saved_ckpt_path = os.path.join(*[params['DIR'], params['MODEL']['arch']+'_'+
                                         params['MODEL']['expert'], params['VAL']['checkpoint']])
        if not params['MODEL']['expert'] == "ensemble":
            assert os.path.exists(saved_ckpt_path), '{} not exit!'.format(saved_ckpt_path)
            print(saved_ckpt_path)
            new_state_dict = torch.load(saved_ckpt_path)
            self.model = load_my_state_dict(self.model.module, new_state_dict['state_dict'])

    def validate(self):
        """validate model"""
        # pylint: disable-msg=too-many-locals
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        self.loss = cross_entropy2d
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\n Universal PGD-10 attack with epsilon=", params["epsilon"], "PGD steps", params["pgd_steps"], "LR", params["TRAIN"]["lr"])
        
        if params["MODEL"]["expert"] == "moe":
            log_dir = os.path.join(params["output_dir"], params["start_time"] + "_" + "frrn_" + params["MODEL"]["expert"] + "_" + params["MODEL"]["gate"] + "_" + str(params["MODEL"]["with_conv"]) + "_dataset_" + params["train_dataset"] + "_pdg_epsilon_" + str(params["epsilon"]) + "_steps_" + str(params["pgd_steps"]) + "/")
        else:
            log_dir = os.path.join(params["output_dir"], params["start_time"] + "_" + "frrn_" + params["MODEL"]["expert"] + "_" + "dataset_" + params["train_dataset"] + "_pdg_epsilon_" + str(params["epsilon"]) + "_steps_" + str(params["pgd_steps"]) + "/")
        print(log_dir)
        
        os.mkdir(log_dir) ##directory for run
        os.mkdir(log_dir + "original_image/") ##directory for original images
        os.mkdir(log_dir + "original_segmentation/") ##directory for original segmentation
        os.mkdir(log_dir + "delta/") ##directory for delta
        os.mkdir(log_dir + "adv_image/") ##directory for adversarial image
        os.mkdir(log_dir + "adv_segmentation/") ##directory for adversarial segmentation

        writer = SummaryWriter(log_dir)
        print("Prining logs to ", log_dir)
        
        delta = torch.empty(3, params["DATASET"]["img_height"], params["DATASET"]["img_width"]).uniform_(-params["epsilon"], params["epsilon"]).to(device)
        optimizer = torch.optim.Adam([delta], lr=params["TRAIN"]["lr"])
        delta.data = torch.clamp(delta.data, min=-params["epsilon"], max=params["epsilon"]) 
        delta.requires_grad = True
        
        for epoch_from_zero in tqdm(range(params["TRAIN"]["num_epochs"])):
            epoch = epoch_from_zero + 1 
            batch_loss = []

            for batch_idx, sample in enumerate(tbar):
                image, label, label_path = sample
                image = image.to(self.device)
                image.requires_grad = True
                label = label.to(self.device)



                adv_img = image + delta
                adv_img.data = torch.clamp(adv_img, min=0, max=1)

                for i in range (params["pgd_steps"]):
                    delta.requires_grad = True
                    output = self.model(adv_img)

                    loss = self.loss(output, label)
                    wandb.log({"Batch loss": loss.cpu().detach().numpy()})
                    optimizer.zero_grad()
                    loss.backward()

                    delta.grad = -torch.sign(delta.grad)
                    optimizer.step()

                    delta.data = torch.clamp(delta.data, min=-params["epsilon"], max=params["epsilon"]) 
                    adv_img = image + delta
                    adv_img.data = torch.clamp(adv_img, min=0, max=1)

                label = label.cpu().numpy()
                pred = output.data.cpu().numpy()
                del output
                pred = np.argmax(pred, axis=1)
                # Add batch sample into evaluator
                self.evaluator.add_batch(label, pred)

            if self.params['VAL']['visualize']:
                for i in range(output.shape[0]):
                    # Val batch size is 1.
                    pred = output[i].data.max(0)[1].cpu().numpy()
                    ground_true = label[i]
                    pred_colors = colorize(self.label_colors, pred)
                    gt_colors = colorize(self.label_colors, ground_true)
                    # save wrt. label path and weight path
                    _, city, label_name = label_path[i].split('/')[-3:]
                    pred_save_path = os.path.join(
                        os.path.dirname(self.params['VAL']['checkpoint']), 'pred_results',
                        city, label_name)
                    if not os.path.exists(os.path.dirname(pred_save_path)):
                        os.makedirs(os.path.dirname(pred_save_path))
                    save_stack_img([gt_colors, pred_colors], pred_save_path)

            wandb.log({"Epoch loss": loss.cpu().detach().numpy(), "epoch": epoch})
            if (epoch % 1 == 0):
                # Images to save
                image_prep = torch.clone(image)
                original_segmask = calculate_and_return_segmentation_mask(image, self.model, self.label_colors)
                grid_image = torchvision.utils.make_grid(image_prep, nrow=params["TRAIN"]["batch_size"], normalize=False)
                grid_original_seg = torchvision.utils.make_grid(original_segmask, nrow=params["TRAIN"]["batch_size"], normalize=False)

                images_wandb = wandb.Image(grid_image, caption="Top: image_1, Bottom: image_2")
                wandb.log({"original image": images_wandb})
                images_wandb = wandb.Image(grid_original_seg, caption="Top: image_1, Bottom: image_2")
                wandb.log({"original segmentation": images_wandb})

                adv_img_prep = prepare_image(adv_img)
                adv_segmask = calculate_and_return_segmentation_mask(adv_img, self.model, self.label_colors)
                grid_adv_image = torchvision.utils.make_grid(adv_img_prep, nrow=params["TRAIN"]["batch_size"], normalize=True)
                grid_adv_seg = torchvision.utils.make_grid(adv_segmask, nrow=params["TRAIN"]["batch_size"], normalize=True)

                images_wandb = wandb.Image(grid_adv_image, caption="Top: image_1, Bottom: image_2")
                wandb.log({"epoch": epoch, "adv image": images_wandb})
                images_wandb = wandb.Image(grid_adv_seg, caption="Top: image_1, Bottom: image_2")
                wandb.log({"epoch": epoch, "adv segmentation": images_wandb})

                ## save delta as tensor
                torch.save(delta, os.path.join(log_dir, "delta", "delta_" + str(epoch) + ".pt"))
                delta_as_img = delta.detach().cpu().numpy()
                delta_as_img = delta_as_img / params["epsilon"]
                delta_as_img = (delta_as_img + 1) / 2
                delta_as_img = np.transpose(delta_as_img, (1, 2, 0))
                delta_as_img = wandb.Image(delta_as_img, caption="Adversarial noise")
                wandb.log({"Delta as image": delta_as_img})

        # Fast test during the training
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        miou, log_miou_cls = self.evaluator.mean_intersection_over_union(self.label_names)
        fwiou = self.evaluator.frequency_weighted_intersection_over_union()
        print("Validation:")
        print("pAcc:{}, mAcc:{}, m_iou:{}, fwIoU: {}".format(acc, acc_class, miou, fwiou))

        # Mark the run as finished
        wandb.finish()        
        writer.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
            'e.g: params/params_moe.py')
    else:
        print('STARTING EVALUATION WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                torch.autograd.set_detect_anomaly(True)

                wandb.init(project="adv_attacks_on_moe",reinit=True)
                ## load config file into wandb
                wandb.config = yaml.safe_load(stream)
                ## initialize params from loaded config file
                params = wandb.config
                params["start_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if params["MODEL"]["expert"] == "moe":
                    params["checkname"] = str(params["output_dir"] + params["start_time"] + "_frrn_"  + params["MODEL"]["expert"] + "_" + params["MODEL"]["gate"] + "_" + params["train_dataset"] + "_pdg_epsilon_" + str(params["epsilon"]) + "_steps_" + str(params["pgd_steps"]))
                else:
                    params["checkname"] = str(params["output_dir"] + params["start_time"] + "_frrn_"  + params["MODEL"]["expert"] +  "_" + params["train_dataset"] + "_pdg_epsilon_" + str(params["epsilon"]) + "_steps_" + str(params["pgd_steps"]))

                wandb.run.name = params["checkname"]
                wandb.run.save()
                
                validator = Validator(params)
                validator.validate()
            except yaml.YAMLError as exc:
                print(exc)

