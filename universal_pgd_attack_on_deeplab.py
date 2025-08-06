## add system paths
import sys
import yaml
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
from PIL import Image
from datetime import datetime
import sys
from tqdm import tqdm
import torch
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter

from dataloader.a2d2_loader import get_dataloader
from models.deeplab_ensemble import Ensemble
from models.deeplab_modeling import _load_model
from models.deeplab_moe import MoE
from utils.metrics import Evaluator
from utils.train_utils import calculate_and_save_segmentation_mask, calculate_and_return_segmentation_mask, colorize, prepare_image, save_image

import wandb


def run_attack(params):
    params = params
    is_cuda = len(params["gpu_ids"]) > 0
    device = torch.device('cuda', params["gpu_ids"][0]) \
        if is_cuda else torch.device('cpu')

    nclass = params['DATASET']['num_class']
    datasets = list(s for s in params["DATASET"]["dataset"].split(','))
    train_loader, val_loader, label_names, label_colors = get_dataloader(params, datasets)

    evaluator = Evaluator(nclass)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    if params["MODEL"]["expert"] == "moe":
            linear_feat = (params["DATASET"]["img_height"] // params["MODEL"]["out_stride"] + 1)**2
            linear_feat = 3249
            print("Linear features", linear_feat)
            model = MoE(
                arch=params["MODEL"]["arch"],
                backbone=params["MODEL"]["backbone"],
                output_stride=params["MODEL"]["out_stride"],
                num_classes=nclass,
                linear_features=linear_feat,
                checkpoint1=params["MODEL"]["checkpoint_moe_expert_1"], 
                checkpoint2=params["MODEL"]["checkpoint_moe_expert_2"],
                gate_type=params["MODEL"]["gate"],
                with_conv=params["MODEL"]["with_conv"],
                allow_gradient_flow = True
            )
            
            if params["TEST"]["checkpoint"] is not None:
                if not os.path.isfile(params["TEST"]["checkpoint"]):
                    raise RuntimeError("=> no checkpoint found at '{}'".format(
                        params["TEST"]["checkpoint"]))

                print("Loading checkpoint from", params["TEST"]["checkpoint"])
                checkpoint = torch.load(params["TEST"]["checkpoint"])
                params["start_epoch"] = checkpoint["epoch"]

                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint '{}' (epoch {})".format(
                    params["TEST"]["checkpoint"], checkpoint["epoch"]))
                
                if is_cuda:
                    model = model.to(device)
                    model.expert1.to(device)
                    model.expert2.to(device)

            else:
                raise RuntimeError("=> no checkpoint in input arguments")
                
    elif params["MODEL"]["expert"] == "ensemble":
        model = Ensemble(arch=params["MODEL"]["arch"], 
                              backbone=params["MODEL"]["backbone"], 
                              output_stride=params["MODEL"]["out_stride"],     
                              num_classes=nclass, 
                              checkpoint1=params["TEST"]["checkpoint_moe_expert_1"], 
                              checkpoint2=params["TEST"]["checkpoint_moe_expert_2"], 
                              ens_type=params["MODEL"]["ens_type"]
                             )
        model.expert1.to(device)
        model.expert2.to(device)
    else:
        model = _load_model(params["MODEL"]["arch"], params["MODEL"]["backbone"], nclass, output_stride=params["MODEL"]["out_stride"], pretrained_backbone=True, input_channels=3)

        if params["TEST"]["checkpoint"] is not None:
            if not os.path.isfile(params["TEST"]["checkpoint"]):
                raise RuntimeError("=> no checkpoint found at '{}'".format(
                    params["TEST"]["checkpoint"]))

            print("Loading checkpoint from", params["TEST"]["checkpoint"])
            checkpoint = torch.load(params["TEST"]["checkpoint"])
            params["start_epoch"] = checkpoint["epoch"]

            model.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                params["TEST"]["checkpoint"], checkpoint["epoch"]))

            if is_cuda:
                model = torch.nn.DataParallel(model, device_ids=params["gpu_ids"])
                model = model.to(device)

        else:
            raise RuntimeError("=> no checkpoint in input arguments")

    log_dir = params["checkname"]
    writer = SummaryWriter(log_dir)
    print("Prining logs to ", log_dir)
    
    os.mkdir(log_dir + "/original_image/") ##directory for original images
    os.mkdir(log_dir + "/original_segmentation/") ##directory for original segmentation
    os.mkdir(log_dir + "/delta/") ##directory for delta
    os.mkdir(log_dir + "/adv_image/") ##directory for adversarial image
    os.mkdir(log_dir + "/adv_segmentation/") ##directory for adversarial segmentation
    
    tbar = tqdm(train_loader)
    num_img_tr = len(train_loader)
    test_loss = 0.0
    
    ## initialized delta
    delta = torch.empty(3, params["DATASET"]["img_height"], params["DATASET"]["img_width"]).uniform_(-params["ATTACK"]["epsilon"], params["ATTACK"]["epsilon"]).to(device)

    delta.data = torch.clamp(delta.data, min=-params["ATTACK"]["epsilon"], max=params["ATTACK"]["epsilon"]) 
    optimizer = optim.Adam([delta], lr=params["ATTACK"]["lr"])

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    model.eval()
    
    epoch_loss = []
    all_batches_loss = []
    
    for epoch_from_zero in tqdm(range(params["ATTACK"]["num_epochs"])):
        epoch = epoch_from_zero + 1 
        batch_loss = []
       
        for batch_idx, sample in enumerate(tbar):
            image, target = sample
            image = image.type(torch.float32)
            target = target.type(torch.long)
            if torch.cuda.is_available():
                image, target = image.cuda(), target.cuda()

            delta.requires_grad = True
            
            adv_img = image + delta
            
            for i in range (params["ATTACK"]["pgd_steps"]):
                delta.requires_grad = True
                output = model(adv_img)

                loss = criterion(output, target)
                
                wandb.log({"Batch loss": loss.cpu().detach().numpy()})
                
                optimizer.zero_grad()
                test_loss += loss.item()
                tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))

                loss.backward()

                delta.grad = -torch.sign(delta.grad)
                optimizer.step()

                delta.data = torch.clamp(delta.data, min=-params["ATTACK"]["epsilon"], max=params["ATTACK"]["epsilon"]) 
                adv_img = image + delta

        epoch_loss.append(np.mean(batch_loss))

        wandb.log({"Epoch loss": loss.cpu().detach().numpy(), "epoch": epoch})
        
        if epoch % params["ATTACK"]["eval_interval"] == (params["ATTACK"]["eval_interval"] - 1):
            ## Tensorboard
            image_prep = torch.clone(image)
            original_segmask = calculate_and_return_segmentation_mask(image, model)
            grid_image = torchvision.utils.make_grid(image_prep, nrow=params["TRAIN"]["batch_size"], normalize=True)
            grid_original_seg = torchvision.utils.make_grid(original_segmask, nrow=params["TRAIN"]["batch_size"], normalize=True)
            writer.add_image("original image", grid_image, epoch)
            writer.add_image("original segmentation", grid_original_seg, epoch)

            images_wandb = wandb.Image(grid_image, caption="Top: image_1, Bottom: image_2")
            wandb.log({"original image": images_wandb})
            images_wandb = wandb.Image(grid_original_seg, caption="Top: image_1, Bottom: image_2")
            wandb.log({"original segmentation": images_wandb})
            
            writer.add_image('delta', delta, epoch)
            
            adv_img_prep = prepare_image(adv_img)
            adv_segmask = calculate_and_return_segmentation_mask(adv_img, model)
            grid_adv_image = torchvision.utils.make_grid(adv_img_prep, nrow=params["TRAIN"]["batch_size"], normalize=True)
            grid_adv_seg = torchvision.utils.make_grid(adv_segmask, nrow=params["TRAIN"]["batch_size"], normalize=True)

            writer.add_image("adv image", grid_adv_image, epoch)
            writer.add_image("adv segmentation", grid_adv_seg, epoch)

            images_wandb = wandb.Image(grid_adv_image, caption="Top: image_1, Bottom: image_2")
            wandb.log({"adv image": images_wandb})
            images_wandb = wandb.Image(grid_adv_seg, caption="Top: image_1, Bottom: image_2")
            wandb.log({"adv segmentation": images_wandb})

            ##Save original Images
            save_image(image, epoch, "original_image", log_dir+"/")

            ## Save segmentation result on original image
            calculate_and_save_segmentation_mask(image, model, epoch, "original_segmentation", log_dir+"/")
            
            ## Save adversarial image
            save_image(adv_img, epoch, "adv_image", log_dir+"/") ##save adversarial image
            ## Save segmentation result on adversarial image
            calculate_and_save_segmentation_mask(adv_img, model, epoch, "adv_segmentation", log_dir+"/") ##save adversarial segmentation  

            ## save delta
            ## save delta as tensor
            torch.save(delta, os.path.join(log_dir, "delta", "delta_" + str(epoch) + ".pt"))
            ## save delta as image
            delta_to_save = delta.unsqueeze(0)
            # To display small pixel values in file
            # map from [-eps, eps] to [-1, 1]
            delta_to_save = delta_to_save / params["ATTACK"]["epsilon"]
            # map from [-1, 1] to [0,1]
            delta_to_save = (delta_to_save + 1) / 2
            save_image(delta_to_save, epoch, "delta", log_dir+"/") ##save delta 
       
    # Mark the run as finished
    wandb.finish()        
    writer.close()
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
              'e.g: params/params_moe.py')
    else:
        print('STARTING PGD ATTACK WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                params["gpu_ids"] = [0]
                params["start_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                if params["MODEL"]["expert"] == "ensemble":
                    checkpoint_name = "ensemble_" + params["MODEL"]["ens_type"]
                else:
                    checkpoint_name = params["TEST"]["checkpoint"].split("/")[-2]
                print(checkpoint_name)
                params["checkname"] = params["output"] + params["start_time"] + "_attack_on_" + checkpoint_name +  "_pdg_epsilon_" + str(params["ATTACK"]["epsilon"]) + "_steps_" + str(params["ATTACK"]["pgd_steps"])
                torch.manual_seed(1)
                
                ## initialize wandb
                wandb.init(project="adv_attacks_on_moe", reinit=True)
                wandb.config = params
                wandb.run.name = params["checkname"]
                wandb.run.save()
                wandb.run.name = params["checkname"]
                wandb.run.save()

                torch.manual_seed(1)

                run_attack(params)

            except yaml.YAMLError as exc:
                print(exc)

                

    
