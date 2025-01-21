"""
Given yaml config file for model training
"""
from datetime import datetime
import os
from PIL import Image
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import sys
import wandb
import yaml

from dataloader.a2d2_loader import get_dataloader
from models.frrn import FRRNet
from models.frrn_ensemble import Ensemble
from models.frrn_moe import MoE
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.train_utils import cross_entropy2d, EarlyStopping, colorize

class Trainer():
    """Define model trainer"""
    def __init__(self, _config):
        self.params = params = _config
        params["gpu_ids"] = [0]
        self.is_cuda = len(params["gpu_ids"]) > 0
        self.device = torch.device('cuda', params["gpu_ids"][0]) \
            if self.is_cuda else torch.device('cpu')
        self.saver = Saver(params)
        log_dir = os.path.join(params["output"], params["start_time"] + "_" + params["checkname"] + "/")
        
        self.writer = SummaryWriter(log_dir)
        
        # Datasets
        self.datasets = list(s for s in params["DATASET"]["dataset"].split(','))
        self.train_loader, self.val_loader, self.label_names, self.label_colors \
            = get_dataloader(params, self.datasets)
        
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
            trainable_parameters = model._get_train_params()
        else:
            self.model = FRRNet(out_channels=params['DATASET']['num_class'])
            trainable_parameters = model.parameters()
        
        self.optimizer = torch.optim.Adam(
            trainable_parameters, lr=params["TRAIN"]["lr"], betas=(0.9, 0.99),
            weight_decay=params["TRAIN"]["weight_decay"])
        self.scheduler = CosineAnnealingLR(
            self.optimizer, params["TRAIN"]["num_epochs"], eta_min=1e-6)
        self.loss = cross_entropy2d

        self.evaluator = Evaluator(params['DATASET']['num_class'])
        if self.is_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=params["gpu_ids"])
            self.model = self.model.to(self.device)
            
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total trainable params", pytorch_total_params)

        self.best_pred = 0.0
        if os.path.isfile(params["TRAIN"]["resume"]):
            checkpoint = torch.load(params["TRAIN"]["resume"])
            params["TRAIN"]["start_epoch"] = checkpoint['epoch']
            if self.is_cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(params["TRAIN"]["resume"],
                                                                checkpoint['epoch']))

    def training(self, epoch):
        """train model iteration from one epoch"""
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, label = sample
            image = image.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.loss(output, label)
            loss.backward()
            self.optimizer.step()
            # adjust learning rate each step
            # self.scheduler.step()
            train_loss += loss.item()
            tbar.set_description('Epoch: {:<3d}, Train Loss: {:.3f}'.format(
                epoch, train_loss / (i + 1)))
            with torch.no_grad():
                wandb.log({"train/total_loss_iter": float(loss.item())})

        with torch.no_grad():
            wandb.log({"train/total_loss_epoch": float(loss.item()), "epoch": epoch})

    def validation(self, epoch):
        """validate model"""
        # pylint: disable-msg=too-many-locals
        test_loss = 0.0
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        num_img_val = len(self.val_loader)
        for i, sample in enumerate(tbar):
            image, label = sample
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                output = self.model(image)
            loss = self.loss(output, label)
            test_loss += loss.item()
            tbar.set_description('Epoch: {:<3d}, Val Loss: {:.3f}'.format(
                epoch, test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            label = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(label, pred)
        
        pred = output[0].data.max(0)[1].cpu().numpy()
        ground_true = label[0]

        wandb.log({"val_img": wandb.Image(image)})
        wandb.log({"val_gt": wandb.Image(colorize(self.label_colors, ground_true))})
        wandb.log({"val_pred": wandb.Image(colorize(self.label_colors, pred))})

        # Fast test during the training
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        miou, log_miou_cls = self.evaluator.mean_intersection_over_union(self.label_names)
        fwiou = self.evaluator.frequency_weighted_intersection_over_union()
        
        self.writer.add_scalar("val/total_loss_epoch", test_loss, epoch)
        self.writer.add_scalar("val/mIoU", miou, epoch)
        self.writer.add_scalar("val/Acc", acc, epoch)
        self.writer.add_scalar("val/Acc_class", acc_class, epoch)
        self.writer.add_scalar("val/fwIoU", fwiou, epoch)
        
        wandb.log({"val/total_loss_epoch": test_loss, "epoch": epoch})
        wandb.log({"val/mIoU": miou, "epoch": epoch})
        wandb.log({"val/Acc": acc, "epoch": epoch})
        wandb.log({"val/Acc_class": acc_class, "epoch": epoch})
        wandb.log({"val/fwIoU": fwiou, "epoch": epoch})
        wandb.log({"val/loss": test_loss, "epoch": epoch})

        new_pred = miou
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        return test_loss/num_img_val


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
            'e.g: params/params_moe.py')
    else:
        print('STARTING EVALUATION WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                
                torch.manual_seed(params["TRAIN"]["seed"])
                torch.cuda.manual_seed(params["TRAIN"]["seed"])
                np.random.seed(params["TRAIN"]["seed"])
                random.seed(params["TRAIN"]["seed"])

                # Setup device
                params["gpu_ids"] = [0]
                params["output"] = params["DIR"]
                params["dataset"] = ""
                params["checkname"] = params['MODEL']['arch']+'_'+params['MODEL']['expert']+ params['MODEL']['gate'] + str(params["DATASET"]["img_height"]) + "_" + str(params["DATASET"]["img_width"]) 
                
                params["start_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                ## initialize wandb
                wandb.init(project="adv_attacks_on_moe", reinit=True)
                wandb.config = params
                wandb.run.name = params["start_time"] + "_" + params["checkname"]
                wandb.run.save()

        
                trainer = Trainer(params)
                print('Starting Epoch: {}'.format(str(params["TRAIN"]["start_epoch"])))
                print('Total Epoches: {}'.format(str(params["TRAIN"]["num_epochs"])))
                early_stopping = EarlyStopping(params["TRAIN"]["patience"])
                for epoch in range(params["TRAIN"]["start_epoch"], params["TRAIN"]["num_epochs"]):
                    trainer.training(epoch)
                    trainer.scheduler.step(epoch)
                    if epoch % params["TRAIN"]["eval_interval"] == (params["TRAIN"]["eval_interval"] - 1):
                        test_loss = trainer.validation(epoch)
                        early_stopping(test_loss)
                        if early_stopping.early_stop:
                            break
            except yaml.YAMLError as exc:
                print(exc)
