import os
import numpy as np
from tqdm import tqdm
import sys
import yaml
import wandb
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader.a2d2_loader import get_dataloader
from models.deeplab_modeling import _load_model
from models.deeplab_moe import MoE
from utils.metrics import Evaluator
from utils.train_utils import colorize
from utils.saver import Saver

class Trainer:
    # pylint: disable=too-many-instance-attributes
    """Class for training neural network

    Args:
        object ([type]): [description]
    """

    def __init__(self, params):
        self.params = params

        # Define Saver
        self.saver = Saver(params)
        self.writer = SummaryWriter(params["checkname"])
        print("Prining logs to ", params["checkname"])

        self.nclass = params['DATASET']['num_class']
        self.datasets = list(s for s in params["DATASET"]["dataset"].split(','))
        self.train_loader, self.val_loader, self.label_names, self.label_colors = get_dataloader(params, self.datasets)
        
        if params["MODEL"]["expert"] == "moe":
            linear_feat = (params["DATASET"]["img_height"] // params["MODEL"]["out_stride"] + 1)**2
            linear_feat = 3249
            print("Linear features", linear_feat)
            self.model = MoE(
                arch=params["MODEL"]["arch"],
                backbone=params["MODEL"]["backbone"],
                output_stride=params["MODEL"]["out_stride"],
                num_classes=self.nclass,
                linear_features=linear_feat,
                checkpoint1=params["MODEL"]["checkpoint_moe_expert_1"], 
                checkpoint2=params["MODEL"]["checkpoint_moe_expert_2"],
                gate_type=params["MODEL"]["gate"],
                with_conv=params["MODEL"]["with_conv"],
                allow_gradient_flow = False
            )
            
        else:
            self.model = _load_model(params["MODEL"]["arch"], params["MODEL"]["backbone"], self.nclass, output_stride=self.params["MODEL"]["out_stride"], pretrained_backbone=True, input_channels=3)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total trainable params", pytorch_total_params)
        
        self.optimizer = torch.optim.SGD(
            params=list(self.model.parameters()),
            lr=params["TRAIN"]["lr"],
            momentum=params["TRAIN"]["momentum"],
            weight_decay=params["TRAIN"]["weight_decay"],
            nesterov=False
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.params["gpu_ids"]
            )
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if os.path.isfile(params["TRAIN"]["resume"]):
            if not os.path.isfile(params["TRAIN"]["resume"]):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'".format(params["TRAIN"]["resume"])
                )
            checkpoint = torch.load(params["TRAIN"]["resume"])
            params["TRAIN"]["start_epoch"] = checkpoint["epoch"]
            if torch.cuda.is_available():
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if not params["ft"]:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    params["TRAIN"]["resume"], checkpoint["epoch"]
                )
            )

    def training(self, epoch):
        """Train the neural network

        Args:
            epoch (int): number of desired epochs
        """
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        batch_idx = 0
        for batch_idx, sample in enumerate(tbar):     
            image, target = sample
            image = image.type(torch.float32)
            target = target.type(torch.long)

            if torch.cuda.is_available():
                image, target = image.cuda(), target.cuda()
            if batch_idx >= num_img_tr - 1:  # temp fix
                break
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.writer.add_scalar(
                "train/total_loss_iter",
                loss.item(),
                batch_idx + num_img_tr * epoch,
            )
            with torch.no_grad():
                wandb.log({"train/total_loss_iter": float(loss.item())})

        with torch.no_grad():
            wandb.log({"train/total_loss_epoch": float(loss.item()), "epoch": epoch})

        # --- mean the train loss
        train_loss /= num_img_tr
        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print(
            "[Epoch: %d, numImages: %5d]"
            % (epoch, batch_idx * self.params["TRAIN"]["batch_size"] + image.data.shape[0])
        )
        print("Loss: %.3f" % train_loss)

        if self.params["TRAIN"]["no_val"]:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best,
            )

    def validation(self, epoch):
        """Validation step

        Args:
            epoch (int): number of desired epochs
        """
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        num_img_val = len(self.val_loader)
        test_loss = 0.0
        i = 0
        for i, sample in enumerate(tbar):
            image, target = sample
            image = image.type(torch.float32)
            target = target.type(torch.long)
            if torch.cuda.is_available():
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
        # --- mean the test loss
        test_loss /= num_img_val

        for i in range(output.shape[0]):
                    # Val batch size is 1.
            pred = output[i].data.max(0)[1].cpu().numpy()
            target = target[i]
            wandb.log({"val_img": wandb.Image(image)})
            wandb.log({"val_gt": wandb.Image(colorize(self.label_colors, target))})
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
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best,
                "checkpoint_best_pred.pth.tar",
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
              'e.g: params/params_moe.py')
    else:
        print('STARTING TRAINING WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                params["gpu_ids"] = [0]
                params["dataset"] = ""
                params["start_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                params["checkname"] = str(params["output"] + params["start_time"] + "_" + params["MODEL"]["arch"] + "_" + params['MODEL']["expert"] + "_" + params["MODEL"]["backbone"] + "_" + params["DATASET"]["dataset"])
                if params["MODEL"]["expert"] == "moe":
                    with_conv = "with_conv" if params["MODEL"]["with_conv"] else ""
                    params["checkname"] = str(params["output"] + params["start_time"] + "_" + params["MODEL"]["arch"] + "_moe_" + params['MODEL']["gate"] + "_" + with_conv + "_" + params["DATASET"]["dataset"])

                torch.manual_seed(1)
                
                ## initialize wandb
                wandb.init(project="adv_attacks_on_moe", reinit=True)
                wandb.config = params
                wandb.run.name = params["checkname"]
                wandb.run.save()
                
                trainer = Trainer(params)
                print("Starting Epoch:", trainer.params["TRAIN"]["start_epoch"])
                print("Total Epoches:", trainer.params["TRAIN"]["num_epochs"])
                for epoch in range(trainer.params["TRAIN"]["start_epoch"], trainer.params["TRAIN"]["num_epochs"]):
                    trainer.training(epoch)
                    if not trainer.params["TRAIN"]["no_val"] and epoch % params["TRAIN"]["eval_interval"] == (
                            params["TRAIN"]["eval_interval"] - 1):
                        trainer.validation(epoch)

                trainer.writer.close()
            except yaml.YAMLError as exc:
                print(exc)
