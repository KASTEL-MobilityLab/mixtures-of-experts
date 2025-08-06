import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
import yaml

from dataloader.a2d2_loader import get_dataloader
from models.frrn import FRRNet
from models.frrn_ensemble import Ensemble
from models.frrn_moe import MoE
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.train_utils import cross_entropy2d, ensure_dir, colorize, load_my_state_dict, save_stack_img 

class Validator():
    """Define model validator"""
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods
    def __init__(self, _config):
        self.params = params = _config
        params["gpu_ids"] = [0]
        params["output"] = params["DIR"]
        params["dataset"] = ""
        params["checkname"] = params['MODEL']['arch']+'_'+params['MODEL']['expert']+ params['MODEL']['gate']
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
        print("\nFGSM attack with epsilon=", params["epsilon"])
        
        for _, sample in enumerate(tbar):
            image, label, label_path = sample
            image = image.to(self.device)
            image.requires_grad = True
            label = label.to(self.device)
            output = self.model(image)
            
            loss = self.loss(output, label)
            loss.backward()
            
            del output
            
            grad_sign = torch.sign(image.grad)
            perturbed_image = image + params["epsilon"] * grad_sign
            output = self.model(perturbed_image)
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

        # Fast test during the training
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        miou, log_miou_cls = self.evaluator.mean_intersection_over_union(self.label_names)
        fwiou = self.evaluator.frequency_weighted_intersection_over_union()
        print("Validation:")
        print("pAcc:{}, mAcc:{}, m_iou:{}, fwIoU: {}".format(acc, acc_class, miou, fwiou))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
            'e.g: params/params_moe.py')
    else:
        print('STARTING EVALUATION WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                validator = Validator(params)
                validator.validate()
            except yaml.YAMLError as exc:
                print(exc)

