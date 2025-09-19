"""
Given yaml config file for model training
"""
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

from utils.saver import Saver
from utils.metrics import Evaluator
from utils.train_utils import colorize, ensure_dir, save_stack_img, load_my_state_dict


class Validator():
    """Define model validator"""
    def __init__(self, _config):
        self._c = _c = _config
        params["gpu_ids"] = [0]
        params["output"] = params["DIR"]
        params["dataset"] = ""
        params["checkname"] = params['MODEL']['arch']+'_'+params['MODEL']['expert']+ params['MODEL']['gate']

        self.is_cuda = len(_c["gpu_ids"]) > 0
        self.device = torch.device('cuda', _c["gpu_ids"][0]) \
            if self.is_cuda else torch.device('cpu')
        saver = Saver(_c)
        save_path = ensure_dir(os.path.join(saver.directory, 'eval_logs'))
        #_log = saver.create_logger(name="eval_"+"_".join(_c['test_sets']), save_path=save_path)
        self.test_loader, self.label_names, self.label_colors = \
            get_dataloader(_c, _c['test_sets'], 'test')
        
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
            
        self.evaluator = Evaluator(_c['DATASET']['num_class'])
        if self.is_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=_c["gpu_ids"])
            self.model = self.model.to(self.device)
        saved_ckpt_path = os.path.join(*[_c['DIR'], _c['MODEL']['arch']+'_'+
                                         _c['MODEL']['expert']+'_' #+ _c['MODEL']['layer']+'_'
                                         + _c['MODEL']['gate'],
                                         _c['VAL']['checkpoint']])
        assert os.path.exists(saved_ckpt_path), '{} not exit!'.format(saved_ckpt_path)
        print(saved_ckpt_path)
        new_state_dict = torch.load(saved_ckpt_path)
        self.model = load_my_state_dict(self.model.module, new_state_dict['state_dict'])

    def validate(self):
        """validate model"""
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        
        STORE_TO_CSV = True
        if STORE_TO_CSV:
            csv_name = "disagreements_of_frrn_" + self._c['MODEL']['expert']+'_'+ self._c['MODEL']['gate'] + "_on_data_" + ''.join(self._c['test_sets']) + ".csv"
            csv_name = os.path.join("./", csv_name)
            csvfile = open(csv_name, 'w')
            csvfile.write("id_id,perfect_case,moe_agrees_with_expert_1,moe_agrees_with_expert_2,critical_case")
            csvfile.write('\n')

        for i, sample in enumerate(tbar):
            image, label, label_path = sample
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                output = self.model(image)
            # ---upsampling output back to origin size
            #output = F.interpolate(output, size=(self._c['TEST']['img_height'],
            #                                     self._c['TEST']['img_width']),
            #                       mode='bilinear', align_corners=True)
            _, label_h, label_w = label.shape
            new_w = torch.linspace(0, label_w-1, self._c['DATASET']['img_width']).long()
            new_h = torch.linspace(0, label_h-1, self._c['DATASET']['img_height']).long()
            label = label[:, new_h[:, None], new_w]
            label = label.cpu().numpy()
            pred = output.data.cpu().numpy()
            moe_argmax = np.argmax(pred, axis=1).astype(np.uint8)

            # first expert prediction
            _, _, output_expert1 = self.model.forward_expert(self.model.expert1, image)
            output_expert1 = output_expert1.data.cpu().numpy()
            y1_argmax = np.argmax(output_expert1, axis=1).astype(np.uint8)
            # convert_to_img_and_save(y1_argmax, input_file, "_expert1.png")

            _, _, output_expert2 = self.model.forward_expert(self.model.expert2, image)
            output_expert2 = output_expert2.data.cpu().numpy()
            y2_argmax = np.argmax(output_expert2, axis=1).astype(np.uint8)
            # convert_to_img_and_save(y2_argmax, input_file, "_expert2.png")

            expert_agree = np.equal(y1_argmax[0], y2_argmax[0])
            moe_agrees_with_expert_1 = np.equal(y1_argmax[0], moe_argmax[0])
            moe_agrees_with_expert_2 = np.equal(y2_argmax[0], moe_argmax[0])

            # both experts and moe agree
            perfect_cases_mask = np.logical_and(np.equal(y1_argmax[0], moe_argmax[0]),
                                                np.equal(y2_argmax[0], moe_argmax[0]),
                                                expert_agree)
            # pixels where moe chooses the same as expert 1
            normal_case_1_mask = np.logical_and(moe_agrees_with_expert_1, ~expert_agree)

            # pixels where moe chooses the same as expert 2
            normal_case_2_mask = np.logical_and(moe_agrees_with_expert_2, ~expert_agree)

            # pixels where moe chose differently from any of the experts
            critical_cases_mask = np.logical_and(~moe_agrees_with_expert_1, ~moe_agrees_with_expert_2)

            str_output = str(np.sum(perfect_cases_mask)) + "," + str(np.sum(normal_case_1_mask)) + "," + str(
                np.sum(normal_case_2_mask)) + "," + str(np.sum(critical_cases_mask))
            print(str_output)

            if STORE_TO_CSV:
                csvfile.write(str(i) + "," + str_output)
                csvfile.write('\n')

            if self._c['VAL']['visualize']:
                for i in range(output.shape[0]):
                    # Val batch size is 1.
                    pred = output[i].data.max(0)[1].cpu().numpy()
                    ground_true = label[i]
                    pred_colors = colorize(self.label_colors, pred)
                    gt_colors = colorize(self.label_colors, ground_true)
                    # save wrt. label path and weight path
                    _, city, label_name = label_path[i].split('/')[-3:]
                    pred_save_path = os.path.join(
                        os.path.dirname(self._c['VAL']['checkpoint']), 'pred_results',
                        city, label_name)
                    if not os.path.exists(os.path.dirname(pred_save_path)):
                        os.makedirs(os.path.dirname(pred_save_path))
                    save_stack_img([gt_colors, pred_colors], pred_save_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\nPlease pass the desired param file for training as an argument.\n'
            'e.g: params/params_moe.py')
    else:
        print('STARTING DISAGREEMENT ANALYSIS WITH PARAM FILE: ', str(sys.argv[1]))
        with open(str(sys.argv[1]), 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                validator = Validator(params)
                validator.validate()
            except yaml.YAMLError as exc:
                print(exc)

