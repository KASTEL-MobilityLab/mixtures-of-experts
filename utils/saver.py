"""Saver"""
import os
import json
import shutil
from collections import OrderedDict
import glob
import torch


class Saver:
    """Saver"""
    def __init__(self, params):
        self.params = params
        self.directory = os.path.join(params["output"], params["dataset"],
                                      params["checkname"])
        self.runs = sorted(
            glob.glob(os.path.join(self.directory, "experiment_*")))
        run_id = int(self.runs[-1].split("_")[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory,
                                           "experiment_{}".format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename="checkpoint.pth.tar"):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state["best_pred"]
            with open(os.path.join(self.experiment_dir, "best_pred.txt"),
                      "w") as opened_file:
                opened_file.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split("_")[-1]
                    path = os.path.join(
                        self.directory,
                        "experiment_{}".format(str(run_id)),
                        "best_pred.txt",
                    )
                    if os.path.exists(path):
                        with open(path, "r") as opened_file:
                            miou = float(opened_file.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(
                        filename,
                        os.path.join(self.directory, "model_best.pth.tar"),
                    )
            else:
                shutil.copyfile(
                    filename, os.path.join(self.directory,
                                           "model_best.pth.tar"))

    def save_experiment_config(self):
        """Save experiment config"""
        logfile = os.path.join(self.experiment_dir, "parameters.txt")
        log_file = open(logfile, "w")
        experiment_params = OrderedDict()
        experiment_params["datset"] = self.params["dataset"]
        experiment_params["backbone"] = self.params["backbone"]
        experiment_params["out_stride"] = self.params["out_stride"]
        experiment_params["lr"] = self.params["lr"]
        experiment_params["lr_scheduler"] = self.params["lr_scheduler"]
        experiment_params["loss_type"] = self.params["loss_type"]
        experiment_params["epoch"] = self.params["epochs"]
        experiment_params["base_size"] = self.params["base_size"]
        experiment_params["crop_size"] = self.params["crop_size"]

        for key, val in experiment_params.items():
            log_file.write(key + ":" + str(val) + "\n")
        log_file.close()
        logfile = os.path.join(self.experiment_dir, "parameters_all.txt")
        with open(logfile, "w") as opened_file:
            json.dump(self.params, opened_file, indent=4)
