import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import yaml
import wandb
import time
import uuid
import os
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import sys
import os

def add_subfolders_to_path(folder_path):
    for root, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            sys.path.append(full_path)  # Or use sys.path.insert(0, full_path)

# Usage:
folder_to_add = './'
add_subfolders_to_path(folder_to_add)

from models.baseline import BaselineModel
from dataloader import SegmentationDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ConstantLR

class BaseTrainer:
    def __init__(self, load=True, **kwargs):
        run_dir = kwargs.get('run_dir', './runs')
        model = kwargs.get('model_object', 'BaselineModel')

        self.config = {
            **kwargs,
            "model_object": model,
            "checkpoint_dir": str(Path(run_dir) / "checkpoints"),
            "results_dir": str(Path(run_dir) / "results"),
            "logs_dir": str(Path(run_dir) / "logs"),
        }

        self.is_debug = self.config.get("is_debug", False)
        self.silent = self.config.get("silent", False)

        self.epoch = 0
        self.losses = []
        self.step = 0
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.datasets = {}
        self.loaders = {}

        self.timestamp_id = time.strftime("%Y%m%d-%H%M%S")
        self.job_id = uuid.uuid4().hex[:8]
        self.config["timestamp_id"] = self.timestamp_id
        self.run_name = f"{kwargs['config_name']}_{self.timestamp_id}"
        self.saved_checkpoint = False
        # Early stopping with file creation?

        if load:
            self.load()

    def load(self, checkpoint_name=None):
        self.load_logger()
        self.load_train_val_test_datasets()
        if checkpoint_name is not None:
            self.load_checkpoint(checkpoint_name)
        else:
            self.load_model()
        self.load_optimizer()
        self.load_loss()
        self.get_dataloader()
    
    def load_logger(self):
        self.logger = None
        if not self.is_debug:
            logger = self.config.get("logger", "wandb")
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Logger name not provided"

            self.logger = wandb.init(
                project=self.config["wandb_project"],
                name=self.run_name,
                config=self.config,
                dir=self.config["logs_dir"],
            )

    def load_train_val_test_datasets(self):
        self.train_dataset = SegmentationDataset(
            root_dir='./data', split="Train"
        )
        self.val_dataset = SegmentationDataset(
            root_dir='./data', split="Val"
        )
        self.test_dataset = SegmentationDataset(
            root_dir='./data', split="Test"
        )
    
    def get_dataloader(self):
        batch_size = self.config["optim"]["batch_size"]
        eval_batch_size = self.config["optim"]["eval_batch_size"]
        max_epochs = self.config["optim"].get("max_epochs", -1)
        max_steps = self.config["optim"].get("max_steps", -1)
        max_samples = self.config["optim"].get("max_samples", -1)

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=eval_batch_size, shuffle=True
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=eval_batch_size, shuffle=False
        )
    
    def load_model(self):
        loader = list(self.loaders.values())[0] if self.loaders else None
        if loader:
            sample = loader.dataset[0]
            # extract inputs and targets if necessary

        model_config = {
            # add other features if necessary
            **self.config["model"],
        }

        # Extract model class from name
        print(self.config)
        self.model = self.config["model_object"](**model_config).to(self.device)

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")

        optimizer = eval(optimizer)

        self.optimizer = optimizer(
            self.model.parameters(),
            lr=float(self.config["optim"]["lr_initial"]),
            weight_decay=float(self.config["optim"].get("weight_decay", 0.01)),
        )

        if self.config["optim"].get("scheduler", "cosine") == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(self.config["optim"].get("warmup_steps", 100)),
                # T_mult=self.config["optim"].get("T_mult", 2),
                eta_min=float(self.config["optim"].get("lr_min", 0)),
            )
        else:
            self.scheduler = ConstantLR(
                self.optimizer,
                last_epoch=-1,
            )
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
    
    def load_loss(self):
        loss_name = self.config["optim"].get("loss", "BCELoss")
        if loss_name == "BCELoss":
            self.loss = lambda x, y: BCELoss()(x, y)
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")
        
    def load_checkpoint(self, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_name = f"best_checkpoint_{self.run_name}.pt"
        checkpoint_path = self.config["checkpoint_dir"] + f"/{checkpoint_name}"
        self.save_path = checkpoint_path
        self.load_model()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)["model_state_dict"]
        )
        self.model.eval()

    def train(self):
        for i in tqdm(range(self.epoch, self.config["optim"]["max_epochs"])):
            if not self.silent:
                print("-----EPOCH{}-----".format(i + 1))
            self.epoch = i
            self.model.train()
            start_time = time.time()
            print_every = self.config.get("print_every", 20)
            count_iter = 0
            self.best_validation_loss = np.inf
            loss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                # if not self.is_debug:
                #     self.logger.log(
                #         {
                #             "lr": self.optimizer.param_groups[0]["lr"],
                #         }
                #     )

                if self.device.type != "cuda":
                    scaler = None
                    dtype = torch.bfloat16
                else:
                    scaler = torch.cuda.amp.GradScaler()
                    dtype = torch.float16
                scaler = None
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("precision", "foat32") == "float16",
                ):
                    images = images.reshape(-1, 1, 36, 36)
                    output = self.model(images.to(self.device).type(dtype))
                    current_loss = self.loss(output, labels.unsqueeze(1).to(self.device).type(dtype))
                    if scaler is not None:
                        self.optimizer.zero_grad()
                        scaler.scale(current_loss).backward()
                        scaler.unscale_(self.optimizer)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.zero_grad()
                        current_loss.backward()
                        self.optimizer.step()

                if self.config["optim"].get("scheduler", "cosine") == "cosine":
                    self.scheduler.step(i + count_iter / len(self.train_loader))
                else:
                    self.scheduler.step()
                loss += current_loss.item()

                count_iter += 1
                if not self.is_debug:
                    self.logger.log(
                        {
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )
                if count_iter % print_every == 0:
                    time2 = time.time()
                    self.losses.append(loss)
                    if not self.is_debug:
                        self.logger.log(
                            {
                                "loss": loss / print_every,
                                "epoch": i,
                                "iteration": count_iter,
                            }
                        )
                    loss = 0
            self.model.eval()
            val_loss = 0
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            for images, labels in self.val_loader:
                
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("precision", "foat32") == "float16",
                ):         
                    images = images.reshape(-1, 1, 36, 36)
                    output = self.model(images.to(self.device).type(dtype))
                    current_loss = self.loss(output, labels.unsqueeze(1).to(self.device).type(dtype))                   
                val_loss += current_loss.item()
            self.best_validation_loss = min(self.best_validation_loss, val_loss)
            if not self.silent:
                print(
                    "-----EPOCH" + str(i + 1) + "----- done.  Validation loss: ",
                    str(val_loss / len(self.val_loader)),
                )
            if not self.is_debug:
                log_dict = {
                    "validation_loss": val_loss / len(self.val_loader),
                    "epoch": i,
                }
                self.logger.log(
                    log_dict,
                )
            if self.best_validation_loss == val_loss:
                if not self.silent:
                    print("validation loss improved saving checkpoint...")
                self.save_path = (
                    self.config["checkpoint_dir"]
                    + f"/best_checkpoint_{self.run_name}.pt"
                )
                if not (Path(self.config["checkpoint_dir"])).exists():
                    os.makedirs(Path(self.config["checkpoint_dir"]), exist_ok=True)
                # If file exists, delete it
                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                if not self.saved_checkpoint:
                    self.saved_checkpoint = True
                    config_to_save = self.config.copy()
                    config_to_save["model_object"] = config_to_save[
                        "model_object"
                    ].__name__
                    yaml.dump(
                        config_to_save,
                        open(
                            self.save_path.replace(".pt", ".yaml").replace(
                                "best_checkpoint_", ""
                            ),
                            "w",
                        ),
                    )
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "validation_accuracy": val_loss,
                        "loss": loss,
                    },
                    self.save_path,
                )
                if not self.silent:
                    print("checkpoint saved to: {}".format(self.save_path))

                iou = self.get_iou_val(load_checkpoint=False)
                if not self.is_debug:
                    self.logger.log({"iou_val": iou})
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        iou = self.get_iou_val(load_checkpoint=True)
        if not self.is_debug:
            self.logger.log({"iou_val": iou})

    def get_iou_val(self, load_checkpoint=True):
        if not self.silent:
            print("Getting IOU on validation set...")

        preds, true_labels = self.submit_run(split="val", load_checkpoint=load_checkpoint)   
        from sklearn.metrics import jaccard_score

        iou = jaccard_score(true_labels, preds, average='macro')
        if not self.silent:
            print("IOU on validation set: ", iou)
        
        return iou

    def submit_run(self, split="test", load_checkpoint=True):
        if split == "Val" or split == "val":
            loader = self.val_loader
        else:
            if not self.silent:
                print("Submitting run on test set...")
            loader = self.test_loader
            try:
                # Delete train and val loaders
                del self.train_loader
                del self.val_loader
                del self.train_dataset
                del self.val_dataset
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except:
                pass

        if load_checkpoint:
            if not self.silent:
                print("Loading best checkpoint...")
            self.load_model()
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)["model_state_dict"]
            )
        self.model.eval()

        batch_size = self.config["optim"]["eval_batch_size"]
        preds = []
        true_labels = []
        try:
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.get("precision", "float32") == "float16",
            ):
                for images, labels in loader:
                    images = images.reshape(-1, 1, 36, 36)
                    output = self.model(images.to(self.device))                    
                    preds.append(output.detach().cpu().numpy())
                    true_labels.append(labels.detach().cpu().numpy())
        except Exception as e:
            print("Error occurred during prediction:", str(e))
        preds = np.concatenate(preds)
        true_labels = np.concatenate(true_labels)

        # Create a CSV submission file
        try:
            if not self.silent:
                print("Creating submission file...")
            if not os.path.exists(self.config["results_dir"]):
                os.makedirs(self.config["results_dir"], exist_ok=True)
            pd.DataFrame(preds, dtype='int').T.to_csv(
                self.config["results_dir"] + f"/{split}_submission.csv", index=False, header=False
            )

            if split == "Val" or split == "val":
                del self.val_loader
                del self.val_dataset
            else:
                del self.test_loader
                del self.test_dataset
        except Exception as e:
            print("Error creating submission file:", str(e))
             
        return preds, true_labels

