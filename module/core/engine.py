import os
import os.path as osp
import math
import torch
import torchvision
from torch import nn
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

import tools.eval as eval
import tools.test as test
from module.model.net import Model
from module.data.data_load import create_dataloader, create_testloader
from module.utils.event import LOGGER, NCOLS

class Trainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.txt_dir = cfg.dataset.txt_dir
        self.img_size = cfg.dataset.img_size
        self.nc = cfg.dataset.nc
        self.base_channels = cfg.model.base_channels
        self.max_epoch = cfg.params.epoch
        self.warm_up_epoch = cfg.params.warm_up
        self.batch_size = cfg.params.batch_size
        self.device = args.device
        self.model = self.get_model().to(device=self.device)
        self.train_dataloader = self.get_dataloader(self.txt_dir, task="train")
        self.val_dataloader = self.get_dataloader(self.txt_dir, task="val")
        self.test_dataloader = create_testloader(self.txt_dir, self.nc, self.img_size, self.batch_size, rank=-1, workers=self.args.workers)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.params.lr0, momentum=cfg.params.momentum, weight_decay=cfg.params.weight_decay)
        self.loss = nn.BCEWithLogitsLoss().to(device=self.device)
        self.best_acc = 0
        self.eval_interval = cfg.params.eval_interval

        self.train_loss = []
        self.train_acc = []
        self.val_acc = []

    def train(self):
        self.create_output_dir()
        total_step = len(self.train_dataloader)
        for epoch in range(self.max_epoch):
            self.model.train()
            self.mean_loss = torch.zeros(1, device=self.device)
            LOGGER.info(('\n' + '%7s' + '%15s' * 2) % ("Epoch", "loss", "lr"))
            self.pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for step, batch_data in self.pbar:
                imgs, targets = self.preprogess(batch_data)
                
                output = self.model(imgs)
                loss = self.loss(output, targets)
            
                self.lr_resolve(epoch, total_step, step)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.print_details(epoch, step, loss)
                self.train_loss.append(loss.item())

            if (epoch + 1) % self.eval_interval == 0:
                self.eval_and_save()
            self.save_last_model()
        LOGGER.info(f"\nThe best accuracy is {self.best_acc}")


    def get_model(self):
        if self.cfg.model.model is None:
            LOGGER.info('Loading Model.')
            model = Model(self.nc, self.base_channels)
        else:
            LOGGER.info(f'Loading pretrained model {self.cfg.model.model}.')
            model = torch.load(self.cfg.model.model, map_location='cpu')
        print("Model:\n", model)
        return model

    def get_dataloader(self, txt_dir, task):
        if task != "train":
            augment = False
        else:
            augment = True
        dataloader = create_dataloader(txt_dir, self.nc, img_size=self.img_size, batch_size=self.batch_size, hyp=dict(self.cfg.data_aug),
                            augment=augment, rank=-1, workers=self.args.workers, shuffle=True, task=task)
        return dataloader

    def preprogess(self, batch_data):
        imgs = batch_data[0].to(self.device, non_blocking=True).float() / 255
        targets = batch_data[1].to(self.device)
        return imgs, targets

    def print_details(self, epoch, step, loss):
        self.mean_loss = (self.mean_loss * step + loss) / (step + 1)
        self.pbar.set_description(('%7s' + '%15.4g' + '%15.4g') % (f'{epoch + 1}/{self.max_epoch}', \
                                                                (self.mean_loss), self.optimizer.param_groups[0]["lr"]))

    def eval_and_save(self):
        train_acc, train_acc_list = eval.run(model=self.model,
                        dataloader=self.train_dataloader,
                        model_path=None,
                        txt_dir=self.txt_dir,
                        img_size=self.img_size,
                        nc=self.nc,
                        batch_size=self.batch_size,
                        workers=self.args.workers,
                        device=self.device,
                        task="val")
        acc, acc_list= eval.run(model=self.model,
                        dataloader=self.val_dataloader,
                        model_path=None,
                        txt_dir=self.txt_dir,
                        img_size=self.img_size,
                        nc=self.nc,
                        batch_size=self.batch_size,
                        workers=self.args.workers,
                        device=self.device,
                        task="test")
        for i in range(len(acc_list)):
            LOGGER.info(f"{i}: {acc_list[i]} \t {train_acc_list[i]}")
        LOGGER.info(f"The val accuracy is {train_acc}.")
        LOGGER.info(f"The test accuracy is {acc}.")

        self.train_acc.append(train_acc)
        self.val_acc.append(acc)

        if acc > self.best_acc:
            file_name = osp.join(self.args.output_dir, "best_ckpt.pt")
            torch.save(self.model, file_name)
            self.best_acc = acc
            self.test()

    def test(self):
        test.run(
            model=self.model,
            dataloader=self.test_dataloader,
            model_path=None,
            txt_dir=self.txt_dir,
            output_dir=self.args.output_dir,
            img_size=self.img_size,
            nc=self.nc,
            batch_size=self.batch_size,
            workers=self.args.workers,
            device=self.device,
        )

    def save_last_model(self):
        self.model.eval()
        file_name = osp.join(self.args.output_dir, "last_ckpt.pt")
        torch.save(self.model, file_name)
        
    def create_output_dir(self):
        if not osp.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)

    def lr_resolve(self, epoch, total_step, step):
        if epoch < self.warm_up_epoch:
            for param in self.optimizer.param_groups:
                param["lr"] = ((epoch * total_step + step) * self.cfg.params.lr0) / (self.warm_up_epoch * total_step)
        else:
            for param in self.optimizer.param_groups:
                param["lr"] = (math.cos(epoch * (math.pi / 2) / self.max_epoch) * (self.cfg.params.lr0 - self.cfg.params.lr1)) + self.cfg.params.lr1
    
    def show_save_process(self):
        # loss
        plt.figure()
        x = [i for i in range(len(self.train_loss))]
        plt.plot(x, self.train_loss)
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(osp.join(self.args.output_dir, 'loss.png'))

        # acc
        plt.figure()
        x = [i for i in range(len(self.train_acc))]
        plt.plot(x, self.train_acc, 'b', x, self.val_acc, 'r')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.savefig(osp.join(self.args.output_dir, 'accuracy.png'))