import torch
import os.path as osp
from tqdm import tqdm

from module.data.data_load import create_testloader
from module.utils.event import NCOLS

class Tester:
    def __init__(self, txt_dir, output_dir, dataloader, results, img_size, nc, batch_size, workers, device):
        self.txt_dir = txt_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.nc = nc
        self.batch_size = batch_size
        self.workers = workers
        self.device = device

        self.result = results

        assert osp.exists(self.output_dir), "Test Output dir is not exist!"
    
    def test(self, model, dataloader, num):
        pbar = tqdm(dataloader, desc=f"Inferencing model in test datasets.", ncols=NCOLS)

        for i, (imgs, paths) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True).float() / 255

            output = model(imgs)

            output = torch.softmax(output, dim=-1)
            conf, predict = torch.max(output, -1, keepdim=True)

            for j in range(len(paths)):
                self.result[paths[j]][num] = predict[j].item()
        return self.result

    def get_model(self, model_path):
        model = torch.load(model_path)
        print(model)
        return model

    def get_dataloader(self, txt_dir):
        dataloader = create_testloader(txt_dir, self.nc, img_size=self.img_size, batch_size=self.batch_size, rank=-1, workers=self.workers)
        return dataloader
