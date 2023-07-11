import torch
from tqdm import tqdm

from module.data.data_load import create_dataloader
from module.utils.event import NCOLS

class Evaler:
    def __init__(self, txt_dir, img_size, nc, batch_size, workers, device):
        self.txt_dir = txt_dir
        self.img_size = img_size
        self.nc = nc
        self.batch_size = batch_size
        self.workers = workers
        self.device = device
    
    def eval(self, model, dataloader, task):
        pbar = tqdm(dataloader, desc=f"Inferencing model in {task} datasets.", ncols=NCOLS)

        count = 0
        right = 0
        class_count = [0 for i in range(self.nc)] # add non-label obj
        class_right = [0 for i in range(self.nc)]

        for i, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            targets = targets.to(self.device)

            output = model(imgs)

            output = torch.softmax(output, dim=-1)
            conf, predict = torch.max(output, -1, keepdim=True)
            _, targets = torch.max(targets, -1)

            assert len(predict) == len(targets)

            for i in range(len(targets)):
                count += 1
                class_count[targets[i]] += 1
                if predict[i] == targets[i]:
                    right += 1
                    class_right[targets[i]] += 1
            
        AP = right * 1.0 / count if count > 0 else 0.0
        ap_list = [(class_right[i] / class_count[i] if class_count[i] > 0 else 0.0) for i in range(self.nc)]
        return AP, ap_list

    def get_model(self, model_path):
        model = torch.load(model_path)
        print(model)
        return model

    def get_dataloader(self, txt_dir, task):
        dataloader = create_dataloader(txt_dir, self.nc, img_size=self.img_size, batch_size=self.batch_size, hyp=None,
                            augment=False, rank=-1, workers=self.workers, shuffle=True, task=task)
        return dataloader
