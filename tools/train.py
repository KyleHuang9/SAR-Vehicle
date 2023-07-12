import os
import argparse
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os.path as osp
import numpy as np

from module.core.engine import Trainer
from module.utils.config import Config
from tools.data.transdata import transdata, get_paths
import tools.test as test

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Training', add_help=add_help)
    parser.add_argument('--conf-file', default='./configs/default.py', type=str, help='experiments description file')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output-dir', default='./Output', type=str, help='Output directory')
    return parser

def creat_result_list(paths, num):
    results = {}
    for i in range(len(paths)):
        results[paths[i]] = [-1 for _ in range(num)]   
    return results
    
def main(args):
    cfg = Config.fromfile(args.conf_file)
    aug_prob = cfg.data_aug.augment
    transdata()
    
    paths = get_paths()
    results = creat_result_list(paths, cfg.model.model_num)

    for i in range(cfg.model.model_num):
        transdata()
        if aug_prob > 0.5:
            cfg.data_aug.augment = np.random.uniform(0.5, aug_prob)

        trainer = Trainer(cfg=cfg, args=args)
        model, test_dataloader = trainer.train()

        results = test.run(
            model=model,
            dataloader=test_dataloader,
            model_path=None,
            model_num=i,
            results=results,
            txt_dir=cfg.dataset.txt_dir,
            output_dir=args.output_dir,
            img_size=cfg.dataset.img_size,
            nc=cfg.dataset.nc,
            batch_size=cfg.params.batch_size,
            workers=args.workers,
            device=args.device,
        )

        #trainer.show_save_process()

    # save result

    output_txt = osp.join(args.output_dir, "test-result.txt")
    txt = open(output_txt, 'w')

    final_results = results
    paths = results.keys()
    for path in paths:
        predicts = results[path]
        predict = max(predicts, key=predicts.count)
        final_results[path] = predict
        txt.write(path + " " + str(predict) + "\n")
    txt.close()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)