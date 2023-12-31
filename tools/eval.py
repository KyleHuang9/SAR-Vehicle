import argparse
import os.path as osp
from module.core.evaler import Evaler
from module.utils.config import Config
from module.utils.event import LOGGER

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Evaluating', add_help=add_help)
    parser.add_argument('--conf-file', default='./configs/custom.py', type=str, help='experiments description file')
    parser.add_argument('--weights', default='./Output/last_ckpt.pt', type=str, help='model path')
    parser.add_argument('--batch-size', default=32, type=str, help='batch_size')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=5, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--output-dir', default='./Output', type=str, help='Output directory')
    return parser

def run(model, dataloader, model_path, txt_dir, img_size, nc, batch_size, workers, device, task):
    evaler = Evaler(txt_dir=txt_dir, img_size=img_size, nc=nc, batch_size=batch_size, workers=workers,device=device)
    if model_path != None:
        model = evaler.get_model(model_path)
    if dataloader == None:
        dataloader = evaler.get_dataloader(txt_dir, task)
    accuary, acc_list = evaler.eval(model=model, dataloader=dataloader, task=task)
    return accuary, acc_list

def main(args):
    cfg = Config.fromfile(args.conf_file)
    acc, acc_list = run(model=None,
            dataloader=None,
            model_path=args.weights,
            txt_dir=cfg.dataset.img_dir,
            img_size=cfg.dataset.img_size,
            nc=cfg.dataset.nc,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            task='val')
    for i in range(len(acc_list)):
        LOGGER.info(f"{i}: {acc_list[i]}")
    LOGGER.info(("The accuracy is: %.4f") % (acc))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)