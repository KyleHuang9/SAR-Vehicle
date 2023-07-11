import os
import argparse
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from module.core.engine import Trainer
from module.utils.config import Config

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Training', add_help=add_help)
    parser.add_argument('--conf-file', default='./configs/default.py', type=str, help='experiments description file')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output-dir', default='./Output', type=str, help='Output directory')
    return parser

def main(args):
    cfg = Config.fromfile(args.conf_file)
    trainer = Trainer(cfg=cfg, args=args)
    trainer.train()
    trainer.show_save_process()

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)