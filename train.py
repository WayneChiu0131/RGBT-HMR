#!/usr/bin/env python3
"""Training entry point.

Usage:
    python train.py --train_cfg configs/train/<cfg>.yaml \
                    --paths_cfg configs/paths/local.yaml [--max_step N]

Parses TrainOptions (YAML + CLI overrides), builds the Trainer, and runs the
step-based training loop (utils.base_trainer.BaseTrainer.train).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.train_options import TrainOptions
from train.trainer import Trainer


def main():
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()


if __name__ == '__main__':
    main()
