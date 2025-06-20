
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ETRIDataset
from predictors import QCNet
from transforms import TargetBuilder

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="QCNet")
    parser.add_argument('--root', type=str, default="/home/dooseop/DATASET/ETRI/av2format")
    parser.add_argument('--test_processed_dir', type=str, default="test_qcnet")
    parser.add_argument('--batch_size', type=int, default=1, help='DO NOT ALTER THIS')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default="./lightning_logs/version_22/checkpoints/epoch=16-step=2142.ckpt")
    args = parser.parse_args()

    model = {'QCNet': QCNet,}[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)

    test_dataset = {'ETRI_Dataset': ETRIDataset,}[model.dataset](args.root, args.test_processed_dir)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    trainer.test(model, dataloader)