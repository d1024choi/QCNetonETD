
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ETRIDataModule
from predictors import QCNet

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default="/workspace/av2format")
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--devices', type=int, default=2, help='The number of possible GPU devices')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_processed_dir', type=str, default="train_qcnet")
    parser.add_argument('--val_processed_dir', type=str, default="val_qcnet")
    parser.add_argument('--test_processed_dir', type=str, default="test_qcnet")
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='ETRI_Dataset', help='DO NOT ALTER THIS')
    parser.add_argument('--num_historical_steps', type=int, default=20, help='DO NOT ALTER THIS')
    parser.add_argument('--num_future_steps', type=int, default=60, help='DO NOT ALTER THIS')
    parser.add_argument('--num_recurrent_steps', type=int, default=3, help='DO NOT ALTER THIS')
    parser.add_argument('--pl2pl_radius', type=int, default=150)
    parser.add_argument('--pl2a_radius', type=int, default=50)
    parser.add_argument('--a2a_radius', type=int, default=50)
    parser.add_argument('--pl2m_radius', type=int, default=150)
    parser.add_argument('--a2m_radius', type=int, default=150)
    parser.add_argument('--num_t2m_steps', type=int, default=10)
    parser.add_argument('--time_span', type=int, default=10)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--output_head', action='store_true')
    parser.add_argument('--num_modes', type=int, default=6, help='DO NOT ALTER THIS')
    parser.add_argument('--num_freq_bands', type=int, default=64)
    parser.add_argument('--num_map_layers', type=int, default=1)
    parser.add_argument('--num_agent_layers', type=int, default=2)
    parser.add_argument('--num_dec_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--head_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--T_max', type=int, default=64)
    args = parser.parse_args()

    datamodule = {'ETRI_Dataset': ETRIDataModule, }[args.dataset](**vars(args))
    model = QCNet(**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
