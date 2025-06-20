from fvcore.nn import FlopCountAnalysis
from thop import profile

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from datamodules import ETRIDataModule
from predictors import QCNet
from torch_geometric.data import HeteroData
import pickle
import torch


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default="/home/dooseop/DATASET/ETRI/av2format")
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_processed_dir', type=str, default="train_qcnet")
    parser.add_argument('--val_processed_dir', type=str, default="val_qcnet")
    parser.add_argument('--test_processed_dir', type=str, default="test_flops_qcnet")
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=2, help='The number of possible GPU devices')
    parser.add_argument('--max_epochs', type=int, default=4)
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

    from modules.qcnet_agent_encoder import QCNetAgentEncoder
    from modules.qcnet_map_encoder import QCNetMapEncoder
    from modules import QCNetDecoder

    model = QCNet(**vars(args)).float()

    map_encoder = QCNetMapEncoder(**vars(args)).float()
    agent_encoder = QCNetAgentEncoder(**vars(args)).float()
    decoder = QCNetDecoder(**vars(args)).float()

    from datasets import ETRIDataset
    from torch_geometric.loader import DataLoader
    test_dataset = ETRIDataset(args.root, args.test_processed_dir)
    dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    GFLOPs, counter = 0, 0
    for b, data in enumerate(dataloader):

        macs, params = profile(map_encoder, inputs=model.encoder.prepare_map_encoder_inputs(data))
        GFLOPs += 2 * macs / 1e9

        map_enc = map_encoder(*model.encoder.prepare_map_encoder_inputs(data))
        macs, params = profile(agent_encoder, inputs=model.encoder.prepare_agent_encoder_inputs(data, map_enc))
        GFLOPs += 2 * macs / 1e9

        agent_enc = agent_encoder(*model.encoder.prepare_agent_encoder_inputs(data, map_enc))
        scene_enc = {**map_enc, **agent_enc}
        macs, params = profile(decoder, inputs=model.prepare_decoder_inputs(data, scene_enc))
        GFLOPs += 2 * macs / 1e9

        counter += 1

    GFLOPs = GFLOPs / counter
    print(f" ---------------------------------------")
    print(f" The estimated FLOPs : {GFLOPs:.2f} G")
    print(f" ---------------------------------------")