from fvcore.nn import FlopCountAnalysis
from typing import Dict, Optional, Mapping

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from modules.qcnet_agent_encoder import QCNetAgentEncoder
from modules.qcnet_map_encoder import QCNetMapEncoder
class QCNetEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNetEncoder, self).__init__()

        self.dataset = dataset
        self.input_dim = input_dim
        self.num_historical_steps = num_historical_steps

        self.map_encoder = QCNetMapEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.agent_encoder = QCNetAgentEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def prepare_map_encoder_inputs(self, data:HeteroData):

        pos_pt = data['map_point']['position'][:, :self.input_dim].contiguous()   # N x 2
        orient_pt = data['map_point']['orientation'].contiguous()                 # N
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous() # M x 2
        orient_pl = data['map_polygon']['orientation'].contiguous()               # M
        orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)# M x 2

        if self.dataset == 'ETRI_Dataset':
            if self.input_dim == 2:
                x_pt = data['map_point']['magnitude'].unsqueeze(-1)               # N x 1
                x_pl = None
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        map_polygon_batch = data['map_polygon']['batch'] if isinstance(data, Batch) else None # M (int64)
        map_point_batch = data['map_point']['batch'] if isinstance(data, Batch) else None     # N (int64)

        # New
        # We are also passing the connectivity information to the map encoder, ensuring that the map encoder aligns with the original QCNet implementation.
        edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        edge_index_pl2pl = data['map_polygon', 'to', 'map_polygon']['edge_index']

        return (pos_pt, orient_pt, pos_pl, orient_pl, orient_vector_pl, x_pt, x_pl, map_polygon_batch, map_point_batch, edge_index_pt2pl, edge_index_pl2pl )

    def prepare_agent_encoder_inputs(self, data: HeteroData, map_enc: Mapping[str, torch.Tensor], isFake=False):

        map_enc_x_pl = map_enc['x_pl'] # M x obs_len x 128

        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous() # num_agent x obs_len (bool)
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous() # num_agent x obs_len x 2

        # New
        # The motion_vector is constructed here and is passed to the agent_encoder, it is now exaclty the same as the QCent, there was a bug in the previous one.
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous() # num_agent x obs_len (bool)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous() # M x 2
        orient_pl = data['map_polygon']['orientation'].contiguous() # M
        if self.dataset == 'ETRI_Dataset':
            vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous() # num_agent x obs_len x 2
            agent_type = data['agent']['type'].long() # (num_agent x obs_len) x 128
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(self.num_historical_steps)], dim=0) # (num_agent x obs_len)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(self.num_historical_steps)], dim=0) # (M x obs_len)
        else:
            batch_s = torch.arange(self.num_historical_steps,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(self.num_historical_steps,
                                    device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])

        return (map_enc_x_pl, mask, pos_a, motion_vector_a, head_a, pos_pl, orient_pl, vel, agent_type, batch_s, batch_pl)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(*self.prepare_map_encoder_inputs(data))
        agent_enc = self.agent_encoder(*self.prepare_agent_encoder_inputs(data, map_enc))
        return {**map_enc, **agent_enc}
