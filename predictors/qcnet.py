import os
import pickle
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional, Mapping
from torch.nn.utils.rnn import pad_sequence
import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder
from modules import QCNetEncoder
#%%
try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 **kwargs) -> None:
        super(QCNet, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius      # polygon_to_polygon
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius     #polygon_to_agent
        self.a2a_radius = a2a_radius        #agent_to_agent
        self.num_t2m_steps = num_t2m_steps    # num of t
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.test_time_spent = 0
        self.num_test_scenes = 0

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_dec_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head, reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head, reduction='none')

        self.minADE = minADE(max_guesses=num_modes)
        self.minFDE = minFDE(max_guesses=num_modes)

        self.validation_results = []
        self.test_results = []

    def prepare_decoder_inputs(self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor]):

        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']

        agent_batch = data['agent']['batch'] if isinstance(data, Batch) else None
        map_polygon_batch = data['map_polygon']['batch'] if isinstance(data, Batch) else None

        map_num_nodes = data['map_polygon']['num_nodes']
        agent_num_nodes = data['agent']['num_nodes']

        return (pos_m, head_m, x_t, x_pl, x_a, mask_src, mask_dst, pos_t, head_t, pos_pl, orient_pl, agent_batch,
                map_polygon_batch, map_num_nodes, agent_num_nodes)

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(*self.prepare_decoder_inputs(data, scene_enc))
        return pred

    def training_step(self, data, batch_idx):

        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        return loss

    def validation_step(self, data, batch_idx):

        # apply prediction
        data = data
        pred = self(data)

        # preparation for evaluation
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        # For Joint Motion Prediction
        eval_mask_list = []
        all_categories = data['agent']['category']  # List or nested structure
        for categories in all_categories:
            categories = np.array(categories)
            mask = categories == 2  # This marks every occurrence of 2
            eval_mask_list.append(mask)
        eval_mask = np.concatenate(eval_mask_list) 
        eval_mask = torch.tensor(eval_mask, dtype=torch.bool, device='cuda' if torch.cuda.is_available() else 'cpu')

        # 'predict_mask' gives the information about the availability of the ground-truth position at each time step
        valid_mask_future_horizon = data['agent']['predict_mask'][:, self.num_historical_steps:]
        valid_mask_eval = valid_mask_future_horizon[eval_mask]

        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)

        pi_eval = F.softmax(pred['pi'][eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))


    def on_validation_epoch_end(self):
        do_nothing = 0

    
    def test_step(self, data, batch_idx):


        # measure inference time (ms) -------
        start = time.time()
        pred = self(data)
        end = time.time()
        self.test_time_spent += (end - start) * 1000.0
        # measure inference time (ms) -------

        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        # agent-centric to global coordinate system
        origin_eval = data['agent']['position'][:, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][:, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(data['agent']['num_nodes'], 2, 2, device=self.device)
        rot_mat[:, 0, 0], rot_mat[:, 0, 1] = cos, sin
        rot_mat[:, 1, 0], rot_mat[:, 1, 1] = -sin, cos
        traj_eval_pred = torch.matmul(traj_refine[:, :, :, :2], rot_mat.unsqueeze(1)) \
                         + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        traj_eval_pred = traj_eval_pred.cpu()

        # save test results
        # start_end_indices = np.cumsum(np.insert(data['agent']['num_nodes'].to('cpu').numpy(), 0, 0))
        start_end_indices = data['agent']['ptr'].cpu().numpy()

        for idx, (start, end) in enumerate(zip(start_end_indices[:-1], start_end_indices[1:])):
            agent = {
                'num_nodes': end - start,
                'num_valid_nodes': data['agent']['num_valid_nodes'][idx].item(),
                'id': data['agent']['id'][idx],
                'category': data['agent']['category'][idx],
                'predictions': traj_eval_pred[start:end].numpy()
            }

            scene = {
                'log_id': data['log_id'][idx],
                'frm_idx': data['frm_idx'][idx].item(),
                'agent': agent
            }

            self.num_test_scenes += 1
            self.test_results.append(scene)

    
    def on_test_end(self):

        results_dir = os.path.join(self.trainer.log_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)

        for scene in self.test_results:
            log_id, frm_idx = scene['log_id'], scene['frm_idx']
            results_path = os.path.join(results_dir, f'log_{log_id}_{frm_idx:07d}_submission.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(scene, f)
        self.test_results.clear()

        avg_inf_time = self.test_time_spent / self.num_test_scenes
        print(f">> Average Inference Time Per Scene (ms) : {avg_inf_time:.2f}")

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]


# %%
