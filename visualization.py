import pickle
import torch
import numpy as np
from matplotlib import pyplot as plt
import os

_ESTIMATED_VEHICLE_LENGTH_M = 5.0
_ESTIMATED_VEHICLE_WIDTH_M = 2.0
_ESTIMATED_CYCLIST_LENGTH_M = 2.0
_ESTIMATED_CYCLIST_WIDTH_M = 0.7
_ESTIMATED_PEDESTRIAN_LENGTH_M = 0.3
_ESTIMATED_PEDESTRIAN_WIDTH_M = 0.5
_ESTIMATED_BUS_LENGTH_M = 7.0
_ESTIMATED_BUS_WIDTH_M = 2.1


class AV2_vis:

    def __init__(self):
        self.num_historical_steps = 20
        self.num_prediction_steps = 60
        self.agent = None
        self.focal = None
        self.min_speed_threshold = 0.5  # Threshold for considering a vehicle "slow-moving"

    def get_current_pos_and_rotation(self, traj_data, num_nodes, theta):
        orig = traj_data
        cos, sin = theta.cos(), theta.sin()

        rotation_matrix = theta.new_zeros(num_nodes, 2, 2)
        rotation_matrix[:, 0, 0] = cos
        rotation_matrix[:, 0, 1] = -sin
        rotation_matrix[:, 1, 0] = sin
        rotation_matrix[:, 1, 1] = cos
        return orig, rotation_matrix, theta

    def draw_scene(self, ax, traj_data, pred_data, gt_data, isGTAvailable=False):

        assert (traj_data['log_id'] == pred_data['log_id'])
        assert (traj_data['frm_idx'] == pred_data['frm_idx'])
        assert (traj_data['agent']['id'] == pred_data['agent']['id'])

        ax.axis('equal')

        # draw centerlines first ---
        map_data = gt_data['map']
        for map in map_data:
            x_vals = map['Pts'][:, 0]
            y_vals = map['Pts'][:, 1]
            ax.plot(x_vals, y_vals, color='#555555', linewidth=6.0, linestyle='-', alpha=0.6, zorder=0.1)
            ax.plot(x_vals, y_vals, color='white', linewidth=1.0, linestyle='--', alpha=0.8, zorder=0.2)


        # draw trajectories ---
        num_agents = traj_data['agent']['num_nodes']
        pred_future = torch.from_numpy(pred_data['agent']['predictions'])
        pos_obs = traj_data['agent']['position'][:, :self.num_historical_steps, :2]
        if (isGTAvailable):
            pos_fut = gt_data['agent']['position'][:, self.num_historical_steps:, :2]
            f_angle = gt_data['agent']['heading'][:, self.num_historical_steps:]  # Future heading information
        else:
            pos_fut = traj_data['agent']['position'][:, self.num_historical_steps:, :2]
            f_angle = traj_data['agent']['heading'][:, self.num_historical_steps:]  # Future heading information

        c_pos = traj_data['agent']['position'][:, self.num_historical_steps - 1, :2]
        traj_cat = traj_data['agent']['category']
        traj_type = traj_data['agent']['type']
        AV = traj_data['agent']['av_index']
        h_theta = traj_data['agent']['heading'][:, self.num_historical_steps - 1]



        # 1) bboxes and observed trajectories for validate vehicles
        orig_act, rot_act, theta_act = self.get_current_pos_and_rotation(c_pos, num_agents, h_theta)
        agent_cat = traj_cat.cpu().numpy() if isinstance(traj_cat, torch.Tensor) else np.array(traj_cat)
        traj_type = traj_type.cpu().numpy() if isinstance(traj_type, torch.Tensor) else np.array(traj_type)
        AV = AV.cpu().numpy() if isinstance(AV, torch.Tensor) else np.array(AV)
        for i, (pos_obs_i, orig_act_i, pos) in enumerate(zip(pos_obs, orig_act, pos_fut)):
            if orig_act_i[0] <= 0.0 and orig_act_i[1] <= 0.0:
                continue
            traj_obs = pos_obs_i
            agent_cat_i = agent_cat[i]
            zorder = 10
            if i == AV:
                color = 'green'
                self.agent = i
                zorder = 10
                traj_obs = pos_obs_i
            elif agent_cat_i == 2:
                color = 'blue'
                self.focal = i
                zorder = 20
                ax.plot(traj_obs[:, 0], traj_obs[:, 1], marker='.', alpha=1, color=color, zorder=zorder)
            else:
                color = 'black'
                zorder = 20

            if traj_type.ndim == 2 and traj_type.shape[1] > 1:  # Assuming traj_type is 2D
                traj_type_i = np.where(traj_type[i][len(pos_obs) - 1])[0][0]
            else:
                traj_type_i = traj_type[i]

            theta = h_theta[i]
            dtype = orig_act_i.dtype if hasattr(orig_act_i, 'dtype') else torch.float32
            act_rot = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]], dtype=dtype)

            if traj_type_i == 0:
                bbox_l = _ESTIMATED_VEHICLE_LENGTH_M
                bbox_w = _ESTIMATED_VEHICLE_WIDTH_M
            elif traj_type_i == 1:
                bbox_l = _ESTIMATED_PEDESTRIAN_LENGTH_M
                bbox_w = _ESTIMATED_PEDESTRIAN_WIDTH_M
            elif traj_type_i == 2 or traj_type_i == 3:
                bbox_l = _ESTIMATED_CYCLIST_LENGTH_M
                bbox_w = _ESTIMATED_CYCLIST_WIDTH_M

            else:
                bbox_l = 0.5  # static
                bbox_w = 0.5

            bbox = torch.tensor([[-bbox_l / 2, -bbox_w / 2],
                                 [-bbox_l / 2, bbox_w / 2],
                                 [bbox_l / 2, bbox_w / 2],
                                 [bbox_l / 2, -bbox_w / 2]], dtype=dtype)

            if isinstance(rot_act, torch.Tensor) and rot_act.dim() > 2:
                rot_act = rot_act.squeeze(0)

            bbox = torch.matmul(bbox, act_rot.T) + orig_act_i

            if traj_cat[i] != 0:
                ax.fill(bbox[:, 0], bbox[:, 1], color=color, alpha=0.5, zorder=zorder)

            if traj_cat[i] == 0:
                ax.fill(bbox[:, 0], bbox[:, 1], color='black', alpha=0.5, zorder=zorder)

        # 2) draw predicted trajectoreis
        num_modes = pred_future.shape[1]
        for agent_index in range(num_agents):
            if (agent_cat[agent_index] != 2): continue
            for mode_index in range(num_modes):  # pred_size = 6 (number of modes)
                try:
                    x_vals = [pred_future[agent_index, mode_index, t, 0].view(-1)[0].item() for t in range(self.num_prediction_steps)]
                    y_vals = [pred_future[agent_index, mode_index, t, 1].view(-1)[0].item() for t in range(self.num_prediction_steps)]
                    ax.plot(x_vals[:-1], y_vals[:-1], linestyle='--', color='green', linewidth=2, zorder=20)

                    offset_pred = pred_future[agent_index, mode_index, 1:] - pred_future[agent_index, mode_index, :-1] # seq_len-1 x 2
                    est_heading = torch.atan2(offset_pred[:, 1], offset_pred[:, 0])
                    est_speed = 3.6 * 10.0 * offset_pred.pow(2).sum(-1).sqrt() # km/h

                    prev_heading, cur_heading = h_theta[agent_index], 0
                    for heading, speed in zip(est_heading, est_speed):
                        if (speed < 5.0):
                            cur_heading = prev_heading
                        else:
                            cur_heading = heading
                        prev_heading = cur_heading

                    arrow_dx = 0.8 * np.cos(cur_heading.item())
                    arrow_dy = 0.8 * np.sin(cur_heading.item())
                    ax.arrow(x_vals[-2], y_vals[-2], arrow_dx, arrow_dy, head_width=1.5, head_length=3.0,
                             fc='green', ec='green', linewidth=2, zorder=20)

                    # if len(x_vals) >= 2:
                    #
                    #     dx = x_vals[-1] - x_vals[-2]
                    #     dy = y_vals[-1] - y_vals[-2]
                    #     speed = np.sqrt(dx ** 2 + dy ** 2)
                    #
                    #     if speed < self.min_speed_threshold:
                    #         last_known_heading = h_theta[agent_index].item() if isinstance(
                    #             h_theta[agent_index], torch.Tensor) else h_theta[agent_index]
                    #         arrow_dx = 0.8 * np.cos(last_known_heading)
                    #         arrow_dy = 0.8 * np.sin(last_known_heading)
                    #     else:
                    #         arrow_dx = dx
                    #         arrow_dy = dy
                    #
                    #     ax.arrow(x_vals[-2], y_vals[-2], arrow_dx, arrow_dy, head_width=1.5, head_length=3.0,
                    #         fc='green', ec='green', linewidth=2, zorder=20)
                except (IndexError, ValueError) as e:
                    continue


        # 3) if the ground-truth is available, draw the ground-truth future
        if isGTAvailable:
            for i, pos_fut in enumerate(pos_fut):
                agent_cat_i = agent_cat[i]
                if agent_cat_i == 2:
                    color = 'purple'
                    self.agent = i
                    zorder = 20

                    x_vals = [pos[0].item() if isinstance(pos[0], torch.Tensor) else pos[0] for pos in pos_fut]
                    y_vals = [pos[1].item() if isinstance(pos[1], torch.Tensor) else pos[1] for pos in pos_fut]
                    ax.plot(x_vals[:-1], y_vals[:-1], linestyle='--', color=color, linewidth=2, zorder=zorder)

                    future_headings = f_angle[i]
                    arrow_dx = 0.8 * np.cos(f_angle[i, -1])
                    arrow_dy = 0.8 * np.sin(f_angle[i, -1])
                    ax.arrow(x_vals[-2], y_vals[-2], arrow_dx, arrow_dy, head_width=1.5, head_length=3.0, fc=color,
                             ec=color, linewidth=2, zorder=zorder)
                    # if len(x_vals) >= 2:
                    #
                    #     dx = x_vals[-1] - x_vals[-2]
                    #     dy = y_vals[-1] - y_vals[-2]
                    #     speed = np.sqrt(dx ** 2 + dy ** 2)
                    #
                    #     if speed < self.min_speed_threshold and len(future_headings) > 0:
                    #         last_heading = future_headings[-1].item() if isinstance(future_headings[-1], torch.Tensor) \
                    #             else future_headings[-1]
                    #
                    #         arrow_dx = 0.8 * np.cos(last_heading)
                    #         arrow_dy = 0.8 * np.sin(last_heading)
                    #     else:
                    #         arrow_dx = dx
                    #         arrow_dy = dy
                    #
                    #     ax.arrow(x_vals[-2], y_vals[-2], arrow_dx, arrow_dy, head_width=1.5, head_length=3.0, fc=color,
                    #              ec=color, linewidth=2, zorder=zorder)




if __name__ == "__main__":

    traj_vis = AV2_vis()

    isGTAvailable = False
    test_data_folder = "/home/dooseop/DATASET/ETRI/av2format/test1/"
    masked_test_data_folder = "/home/dooseop/DATASET/ETRI/av2format/test_masked/"
    masked_qcnet_test_data_folder = "/home/dooseop/DATASET/ETRI/av2format/test_qcnet/"
    submission_folder = "./lightning_logs/version_23/test_results"

    file_names = [f for f in os.listdir(masked_test_data_folder) if f.endswith('.pkl')]
    for file_name in file_names:

        try:
            test_file_name = file_name.replace('_masked', "")
            with open(os.path.join(test_data_folder, test_file_name), 'rb') as file:
                gt_data = pickle.load(file)
            isGTAvailable = True
        except:
            with open(os.path.join(masked_test_data_folder, file_name), 'rb') as file:
                gt_data = pickle.load(file)
            isGTAvailable = False

        qcnet_test_file_name = file_name.replace('_masked', "_masked_qcnet")
        with open(os.path.join(masked_qcnet_test_data_folder, qcnet_test_file_name), 'rb') as file:
            qcnet_data = pickle.load(file)

        sub_test_file_name = file_name.replace('_masked', '_submission')
        with open(os.path.join(submission_folder, sub_test_file_name), 'rb') as file:
            pred_data = pickle.load(file)


        fig, ax = plt.subplots(figsize=(12, 12))
        traj_vis.draw_scene(ax, qcnet_data, pred_data, gt_data, isGTAvailable=isGTAvailable)

        plt.show()