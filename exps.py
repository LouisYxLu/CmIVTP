import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from models import GroupTrajectoryBank

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ================= 辅助函数  =================

def to_device(data, device):
    """将 numpy 或 tensor 转移到指定设备"""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        data = data.float()
    return data.to(device)

def pairwise_l2(pred, target):
    """计算 L2 距离 (欧氏距离)"""
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=-1) + 1e-6)

def expand_targets(target, modes):
    """扩展 target 以匹配多模态预测的维度"""
    return target.unsqueeze(2).repeat(1, 1, modes, 1)

def simulate_ais_missing(traj_tensor, missing_rate, predefined_mask=None):
    """模拟AIS数据缺失"""
    seq_len, n_agents, _ = traj_tensor.shape
    device = traj_tensor.device

    if predefined_mask is not None:
        mask = predefined_mask.to(device)
    else:
        mask = torch.bernoulli(torch.full((n_agents,), 1 - missing_rate, device=device))
        mask = mask.view(1, n_agents, 1)

    ais_mask_expanded = mask.repeat(seq_len, 1, 2)
    masked_traj_tensor = traj_tensor.clone()
    masked_traj_tensor[..., :2] = masked_traj_tensor[..., :2] * ais_mask_expanded

    return masked_traj_tensor, mask

def plot_trajectories(tar_a, pred_a, tar_v, pred_v,
                      ade_a, fde_a, ade_v, fde_v,
                      path, obs_a=None, missing_mask=None, title_prefix=""):
    """绘图函数，兼容多模态预测 (Modes)"""
    plt.figure(figsize=(14, 6)) 

    # --- 绘制 AIS ---
    plt.subplot(1, 2, 1)
    label_flags = {'obs': False, 'gt_norm': False, 'pred_norm': False, 'gt_miss': False, 'pred_miss': False}
    batch_size = tar_a.shape[0]

    for i in range(batch_size):
        if obs_a is not None:
            lbl_obs = 'Observed' if not label_flags['obs'] else ""
            if not label_flags['obs']: label_flags['obs'] = True
            plt.plot(obs_a[i, :, 1], obs_a[i, :, 0], color='black', alpha=0.7, linewidth=1.5, label=lbl_obs)

        is_missing_target = False
        if missing_mask is not None and missing_mask[0, i, 0] == 0:
            is_missing_target = True
        
        c_gt, c_pred = ('cyan', 'magenta') if is_missing_target else ('blue', 'red')
        lbl_gt = 'GT (Missing AIS)' if is_missing_target else 'GT (Normal)'
        lbl_pred = 'Pred (Missing AIS)' if is_missing_target else 'Pred (Normal)'
        key_gt, key_pred = ('gt_miss', 'pred_miss') if is_missing_target else ('gt_norm', 'pred_norm')

        lbl = lbl_gt if not label_flags[key_gt] else ""
        if not label_flags[key_gt]: label_flags[key_gt] = True
        plt.plot(tar_a[i, :, 1], tar_a[i, :, 0], color=c_gt, alpha=0.6, label=lbl)

        lbl = lbl_pred if not label_flags[key_pred] else ""
        if not label_flags[key_pred]: label_flags[key_pred] = True
        
        # 兼容多模态预测
        if pred_a.ndim == 4: # (Batch, Seq, Modes, 2)
            for m in range(pred_a.shape[2]):
                plt.plot(pred_a[i, :, m, 1], pred_a[i, :, m, 0], color=c_pred, alpha=0.3, linestyle='--', label=lbl if m==0 else "")
        else:
            plt.plot(pred_a[i, :, 1], pred_a[i, :, 0], color=c_pred, alpha=0.6, linestyle='--', label=lbl)

    plt.title(f'{title_prefix} AIS Trajectory\nADE: {ade_a:.5f}, FDE: {fde_a:.5f}')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.3)

    # --- 绘制 Video ---
    plt.subplot(1, 2, 2)
    def get_center(data):
        if data.shape[-1] == 4:
            return (data[..., 0] + data[..., 2]) / 2, (data[..., 1] + data[..., 3]) / 2
        return data[..., 1], data[..., 0]

    tar_v_x, tar_v_y = get_center(tar_v)
    pred_v_x, pred_v_y = get_center(pred_v)

    for i in range(batch_size):
        label_obs = 'Observed' if i == 0 else ""
        label_pred = 'Predicted' if i == 0 else ""
        plt.plot(tar_v_y[i], tar_v_x[i], color='blue', alpha=0.5, label=label_obs)
        
        if pred_v.ndim == 4:
            for m in range(pred_v.shape[2]):
                plt.plot(pred_v_y[i, :, m], pred_v_x[i, :, m], color='red', alpha=0.3, label=label_pred if m==0 else "")
        else:
            plt.plot(pred_v_y[i], pred_v_x[i], color='red', alpha=0.5, label=label_pred)

    plt.title(f'Video (Box Center) - ADE: {ade_v:.5f}, FDE: {fde_v:.5f}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ================= 实验控制类 =================

class Exp:
    def __init__(self, args, model, train_data, test_data, scaler, test_masks=None):
        self.args = args
        self.device = torch.device(args['device'])
        self.model = model.to(self.device)
        self.train_data = train_data
        self.test_data = test_data
        self.scaler = scaler
        self.test_masks = test_masks

        self.save_path = args['save_root'] + '/' + args['model_name']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)

        self.best_metrics = {key: {'ade': 100.0, 'fde': 100.0} for key in self.test_data.keys()}
        self.global_best_ade = 100.0

        if hasattr(self.model, 'use_gtb') and self.model.use_gtb:
            gtb_path = os.path.join(self.save_path, 'trajectory_bank_ais.pkl')
            self.model.gtb_ais = GroupTrajectoryBank(n_clusters=16, observed_len=args['observed_len'], pred_len=args['pred_len'])
            if os.path.exists(gtb_path):
                self.model.gtb_ais.load_bank(gtb_path)
            else:
                self.model.gtb_ais.build_bank(self.train_data, save_path=gtb_path)

    def train(self):
        print(f"Start training {self.args['model_name']}...")
        log_file = os.path.join(self.save_path, f"AF-{self.args['model_name']}.txt")

        for ep in range(self.args['epoch']):
            self.model.train()
            train_losses = []
            indices = list(range(len(self.train_data)))
            random.shuffle(indices)

            for idx, D_id in enumerate(indices):
                self.optimizer.zero_grad()
                batch_data = self.train_data[D_id]
                traj_tensor_raw = to_device(batch_data[0], self.device)
                img_tensor = to_device(batch_data[1], self.device)
                bbox_tensor = to_device(batch_data[2], self.device)

                masked_traj, ais_mask = simulate_ais_missing(traj_tensor_raw, self.args['train_missing_rate'])
                obs_len = self.args['observed_len']
                inputs = (masked_traj[:obs_len], img_tensor[:obs_len], bbox_tensor[:obs_len])
                
                pred_a, pred_v, kl_loss = self.model(inputs, ais_mask=ais_mask, target_future=traj_tensor_raw[-self.args['pred_len']:])

                target_future = traj_tensor_raw[-self.args['pred_len']:]
                target_ais = target_future[..., :2]
                target_vid_bbox = target_future[..., 2:] 
                target_vid_center = torch.stack([(target_vid_bbox[..., 0] + target_vid_bbox[..., 2]) / 2, (target_vid_bbox[..., 1] + target_vid_bbox[..., 3]) / 2], dim=-1) if target_vid_bbox.shape[-1] == 4 else target_vid_bbox

                # 兼容多模态预测 (AIS)
                if pred_a.dim() == 4:
                    loss_a = pairwise_l2(pred_a, expand_targets(target_ais, pred_a.shape[2])).min(dim=2).values.mean()
                else:
                    loss_a = pairwise_l2(pred_a, target_ais).mean()

                # 兼容多模态预测 (Video)
                if pred_v.dim() == 4:
                    loss_v = pairwise_l2(pred_v, expand_targets(target_vid_center, pred_v.shape[2])).min(dim=2).values.mean()
                else:
                    loss_v = pairwise_l2(pred_v, target_vid_center).mean()

                loss = loss_a + loss_v + 0.01 * kl_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            log_str_parts = [f"Epoch: {ep}"]
            current_epoch_metrics = {}

            for key in ['High', 'Mid', 'Low']:
                if key not in self.test_data: continue
                dataset_list = self.test_data[key]
                Ade_a, Fde_a, Ade_v, Fde_v = [], [], [], []
                final_pred_a = final_pred_v = final_tar_a = final_tar_v = final_obs_a = final_mask = None

                with torch.no_grad():
                    for T_id in range(len(dataset_list)):
                        batch_data = dataset_list[T_id]
                        traj_tensor_raw = to_device(batch_data[0], self.device)
                        img_tensor = to_device(batch_data[1], self.device)
                        bbox_tensor = to_device(batch_data[2], self.device)
                        
                        masked_traj, ais_mask = simulate_ais_missing(traj_tensor_raw, self.args['train_missing_rate'])
                        inputs = (masked_traj[:self.args['observed_len']], img_tensor[:self.args['observed_len']], bbox_tensor[:self.args['observed_len']])
                        pred_a, pred_v, _ = self.model(inputs, ais_mask=ais_mask)
                        
                        target_future = traj_tensor_raw[-self.args['pred_len']:]
                        target_ais = target_future[..., :2]
                        target_vid_center = torch.stack([(target_future[..., 2:][..., 0] + target_future[..., 2:][..., 2]) / 2, (target_future[..., 2:][..., 1] + target_future[..., 2:][..., 3]) / 2], dim=-1) if target_future[..., 2:].shape[-1] == 4 else target_future[..., 2:]

                        if pred_a.dim() == 4:
                            ade_a_tensor = pairwise_l2(pred_a, expand_targets(target_ais, pred_a.shape[2]))
                            Ade_a.append(ade_a_tensor.min(-1).values.mean().item())
                            Fde_a.append(ade_a_tensor[-1].min(-1).values.mean().item())
                        else:
                            ade_a_tensor = pairwise_l2(pred_a, target_ais)
                            Ade_a.append(ade_a_tensor.mean().item())
                            Fde_a.append(ade_a_tensor[-1].mean().item())
                            
                        if pred_v.dim() == 4:
                            ade_v_tensor = pairwise_l2(pred_v, expand_targets(target_vid_center, pred_v.shape[2]))
                            Ade_v.append(ade_v_tensor.min(-1).values.mean().item())
                            Fde_v.append(ade_v_tensor[-1].min(-1).values.mean().item())
                        else:
                            ade_v_tensor = pairwise_l2(pred_v, target_vid_center)
                            Ade_v.append(ade_v_tensor.mean().item())
                            Fde_v.append(ade_v_tensor[-1].mean().item())

                        if T_id == len(dataset_list) - 1:
                            final_pred_a, final_pred_v = pred_a.detach().cpu().numpy(), pred_v.detach().cpu().numpy()
                            final_tar_a, final_tar_v = target_ais.detach().cpu().numpy(), target_vid_center.detach().cpu().numpy()
                            final_obs_a, final_mask = inputs[0][..., :2].detach().cpu().numpy(), ais_mask.detach().cpu().numpy()

                cur_ade, cur_fde, cur_ade_v, cur_fde_v = np.mean(Ade_a), np.mean(Fde_a), np.mean(Ade_v), np.mean(Fde_v)
                self.best_metrics[key]['ade'] = min(self.best_metrics[key]['ade'], cur_ade)
                self.best_metrics[key]['fde'] = min(self.best_metrics[key]['fde'], cur_fde)
                current_epoch_metrics[key] = cur_ade

                log_str_parts.append(f"{key}-AIS - ADE: {cur_ade:.4f} (best: {self.best_metrics[key]['ade']:.4f}), FDE: {cur_fde:.4f} | Video - ADE: {cur_ade_v:.4f}, FDE: {cur_fde_v:.4f}")

                # 绘图兼容多模态
                if final_pred_a.ndim == 4:
                    traj_ADE_a = np.linalg.norm(final_pred_a.mean(axis=2) - final_tar_a, axis=-1).mean()
                    traj_FDE_a = np.linalg.norm(final_pred_a.mean(axis=2) - final_tar_a, axis=-1)[-1].mean()
                else:
                    traj_ADE_a = np.linalg.norm(final_pred_a - final_tar_a, axis=-1).mean()
                    traj_FDE_a = np.linalg.norm(final_pred_a - final_tar_a, axis=-1)[-1].mean()
                    
                if final_pred_v.ndim == 4:
                    traj_ADE_v = np.linalg.norm(final_pred_v.mean(axis=2) - final_tar_v, axis=-1).mean()
                    traj_FDE_v = np.linalg.norm(final_pred_v.mean(axis=2) - final_tar_v, axis=-1)[-1].mean()
                else:
                    traj_ADE_v = np.linalg.norm(final_pred_v - final_tar_v, axis=-1).mean()
                    traj_FDE_v = np.linalg.norm(final_pred_v - final_tar_v, axis=-1)[-1].mean()
                
                plot_trajectories(final_tar_a.transpose(1, 0, 2), final_pred_a.transpose(1, 0, 2, 3) if final_pred_a.ndim == 4 else final_pred_a.transpose(1, 0, 2),
                                  final_tar_v.transpose(1, 0, 2), final_pred_v.transpose(1, 0, 2, 3) if final_pred_v.ndim == 4 else final_pred_v.transpose(1, 0, 2),
                                  traj_ADE_a, traj_FDE_a, traj_ADE_v, traj_FDE_v,
                                  os.path.join(self.save_path, f'pred_{ep}_{key}.png'), obs_a=final_obs_a.transpose(1, 0, 2), missing_mask=final_mask, title_prefix=f"[{key}] ")

            full_log_msg = ", ".join(log_str_parts)
            print(full_log_msg)
            with open(log_file, 'a', encoding='utf-8') as f: f.write(full_log_msg + '\n')

            self.scheduler.step(np.mean(train_losses) if train_losses else 0.0)
            main_key = 'High' if 'High' in current_epoch_metrics else list(current_epoch_metrics.keys())[0]
            if current_epoch_metrics[main_key] < self.global_best_ade:
                self.global_best_ade = current_epoch_metrics[main_key]
                torch.save({'state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, os.path.join(self.save_path, f'{self.args["model_name"]}_best_model.tar'))

    def test(self):
        print(f"Running prediction for {self.args['model_name']}...")
        model_path = os.path.join(self.save_path, f'{self.args["model_name"]}_best_model.tar')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['state_dict'])
            print("Loaded best model.")
        
        self.model.eval()
        for key, dataset_list in self.test_data.items():
            if not dataset_list: continue
            print(f"--- Testing on {key} Dataset ---")
            indices_to_test = range(len(dataset_list)) if len(dataset_list) <= 10 else np.linspace(0, len(dataset_list) - 1, 10, dtype=int)

            for save_idx, data_idx in enumerate(indices_to_test):
                batch_data = dataset_list[data_idx]
                traj_tensor_raw = to_device(batch_data[0], self.device)
                img_tensor = to_device(batch_data[1], self.device)
                bbox_tensor = to_device(batch_data[2], self.device)
                
                masked_Real, ais_mask = simulate_ais_missing(traj_tensor_raw, self.args['train_missing_rate'])
                inputs = (masked_Real[:self.args['observed_len']], img_tensor[:self.args['observed_len']], bbox_tensor[:self.args['observed_len']])

                with torch.no_grad():
                    Pred_a, Pred_v, _ = self.model(inputs, ais_mask=ais_mask)

                future = traj_tensor_raw[-self.args['pred_len']:]
                target_a = future[..., :2]
                target_v = future[..., 2:]
                target_v_center = torch.stack([(target_v[..., 0] + target_v[..., 2]) / 2, (target_v[..., 1] + target_v[..., 3]) / 2], dim=-1) if target_v.shape[-1] == 4 else target_v

                if Pred_a.dim() == 4:
                    ade_a_tensor = pairwise_l2(Pred_a, expand_targets(target_a, Pred_a.shape[2]))
                    traj_ADE_a = ade_a_tensor.min(-1).values.mean().item()
                    traj_FDE_a = ade_a_tensor[-1].min(-1).values.mean().item()
                else:
                    ade_a_tensor = pairwise_l2(Pred_a, target_a)
                    traj_ADE_a = ade_a_tensor.mean().item()
                    traj_FDE_a = ade_a_tensor[-1].mean().item()

                if Pred_v.dim() == 4:
                    ade_v_tensor = pairwise_l2(Pred_v, expand_targets(target_v_center, Pred_v.shape[2]))
                    traj_ADE_v = ade_v_tensor.min(-1).values.mean().item()
                    traj_FDE_v = ade_v_tensor[-1].min(-1).values.mean().item()
                else:
                    ade_v_tensor = pairwise_l2(Pred_v, target_v_center)
                    traj_ADE_v = ade_v_tensor.mean().item()
                    traj_FDE_v = ade_v_tensor[-1].mean().item()
                
                print(f'[{key} - Sample {save_idx}] AIS ADE: {traj_ADE_a:.4f}, Video ADE: {traj_ADE_v:.4f}')

                prefix = f"{self.args['model_name']}_{key}_sample_{save_idx}"
                np.save(os.path.join(self.save_path, f'{prefix}_Preds_AIS.npy'), Pred_a.detach().cpu().numpy())
                np.save(os.path.join(self.save_path, f'{prefix}_Preds_Video.npy'), Pred_v.detach().cpu().numpy())
                np.save(os.path.join(self.save_path, f'{prefix}_Reals.npy'), traj_tensor_raw.detach().cpu().numpy())
                np.save(os.path.join(self.save_path, f'{prefix}_Obs_AIS.npy'), inputs[0][..., :2].detach().cpu().numpy())
                np.save(os.path.join(self.save_path, f'{prefix}_Mask.npy'), ais_mask.detach().cpu().numpy())

                plot_trajectories(
                    target_a.detach().cpu().numpy().transpose(1, 0, 2), 
                    Pred_a.detach().cpu().numpy().transpose(1, 0, 2, 3) if Pred_a.dim() == 4 else Pred_a.detach().cpu().numpy().transpose(1, 0, 2),
                    target_v_center.detach().cpu().numpy().transpose(1, 0, 2), 
                    Pred_v.detach().cpu().numpy().transpose(1, 0, 2, 3) if Pred_v.dim() == 4 else Pred_v.detach().cpu().numpy().transpose(1, 0, 2),
                    traj_ADE_a, traj_FDE_a, traj_ADE_v, traj_FDE_v,
                    os.path.join(self.save_path, f'{prefix}_vis.png'), 
                    obs_a=inputs[0][..., :2].detach().cpu().numpy().transpose(1, 0, 2), 
                    missing_mask=ais_mask.detach().cpu().numpy(), title_prefix=f"[{key}-{save_idx}] "
                )