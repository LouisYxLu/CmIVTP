import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle
from torchvision import models
from torchvision.ops import RoIAlign
from sklearn.cluster import KMeans
from tqdm import tqdm

IMG_SIZE = 224
original_img_height = 1440
original_img_width = 2560
EPS = 1e-6
BANK_FEATURE_VERSION = 'shape_v2_relstart_scale_medoid'

class GroupTrajectoryBank:
    def __init__(self, n_clusters=16, observed_len=36, pred_len=36, feature_version=BANK_FEATURE_VERSION):
        self.n_clusters = n_clusters
        self.observed_len = observed_len
        self.pred_len = pred_len
        self.trajectory_len = observed_len + pred_len
        self.feature_version = feature_version

        self.bank_observed = None
        self.bank_future = None
        self.kmeans = None
        self.cluster_members = None
        self.bank_features = None

    def _extract_motion_feature(self, traj):
        traj = np.asarray(traj, dtype=np.float32).copy()
        if not np.isfinite(traj).all(): return np.zeros((traj.shape[0] * 2,), dtype=np.float32)
        traj = traj - traj[0:1]
        disp = traj[-1] - traj[0]
        scale = np.linalg.norm(disp)
        if scale < 1e-4: scale = np.linalg.norm(traj.std(axis=0))
        if scale < 1e-4: scale = 1.0
        traj = traj / scale
        return traj.reshape(-1).astype(np.float32)

    def build_bank(self, train_data, save_path=None):
        print("Building Group Trajectory Bank...")
        all_trajectories = []
        for batch_tuple in tqdm(train_data, desc="Extracting trajectories"):
            batch = np.asarray(batch_tuple[0], dtype=np.float32)
            if batch.ndim != 3 or batch.shape[0] < self.trajectory_len: continue
            n_agents = batch.shape[1]
            for agent_idx in range(n_agents):
                trajectory = batch[:self.trajectory_len, agent_idx, :2].astype(np.float32)
                if np.isfinite(trajectory).all(): all_trajectories.append(trajectory)

        if not all_trajectories:
            print("Warning: No valid trajectories found!")
            return

        all_trajectories = np.asarray(all_trajectories, dtype=np.float32)
        observed_trajectories = all_trajectories[:, :self.observed_len, :]
        future_trajectories = all_trajectories[:, self.observed_len:, :]

        observed_features = np.stack([self._extract_motion_feature(traj) for traj in observed_trajectories], axis=0)
        actual_clusters = min(self.n_clusters, len(observed_features))
        
        self.kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=20, max_iter=500)
        cluster_labels = self.kmeans.fit_predict(observed_features)

        self.bank_observed, self.bank_future, self.cluster_members, self.bank_features = [], [], [], []

        for cluster_id in range(actual_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) == 0: continue
            cluster_feats = observed_features[cluster_mask]
            feat_center = cluster_feats.mean(axis=0, keepdims=True)
            rep_idx = int(np.argmin(np.linalg.norm(cluster_feats - feat_center, axis=1)))

            self.bank_observed.append(observed_trajectories[cluster_mask][rep_idx])
            self.bank_future.append(future_trajectories[cluster_mask][rep_idx])
            self.cluster_members.append(all_trajectories[cluster_mask])
            self.bank_features.append(cluster_feats[rep_idx])

        self.bank_observed = np.asarray(self.bank_observed, dtype=np.float32)
        self.bank_future = np.asarray(self.bank_future, dtype=np.float32)
        self.bank_features = np.asarray(self.bank_features, dtype=np.float32)
        if save_path: self.save_bank(save_path)

    def search_trajectory(self, observed_traj):
        if self.bank_observed is None or len(self.bank_future) == 0:
            return np.zeros((self.pred_len, observed_traj.shape[1], 2), dtype=np.float32)
        predicted_futures = []
        for agent_idx in range(observed_traj.shape[1]):
            agent_obs = observed_traj[:, agent_idx, :].astype(np.float32)
            if not np.isfinite(agent_obs).all():
                predicted_futures.append(np.zeros((self.pred_len, 2), dtype=np.float32))
                continue
            query_feat = self._extract_motion_feature(agent_obs)
            query_norm = np.linalg.norm(query_feat) + 1e-8
            bank_norms = np.linalg.norm(self.bank_features, axis=1) + 1e-8
            sims = np.dot(self.bank_features, query_feat) / (bank_norms * query_norm)
            predicted_futures.append(self.bank_future[int(np.argmax(sims))])
        return np.stack(predicted_futures, axis=1).astype(np.float32)

    def search_trajectory_with_info(self, observed_traj):
        if self.bank_observed is None or len(self.bank_future) == 0:
            n_agents = observed_traj.shape[1]
            return (
                np.zeros((self.pred_len, n_agents, 2), dtype=np.float32),
                np.full((n_agents,), -1, dtype=np.int64),
                np.full((n_agents,), -1.0, dtype=np.float32)
            )

        n_agents = observed_traj.shape[1]
        predicted_futures = []
        best_indices = []
        best_similarities = []

        for agent_idx in range(n_agents):
            agent_obs = observed_traj[:, agent_idx, :].astype(np.float32)
            if not np.isfinite(agent_obs).all():
                predicted_futures.append(np.zeros((self.pred_len, 2), dtype=np.float32))
                best_indices.append(-1)
                best_similarities.append(-1.0)
                continue
            
            query_feat = self._extract_motion_feature(agent_obs)
            query_norm = np.linalg.norm(query_feat) + 1e-8
            bank_norms = np.linalg.norm(self.bank_features, axis=1) + 1e-8
            sims = np.dot(self.bank_features, query_feat) / (bank_norms * query_norm)
            
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            
            predicted_futures.append(self.bank_future[best_idx])
            best_indices.append(best_idx)
            best_similarities.append(best_sim)

        return (
            np.stack(predicted_futures, axis=1).astype(np.float32),
            np.asarray(best_indices, dtype=np.int64),
            np.asarray(best_similarities, dtype=np.float32)
        )

    def get_cluster_statistics(self):
        if self.cluster_members is None:
            return None
        return [len(members) if members is not None else 0 for members in self.cluster_members]

    def save_bank(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'bank_observed': self.bank_observed, 'bank_future': self.bank_future,
                         'bank_features': self.bank_features, 'n_clusters': self.n_clusters,
                         'observed_len': self.observed_len, 'pred_len': self.pred_len,
                         'cluster_members': self.cluster_members,
                         'feature_version': self.feature_version}, f)

    def load_bank(self, path):
        with open(path, 'rb') as f: data = pickle.load(f)
        self.bank_observed = data['bank_observed']
        self.bank_future = data['bank_future']
        self.bank_features = data.get('bank_features', None)
        self.n_clusters = data['n_clusters']
        self.observed_len = data['observed_len']
        self.pred_len = data['pred_len']
        self.cluster_members = data.get('cluster_members', None)
        self.feature_version = data.get('feature_version', 'unknown')

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, tuple) else kernel_size // 2
        self.conv = nn.Conv2d(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, self.kernel_size, padding=self.padding, bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        c_next = torch.sigmoid(cc_f) * c_cur + torch.sigmoid(cc_i) * torch.tanh(cc_g)
        h_next = torch.sigmoid(cc_o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, return_all_layers=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.cell_list = nn.ModuleList([
            ConvLSTMCell(self.input_dim if i == 0 else self.hidden_dim[i - 1],
                         self.hidden_dim[i], self.kernel_size[i], True)
            for i in range(self.num_layers)
        ])

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first: input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = [cell.init_hidden(b, (h, w)) for cell in self.cell_list]
        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(input_tensor.size(1)):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], [h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        if not self.batch_first:
            layer_output_list = [layer.permute(1, 0, 2, 3, 4) for layer in layer_output_list]
        return layer_output_list, last_state_list

class TargetAwareCNN(nn.Module):
    def __init__(self, feature_dim=256, roi_size=7):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.roi_align = RoIAlign(output_size=(roi_size, roi_size), spatial_scale=1.0 / 32.0, sampling_ratio=2)
        self.target_encoder = nn.Sequential(nn.Conv2d(512, 256, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, feature_dim))
        self.bbox_encoder = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64))
        self.global_encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, feature_dim // 2))
        self.fusion = nn.Sequential(nn.Linear(feature_dim + 64 + feature_dim // 2, feature_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(feature_dim, feature_dim))

    def forward(self, images, bboxes):
        batch_size, n_agents = images.shape[0], bboxes.shape[1]
        feature_maps = self.backbone(images)
        roi_boxes = torch.cat([torch.full((n_agents, 1), b, device=images.device) for b in range(batch_size)], dim=0)
        roi_boxes = torch.cat([roi_boxes, bboxes.view(-1, 4)], dim=1)
        roi_features = self.roi_align(feature_maps, roi_boxes)
        target_features = self.target_encoder(roi_features).view(batch_size, n_agents, -1)
        bbox_features = self.bbox_encoder(bboxes)
        global_features = self.global_encoder(feature_maps).unsqueeze(1).repeat(1, n_agents, 1)
        return self.fusion(torch.cat([target_features, bbox_features, global_features], dim=-1)), feature_maps

class CmIT(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.05):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            activation='gelu'
        )

    def forward(self, feat1, feat2):
        return self.decoder_layer(feat1, feat2)

class MultiModes(nn.Module):
    def __init__(self, hidden_dim, feat_out, pred_len, K_modes, latent_dim):
        super().__init__()
        self.hidden_dim, self.feat_out, self.pred_len, self.K, self.latent_dim = hidden_dim, feat_out, pred_len, K_modes, latent_dim
        self.mode_emb = nn.Embedding(self.K, hidden_dim)
        self.motion_encoder = nn.GRU(4, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.posterior_mu = nn.Linear(hidden_dim * 3, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim * 3, latent_dim)
        self.future_encoder = nn.GRU(feat_out, hidden_dim)
        self.pro1 = nn.Sequential(nn.Linear(hidden_dim * 2 + latent_dim, hidden_dim), nn.ReLU())
        
        self.mlp_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, pred_len * hidden_dim)
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)

    def decode_with_z(self, out, obs_traj, Z):
        n_agents = out.shape[1]
        vel = torch.cat([torch.zeros_like(obs_traj[:1]), obs_traj[1:] - obs_traj[:-1]], dim=0)
        acc = torch.cat([torch.zeros_like(vel[:1]), vel[1:] - vel[:-1]], dim=0)
        _, motion_h = self.motion_encoder(torch.cat([vel, acc], dim=-1))
        fused_features = out.mean(dim=0) + motion_h.squeeze(0)
        
        dec_in = fused_features.unsqueeze(1).repeat(1, self.K, 1)
        m_emb = self.mode_emb(torch.arange(self.K, device=out.device).unsqueeze(0).expand(n_agents, self.K))
        
        forward_h = self.pro1(torch.cat([dec_in, Z, m_emb], dim=-1)).view(-1, self.hidden_dim)
        
        decoded = self.mlp_decoder(forward_h) 
        decoded = decoded.view(-1, self.pred_len, self.hidden_dim) 
        decoded = decoded.permute(1, 0, 2).contiguous() 
        outputs = decoded.view(self.pred_len, n_agents, self.K, self.hidden_dim)
        
        return outputs

    def forward(self, out, obs_traj, target_future=None):
        n_agents = out.shape[1]
        vel = torch.cat([torch.zeros_like(obs_traj[:1]), obs_traj[1:] - obs_traj[:-1]], dim=0)
        acc = torch.cat([torch.zeros_like(vel[:1]), vel[1:] - vel[:-1]], dim=0)
        _, motion_h = self.motion_encoder(torch.cat([vel, acc], dim=-1))
        
        fused_features = out.mean(dim=0) + motion_h.squeeze(0)
        dec_in = fused_features.unsqueeze(1).repeat(1, self.K, 1)
        m_emb = self.mode_emb(torch.arange(self.K, device=out.device).unsqueeze(0).expand(n_agents, self.K))
        
        p_mu = self.prior_mu(torch.cat([dec_in, m_emb], dim=-1))
        p_logvar = self.prior_logvar(torch.cat([dec_in, m_emb], dim=-1))

        if target_future is not None:
            _, future_h = self.future_encoder(target_future)
            q_mu = self.posterior_mu(torch.cat([dec_in, future_h.squeeze(0).unsqueeze(1).repeat(1, self.K, 1), m_emb], dim=-1))
            q_logvar = self.posterior_logvar(torch.cat([dec_in, future_h.squeeze(0).unsqueeze(1).repeat(1, self.K, 1), m_emb], dim=-1))
            Z = self.reparameterize(q_mu, q_logvar)
            kl_loss = -0.5 * torch.sum(1 + q_logvar - p_logvar - ((q_mu - p_mu).pow(2) + q_logvar.exp()) / (p_logvar.exp() + EPS), dim=-1).mean()
        else:
            Z = self.reparameterize(p_mu, p_logvar)
            kl_loss = torch.tensor(0.0, device=out.device)

        return self.decode_with_z(out, obs_traj, Z), kl_loss, Z, p_mu, p_logvar

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x): return x + self.pe[:x.size(0)]

def scale_bboxes_to_224(bbox_tensor):
    if bbox_tensor is None: return None
    scale_w, scale_h = IMG_SIZE / original_img_width, IMG_SIZE / original_img_height
    bbox_scaled = bbox_tensor.clone()
    bbox_scaled[..., [0, 2]] *= scale_w
    bbox_scaled[..., [1, 3]] *= scale_h
    return bbox_scaled

class ProposedModel(nn.Module):
    def __init__(self, feats_in, feats_out, feats_hidden, layers, pred_len, use_gtb=True, gtb_ais=None, use_bbox=True, latent_dim=16):
        super().__init__()
        self.pred_len = pred_len
        self.feats_hidden = feats_hidden
        self.use_gtb = use_gtb
        self.gtb_ais = gtb_ais
        self.use_bbox = use_bbox
        self.K = 5 
        self.latent_dim = latent_dim

        self.ais_encoder = nn.Sequential(
            nn.Linear(3, feats_hidden // 2),
            nn.GELU(),
            nn.Linear(feats_hidden // 2, feats_hidden)
        )
        self.lin2 = nn.Linear(2, feats_hidden)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(feats_hidden)
        self.ln2 = nn.LayerNorm(feats_hidden)
        self.target_cnn = TargetAwareCNN(feature_dim=feats_hidden)
        self.temporal_convlstm = ConvLSTM(input_dim=512, hidden_dim=[256, 128], kernel_size=(3, 3), num_layers=2)
        self.temporal_encoder = nn.Sequential(nn.Conv2d(128, feats_hidden, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.ffn1 = nn.Sequential(nn.Linear(feats_hidden * 2, feats_hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(feats_hidden, feats_hidden), nn.LayerNorm(feats_hidden))
        
        if self.use_gtb:
            self.gtb_refine_ais = nn.Sequential(nn.Linear(feats_out + feats_hidden, feats_hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(feats_hidden, feats_hidden // 2), nn.ReLU(), nn.Linear(feats_hidden // 2, feats_out))
            
            # ==========================================
            # 【新增】: GTB 动态门控网络 (Dynamic Gating Network)
            # 作用：根据智能体的融合特征自适应学习权重 alpha
            # ==========================================
            self.gating_net = nn.Sequential(
                nn.Linear(feats_hidden, feats_hidden // 4),
                nn.ReLU(),
                nn.Linear(feats_hidden // 4, 1),
                nn.Sigmoid() # 保证输出在 [0, 1] 之间
            )

        self.cmit1 = CmIT(d_model=feats_hidden, nhead=8, dropout=0.1)
        self.cmit2 = CmIT(d_model=feats_hidden, nhead=8, dropout=0.1)
        self.emb1, self.emb2, self.emb3 = PositionalEncoding(feats_hidden), PositionalEncoding(feats_hidden), PositionalEncoding(feats_hidden)
        self.lin3_a = nn.Linear(feats_hidden, feats_out)
        self.lin3_v = nn.Linear(feats_hidden, feats_out)
        self.multi_modes = MultiModes(feats_hidden, feats_out, pred_len, self.K, self.latent_dim)

    def forward(self, inputs, ais_mask=None, target_future=None):
        trajectory_data, images, bboxes = inputs[0], inputs[1], inputs[2] if len(inputs) > 2 and self.use_bbox else None
        if bboxes is not None: bboxes = scale_bboxes_to_224(bboxes)

        ais = trajectory_data[:, :, :2]
        vid_raw = trajectory_data[:, :, 2:] 
        vid = torch.stack([(vid_raw[..., 0] + vid_raw[..., 2]) / 2, (vid_raw[..., 1] + vid_raw[..., 3]) / 2], dim=-1) if vid_raw.shape[-1] == 4 else vid_raw

        obs_len, n_agents = ais.shape[:2]

        last_obs_vid = vid[-1]
        last_obs_ais = ais[-1] * ais_mask.squeeze(0) + last_obs_vid * (1 - ais_mask.squeeze(0)) if ais_mask is not None else ais[-1]
        anchor_a = last_obs_ais.unsqueeze(0).unsqueeze(2)
        anchor_v = last_obs_vid.unsqueeze(0).unsqueeze(2)

        if ais_mask is not None:
            mask_feat = ais_mask[..., 0:1].expand(obs_len, n_agents, 1)
        else:
            mask_feat = torch.ones_like(ais[..., 0:1])
        ais_with_mask = torch.cat([ais, mask_feat], dim=-1)
        
        x_a = self.emb1(self.ln1(self.ais_encoder(ais_with_mask)))
        x_v = self.emb2(self.ln2(self.relu(self.lin2(vid))))
        
        # 优化：移除低效的 for 循环，直接利用 TargetAwareCNN 内部的 batch 机制进行并行提取
        # images 形状为 [obs_len, C, H, W]，bboxes 形状为 [obs_len, n_agents, 4]
        image_features_list, all_feature_maps = self.target_cnn(images, bboxes)
        
        # 将所有时间步的特征图输入到 ConvLSTM
        lstm_out, _ = self.temporal_convlstm(all_feature_maps.unsqueeze(1))
        lstm_features = lstm_out[0].squeeze(1) # [obs_len, 128, H', W']

        # 优化（续）：移除 temporal_encoder 的 for 循环，进行向量化处理
        # 保持原本硬编码的 -0.1 衰减率
        time_weights = torch.exp(-0.1 * torch.arange(obs_len-1, -1, -1, device=ais.device, dtype=torch.float32)).unsqueeze(1)
        
        # 批量提取时序特征并应用时间衰减
        t_feats = self.temporal_encoder(lstm_features) * time_weights
        temporal_image_features = t_feats.unsqueeze(1).repeat(1, n_agents, 1)

        # 特征融合
        image_features = self.emb3(self.ffn1(torch.cat((image_features_list, temporal_image_features), dim=2)))
        
        x_v = self.cmit1(x_v, image_features)
        out = self.cmit2(x_a, x_v)
        
        out_features, kl_loss, _, _, _ = self.multi_modes(out, ais, target_future[..., :2] if target_future is not None else None)

        out_a_base = self.lin3_a(out_features)[-self.pred_len:] + anchor_a
        out_v_base = self.lin3_v(out_features)[-self.pred_len:] + anchor_v
        out_a = out_a_base
        out_v = out_v_base

        if self.use_gtb and self.gtb_ais is not None:
            try:
                gtb_pred_ais = torch.tensor(self.gtb_ais.search_trajectory(ais.detach().cpu().numpy()), dtype=torch.float32, device=ais.device)
                if out_a_base.dim() == 4:
                    gtb_pred_ais_expanded = gtb_pred_ais.unsqueeze(2).repeat(1, 1, self.multi_modes.K, 1)
                    combined_ais = torch.cat([gtb_pred_ais_expanded.reshape(self.pred_len, -1, 2), out_features[-self.pred_len:].reshape(self.pred_len, -1, self.feats_hidden)], dim=-1)
                    offset_ais = self.gtb_refine_ais(combined_ais).reshape(self.pred_len, n_agents, self.multi_modes.K, -1)
                    
                    # 精炼后的 GTB 预测轨迹
                    gtb_refined = gtb_pred_ais_expanded + 0.5 * offset_ais
                    
                    # ==========================================
                    # 【修改】: 动态计算融合权重 alpha
                    # ==========================================
                    # 提取智能体在时间维度上的平均上下文特征 [n_agents, feats_hidden]
                    agent_context = out.mean(dim=0) 
                    
                    # 通过门控网络计算权重 alpha [n_agents, 1]
                    alpha = self.gating_net(agent_context)
                    
                    # 扩展 alpha 维度以匹配轨迹张量 [pred_len, n_agents, K, 2]
                    # 扩展后形状为 [1, n_agents, 1, 1]
                    alpha = alpha.unsqueeze(0).unsqueeze(-1)
                    
                    # 动态融合：网络基础预测占 (1 - alpha)，GTB预测占 alpha
                    out_a = (1.0 - alpha) * out_a_base + alpha * gtb_refined
            except Exception: pass
        
        return out_a, out_v, kl_loss