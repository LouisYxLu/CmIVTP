import pandas as pd
import numpy as np
from datetime import datetime
import os
import cv2
from tqdm import tqdm
from torchvision import transforms

IMG_SIZE = 224
EPS = 1e-6

class StandardScaler:
    def __init__(self, ranges, original_img_size=(1440, 2560)):
        """
        使用固定的物理边界进行 Min-Max 归一化
        ranges: 传入 AREA_CONFIGS 中的 ranges 字典
        """
        self.original_h, self.original_w = original_img_size
        
        # 根据 trajectory_cols = ['lon', 'lat', 'xmid', 'ymid'] 的顺序设置 min 和 max
        self.min_vals = np.array([
            ranges['lon'][0], ranges['lat'][0], ranges['x'][0], ranges['y'][0]
        ], dtype=np.float32)
        
        self.max_vals = np.array([
            ranges['lon'][1], ranges['lat'][1], ranges['x'][1], ranges['y'][1]
        ], dtype=np.float32)
        
        self.eps = EPS
        
        # 图像的归一化保持不变
        self.trans_pic = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def trans(self, data):
        """
        纯 Min-Max 归一化: (data - min) / (max - min)
        data 形状应为 [..., 4] 对应 ['lon', 'lat', 'xmid', 'ymid']
        """
        scale = (self.max_vals - self.min_vals) + self.eps
        data = (data - self.min_vals) / scale
        return data

    def inverse_trans(self, data):
        """
        反归一化，将 [0, 1] 的数据还原为真实的经纬度和像素坐标
        """
        scale = (self.max_vals - self.min_vals)
        return data * scale + self.min_vals


AREA_CONFIGS = {
    'Area-I': {
        'train_path': './data/Area-I/Train/Area-I-train.csv',
        'test_paths': {
            'High': './data/Area-I/Test-High/Area-I-test-high.csv',
            'Mid':  './data/Area-I/Test-Mid/Area-I-test-mid.csv',
            'Low':  './data/Area-I/Test-Low/Area-I-test-low.csv'
        },
        'img_root': './data/Area-I/images',
        'ranges': {
            'lon': (121.50, 121.60),
            'lat': (31.24, 31.26),
            'x': (0.0, 2560.0),
            'y': (0.0, 1440.0)
        }
    },
    'Area-II': {
        'train_path': './data/Area-II/Train/Area-II-train.csv',
        'test_paths': {
            'High': './data/Area-II/Test-High/Area-II-test-high.csv',
            'Mid':  './data/Area-II/Test-Mid/Area-II-test-mid.csv',
            'Low':  './data/Area-II/Test-Low/Area-II-test-low.csv'
        },
        'img_root': './data/Area-II/images',
        'ranges': {
            'lon': (121.48, 121.71),
            'lat': (31.18, 31.24),
            'x': (0.0, 2560.0),
            'y': (0.0, 1440.0)
        }
    },
    'Area-III': {
        'train_path': './data/Area-III/Train/Area-III-train.csv',
        'test_paths': {
            'High': './data/Area-III/Test-High/Area-III-test-high.csv',
            'Mid':  './data/Area-III/Test-Mid/Area-III-test-mid.csv',
            'Low':  './data/Area-III/Test-Low/Area-III-test-low.csv'            
        },
        'img_root': './data/Area-III/images',
        'ranges': {
            'lon': (121.48, 121.51),
            'lat': (31.23, 31.26),
            'x': (0.0, 2560.0),
            'y': (0.0, 1440.0)
        }
    },
    'Area-IV': {
        'train_path': './data/Area-IV/Train/Area-IV-train.csv',
        'test_paths': {
            'High': './data/Area-IV/Test-High/Area-IV-test-high.csv',
            'Mid':  './data/Area-IV/Test-Mid/Area-IV-test-mid.csv',
            'Low':  './data/Area-IV/Test-Low/Area-IV-test-low.csv'            
        },
        'img_root': './data/Area-IV/images',
        'ranges': {
            'lon': (121.48, 121.72),
            'lat': (31.10, 32.10),
            'x': (0.0, 2560.0),
            'y': (0.0, 1440.0)
        }
    }
}


def build_batches(df, config, img_root, scaler):
    trajectory_cols = ['lon', 'lat', 'xmid', 'ymid']
    bbox_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    
    # Calculate xmid, ymid if not present
    if 'xmid' not in df.columns: df['xmid'] = (df['xmin'] + df['xmax']) / 2
    if 'ymid' not in df.columns: df['ymid'] = (df['ymin'] + df['ymax']) / 2

    # 使用固定的 scaler 进行归一化，不再需要 fit
    df_traj = df[trajectory_cols].values.astype('float32')
    df[trajectory_cols] = scaler.trans(df_traj)

    S, E = df['time'].iloc[0], df['time'].iloc[-1]
    seq_len = config['pred_len'] + config['observed_len']
    samples = int((E - S) / config['insert_inver'])

    df['picid'] = [datetime.fromtimestamp(t).strftime("%Y-%m-%d-%H-%M-%S") for t in df['time']]
    Pics = {pic: scaler.trans_pic(cv2.imread(os.path.join(img_root, f'{pic}.jpg'))).numpy() if os.path.exists(os.path.join(img_root, f'{pic}.jpg')) else np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32) for pic in tqdm(df['picid'].unique(), desc='Images')}

    Batchs_traj, Batchs_img, Batchs_bbox = [], [], []
    for i in range(samples - seq_len):
        start_time = S + i * config['insert_inver']
        data_son = df[(df['time'] >= start_time) & (df['time'] < start_time + seq_len * config['insert_inver'])]
        window_times = sorted(data_son['time'].unique())
        if len(window_times) < seq_len: continue

        batch_traj, batch_bbox = [], []
        for sh in set(data_son['mmsi']):
            tra = data_son[data_son['mmsi'] == sh]
            if tra.shape[0] >= seq_len:
                batch_traj.append(tra.iloc[:seq_len][trajectory_cols].values.astype('float32'))
                batch_bbox.append(tra.iloc[:seq_len][bbox_cols].values.astype('float32'))
        
        if batch_traj:
            Batchs_traj.append(np.stack(batch_traj, axis=1))
            Batchs_bbox.append(np.stack(batch_bbox, axis=1))
            Batchs_img.append(np.stack([Pics.get(datetime.fromtimestamp(t).strftime("%Y-%m-%d-%H-%M-%S"), np.zeros((3, IMG_SIZE, IMG_SIZE))) for t in window_times[:seq_len]]).astype('float32'))

    return list(zip(Batchs_traj, Batchs_img, Batchs_bbox))


def get_dataloader(config, area_name):
    area_cfg = AREA_CONFIGS[area_name]
    
    # 传入 ranges 初始化 scaler
    scaler = StandardScaler(ranges=area_cfg['ranges'])
    
    df_train = pd.read_csv(area_cfg['train_path'])
    df_train['time'] = [datetime.strptime(t, "%Y-%m-%d-%H-%M-%S").timestamp() for t in df_train.iloc[:, 0]]
    train_batches = build_batches(df_train, config, area_cfg['img_root'], scaler)

    test_batches_dict = {}
    for key, path in area_cfg['test_paths'].items():
        df_test = pd.read_csv(path)
        df_test['time'] = [datetime.strptime(t, "%Y-%m-%d-%H-%M-%S").timestamp() for t in df_test.iloc[:, 0]]
        test_batches_dict[key] = build_batches(df_test, config, area_cfg['img_root'], scaler)

    return train_batches, test_batches_dict, scaler