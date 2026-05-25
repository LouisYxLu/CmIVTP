import torch
import dataset
from exps import Exp
from models import ProposedModel
import gc
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AREAS = ['Area-I','Area-II','Area-III','Area-IV']

CONFIGS = [
    # pred_len = 12
    {'pred_len': 12, 'observed_len': 12, 'train_missing_rate': 0.0},
    {'pred_len': 12, 'observed_len': 12, 'train_missing_rate': 0.1},
    {'pred_len': 12, 'observed_len': 12, 'train_missing_rate': 0.2},
    {'pred_len': 12, 'observed_len': 12, 'train_missing_rate': 0.3},

    # pred_len = 24
    {'pred_len': 24, 'observed_len': 24, 'train_missing_rate': 0.0},
    {'pred_len': 24, 'observed_len': 24, 'train_missing_rate': 0.1},
    {'pred_len': 24, 'observed_len': 24, 'train_missing_rate': 0.2},
    {'pred_len': 24, 'observed_len': 24, 'train_missing_rate': 0.3},

    # pred_len = 36
    {'pred_len': 36, 'observed_len': 36, 'train_missing_rate': 0.0},
    {'pred_len': 36, 'observed_len': 36, 'train_missing_rate': 0.1},
    {'pred_len': 36, 'observed_len': 36, 'train_missing_rate': 0.2},
    {'pred_len': 36, 'observed_len': 36, 'train_missing_rate': 0.3},
]


def get_models(pred_len, observed_len):
    fi = 4 
    fo = 2
    fh = 256
    ly = 1
    return [
        ('Proposed', ProposedModel(fi, fo, fh, ly, pred_len, use_gtb=True, use_bbox=True)), 
    ]

def run_experiment(area_name, cfg):
    args = {
        'save_root': f'./results_{area_name}',
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'pred_len': cfg['pred_len'],
        'observed_len': cfg['observed_len'],
        'insert_inver': 5,
        'epoch': 50,
        'lr': 0.0001,
        'train_missing_rate': cfg['train_missing_rate'],
        'model_name': 'Proposed'
    }

    print(f"\n{'='*80}")
    print(f"Running | Area: {area_name} | Pred_len: {cfg['pred_len']} | Missing: {cfg['train_missing_rate']}")
    print(f"{'='*80}")

    train_data, test_data_dict, scaler = dataset.get_dataloader(args, area_name=area_name)

    models_to_run = get_models(args['pred_len'], args['observed_len'])

    for name, model in models_to_run:
        print(f"\n{'-'*30} Training ({name}) {'-'*30}")
        args['model_name'] = name
        model = model.to(device)

        exp = Exp(args, model, train_data, test_data_dict, scaler)
        
        # 直接调用 train
        exp.train()
        
        del exp, model
        torch.cuda.empty_cache()
        gc.collect()

    del train_data, test_data_dict, scaler
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    total_experiments = len(AREAS) * len(CONFIGS)
    current = 0
    for area in AREAS:
        for cfg in CONFIGS:
            current += 1
            print(f"Global Progress: {current}/{total_experiments}")
            try:
                run_experiment(area, cfg)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
    print("Done.")