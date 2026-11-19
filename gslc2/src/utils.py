import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def sync_gpu(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def environment(device):
    import torchvision
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda': torch.version.cuda, 'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'deterministic_algorithms': True, 'cudnn_benchmark': False,
            'training_precision': 'float32; DDPM autocast bfloat16 on supported CUDA',
            'evaluation_precision': 'float32 (FID accumulators float64)'}
