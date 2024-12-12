from pathlib import Path

import numpy as np
import torch


def prepare_data(path: Path) -> torch.Tensor:
    res = []
    for file in path.glob('**/*'):
        data = np.genfromtxt(file, delimiter=',', skip_header=True, dtype='float32')
        r = torch.from_numpy(data)
        r = r[None, :, :]
        res.append(r)
    return torch.stack(res)
    #return res



def prepare_labels(path: Path) -> torch.Tensor:
    res = []
    for file in path.glob('**/*'):
        label = np.genfromtxt(file, delimiter=',', dtype='float32')
        r = torch.from_numpy(label)
        r = r[None, :, :]
        res.append(r)

    return torch.stack(res)

# a = prepare_data(Path(r'data\united\cfs'))