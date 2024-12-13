from pathlib import Path

import numpy as np
import torch


def prepare_data2d(path: Path) -> torch.Tensor:
    res = []
    for file in path.glob('**/*'):
        data = np.genfromtxt(file, delimiter=',', skip_header=True, dtype='float32')
        r = torch.from_numpy(data)
        r = r[None, :, :]
        res.append(r)
    return torch.stack(res)
    #return res


def prepare_data3d(path: Path):
    res = []
    for file in path.glob('**/*'):
        data = np.genfromtxt(file, delimiter=',', skip_header=True, dtype='float32')
        data = data.reshape(12, 150)

        c2 = np.array([data[0], data[1], data[2]])
        l2 = np.array([data[3], data[4], data[5]])
        s2 = np.array([data[6], data[7], data[8] ])
        surf2 = np.array([data[9], data[10], data[11]])

        r = np.array([c2, l2, s2, surf2])
        r = torch.from_numpy(r)
        res.append(r)
    return torch.stack(res)


def prepare_labels(path: Path) -> torch.Tensor:
    res = []
    for file in path.glob('**/*'):
        label = np.genfromtxt(file, delimiter=',', dtype='float32')
        
        r = torch.from_numpy(label)
        r = r[None, :, :]
        res.append(r)

    return torch.stack(res)

# a = prepare_data(Path(r'data\united\cfs'))