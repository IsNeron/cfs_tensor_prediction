import pandas as pd
import numpy as np
from pathlib import Path

CFS_PATH = Path(r'data\raw\cfs')
CFS_OUT_PATH = Path(r'data\united\cfs')
STOKES_PATH = Path(r'data\raw\permeability')
STOKES_OUT_PATH = Path(r'data\united\perm')

def prepare_data(path: Path) -> np.array:
    res = []
    for file in path.glob('**/*'):
        a = np.genfromtxt(file, delimiter=',', skip_header=True)
        res.append(a)

    return np.array(res)



def prepare_labels(path: Path) -> np.array:
    res = []
    for file in path.glob('**/*'):
        a = np.genfromtxt(file, delimiter=',')
        res.append(a)

    return np.array(res)



def _prepare_cfs(path: Path, out: Path, count: int) -> None:
    i = 0
    while i <= count-1:
        res = {}
        data_path = path / str(i)
        for file in data_path.glob('**/*'):
            name = file.name.split('.')[0]
            _, _, cf, napr = name.split('-')

            data = pd.read_csv(file, header=None)[0].to_numpy()
            res[cf + '_' + napr] = data
        a = pd.DataFrame.from_dict(res)
        a.to_csv(rf'{out}\{i}.csv', index=False)
        i += 1



def _prepare_stokes(path: Path, out: Path, count: int) -> None:
    i = 0
    while i <= count-1:
        res = []
        data_path = path / str(i)
        for file in data_path.glob('**/*'):
            name = file.name.split('.')[0]
            _, _, napr = name.split('_')

            res.append(np.genfromtxt(file, delimiter=','))

        a  = np.array(res)
        np.savetxt(rf'{out}\{i}.csv', a, delimiter=",")
        i += 1   


_prepare_cfs(CFS_PATH, CFS_OUT_PATH, 50)
_prepare_stokes(STOKES_PATH, STOKES_OUT_PATH, 50)

a = prepare_data(CFS_OUT_PATH)
b = prepare_labels(STOKES_OUT_PATH)
