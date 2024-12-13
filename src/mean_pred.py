from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def mean_labels(path: Path) -> torch.Tensor:
    comp_0_0 = []
    comp_0_1 = []
    comp_0_2 = []
    comp_1_0 = []
    comp_1_1 = []
    comp_1_2 = []
    comp_2_0 = []
    comp_2_1 = []
    comp_2_2 = []
    for file in path.glob('**/*'):
        if int(file.name.split('.')[0]) > 30:
            break
        label = np.genfromtxt(file, delimiter=',', dtype='float32')
        comp_0_0.append(label[0][0])
        comp_0_1.append(label[0][1])
        comp_0_2.append(label[0][2])
        comp_1_0.append(label[1][0])
        comp_1_1.append(label[1][1])
        comp_1_2.append(label[1][2])
        comp_2_0.append(label[2][0])
        comp_2_1.append(label[2][1])
        comp_2_2.append(label[2][2])


    mean_0_0 = np.mean(comp_0_0)
    mean_0_1 = np.mean(comp_0_1)
    mean_0_2 = np.mean(comp_0_2)
    mean_1_0 = np.mean(comp_1_0)
    mean_1_1 = np.mean(comp_1_1)
    mean_1_2 = np.mean(comp_1_2)
    mean_2_0 = np.mean(comp_2_0)
    mean_2_1 = np.mean(comp_2_1)
    mean_2_2 = np.mean(comp_2_2)
    
    mae_0_0 = np.sum(np.abs(mean_0_0 - comp_0_0)) / 30
    mae_0_1 = np.sum(np.abs(mean_0_1 - comp_0_1)) / 30
    mae_0_2 = np.sum(np.abs(mean_0_2 - comp_0_2)) / 30
    mae_1_0 = np.sum(np.abs(mean_1_0 - comp_1_0)) / 30
    mae_1_1 = np.sum(np.abs(mean_1_1 - comp_1_1)) / 30
    mae_1_2 = np.sum(np.abs(mean_1_2 - comp_1_2)) / 30
    mae_2_0 = np.sum(np.abs(mean_2_0 - comp_2_0)) / 30
    mae_2_1 = np.sum(np.abs(mean_2_1 - comp_2_1)) / 30
    mae_2_2 = np.sum(np.abs(mean_2_2 - comp_2_2)) / 30

    print('MAE 0_0: ', mae_0_0)
    print('MAE 0_1: ', mae_0_1)
    print('MAE 0_2: ', mae_0_2)
    print('MAE 1_0: ', mae_1_0)
    print('MAE 1_1: ', mae_1_1)
    print('MAE 1_2: ', mae_1_2)
    print('MAE 2_0: ', mae_2_0)
    print('MAE 2_1: ', mae_2_1)
    print('MAE 2_2: ', mae_2_2)


print(mean_labels(Path(r'data\united\perm')))
