from typing import Tuple
import random

import numpy as np
import torch
import torchvision
import porespy as ps
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


def get_central_crop(img: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    h, w, d = img.size()
    h_, w_, d_ = size
    lh = (h - h_) // 2 + (h - h_) % 2
    lw = (w - w_) // 2 + (w - w_) % 2
    ld = (d - d_) // 2 + (d - d_) % 2
    rh = (h - h_) // 2
    rw = (w - w_) // 2
    rd = (d - d_) // 2
    return img[lh:-rh, lw:-rw, ld:-rd]


def generate_porespy_data_with_rotation(
        image_size_: int,
        _porosity_range: Tuple[int, int] = (0.10, 0.25),
        _blobness_range: Tuple[int, int] = (1, 3),
) -> np.ndarray:
    porosity = random.uniform(_porosity_range[0], _porosity_range[1])
    blobness_x = random.uniform(_blobness_range[0], _blobness_range[1])
    blobness_y = random.uniform(_blobness_range[0], _blobness_range[1])
    blobness_z = random.uniform(_blobness_range[0], _blobness_range[1])

    print(f'Porosity:\n\t{porosity = :.4f}')
    print(f'Blobness:\n\t{blobness_x = :.4f}, {blobness_y = :.4f}, {blobness_z = :.4f}')
    data_ = ps.generators.blobs(
        shape=[image_size_, image_size_, image_size_],
        porosity=1 - porosity,
        blobiness=[blobness_x, blobness_y, blobness_z],
    )

    data_2 = np.concatenate([data_, data_[::-1, :, :]], axis=0)
    data_4 = np.concatenate([data_2, data_2[:, ::-1, :]], axis=1)
    data_8 = np.concatenate([data_4, data_4[:, :, ::-1]], axis=2)

    t = torch.from_numpy(data_8)

    angle_x = random.uniform(0, 45)
    angle_y = random.uniform(0, 45)
    angle_z = random.uniform(0, 45)
    print(f'Rotation angles:\n\t{angle_x = :.4f}, {angle_y = :.4f}, {blobness_y = :.4f}, {angle_z = :.4f}')

    rotated_x = torchvision.transforms.functional.rotate(img=t, angle=angle_x)
    rotated_y = torchvision.transforms.functional.rotate(img=rotated_x.permute(1, 0, 2), angle=angle_y).permute(1, 0, 2)
    rotated_z = torchvision.transforms.functional.rotate(img=rotated_y.permute(2, 1, 0), angle=angle_z).permute(2, 1, 0)

    img = get_central_crop(rotated_z, size=(image_size_, image_size_, image_size_))
    return img.cpu().data.numpy()


def main():
    set_seed(0)
    data_test = generate_porespy_data_with_rotation(image_size_=300)
    # i = 0
    # while i < 10:
    #     data_test = generate_porespy_data_with_rotation(image_size_=300)
    #     data_test.tofile(f'/mnt/moredata/tolstygin/generated_data/data/generated_{i}')
    #     i += 1
    # plt.imshow(data_test[0])
    # plt.imshow(data_test[:, 0])
    # plt.imshow(data_test[:, :, 0])


if __name__ == '__main__':
    main()
