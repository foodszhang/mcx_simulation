import numpy as np
import jdata as jd
import os

size = 50
save_dir = f"./green_{size}/"
green_x_0 = np.zeros((size, size, size, size, size), np.float16)
green_y_0 = np.zeros((size, size, size, size, size), np.float16)
green_z_0 = np.zeros((size, size, size, size, size), np.float16)
green_x_1 = np.zeros((size, size, size, size, size), np.float16)
green_y_1 = np.zeros((size, size, size, size, size), np.float16)
green_z_1 = np.zeros((size, size, size, size, size), np.float16)
green_file_name = f"./green_{size}.npz"


for x in range(size):
    for y in range(size):
        for z in range(size):
            filename = f"{x}-{y}-{z}.jnii"
            file_path = os.join(save_dir, filename)
            full_data = jd.loadjd(file_path)
            if len(full_data["NIFTIData"].shape) == 3:
                flux = full_data["NIFTIData"][:, :, :]
            else:
                flux = full_data["NIFTIData"][:, :, :, 0, 0]
            green_x_0[x, y, z, :, :] = flux[0, :, :]
            green_y_0[x, y, z, :, :] = flux[:, 0, :]
            green_z_0[x, y, z, :, :] = flux[:, :, 0]
            green_x_1[x, y, z, :, :] = flux[size - 1, :, :]
            green_y_1[x, y, z, :, :] = flux[:, size - 1, :]
            green_z_1[x, y, z, :, :] = flux[:, :, size - 1]
np.savez_compressed(
    green_file_name,
    green_x_0,
    green_y_0,
    green_z_0,
    green_x_1,
    green_y_1,
    green_z_1,
)
