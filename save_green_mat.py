import jdata as jd
import os
import numpy as np

size = 50
jnii_dir = "./green_50/"
total_green_mat = np.zeros((6, size, size, size, size, size))
# total_green_mat = np.load("green_mat.npz")["arr_0"]
for x in range(0, size):
    for y in range(size):
        for z in range(size):
            name = f"{x}_{y}_{z}.jnii"
            filename = os.path.join(jnii_dir, name)
            if not os.path.exists(filename):
                raise Exception(f"!!!!!!{filename} not  exits")
            try:
                full_data = jd.loadjd(filename)
            except Exception as e:
                print("3453453454error", filename)
                np.savez_compressed("green_mat.npz", total_green_mat)
                raise e
            if len(full_data["NIFTIData"].shape) == 3:
                flux = full_data["NIFTIData"][:, :, :]
            else:
                flux = full_data["NIFTIData"][:, :, :, 0, 0]
            total_green_mat[0, x, y, z, :, :] = flux[0, :, :]
            total_green_mat[1, x, y, z, :, :] = flux[:, 0, :]
            total_green_mat[2, x, y, z, :, :] = flux[:, :, 0]
            total_green_mat[3, x, y, z, :, :] = flux[size - 1, :, :]
            total_green_mat[4, x, y, z, :, :] = flux[:, size - 1, :]
            total_green_mat[5, x, y, z, :, :] = flux[size - 1, :, :]

np.savez_compressed("green_mat.npz", total_green_mat)
