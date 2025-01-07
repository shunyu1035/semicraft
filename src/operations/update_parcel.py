from numba import jit, prange
import numpy as np

@jit(nopython=True)
def update_parcel(parcel, celllength, tStep):
    # 预计算 1/celllength，避免重复计算
    inv_celllength = 1.0 / celllength

    # 更新位置：parcel[:, :3] 为位置，parcel[:, 3:6] 为速度
    parcel[:, :3] += parcel[:, 3:6] * tStep

    # 计算新的 ijk 值并将其直接赋值到 parcel 的第 6、7、8 列
    # ijk = np.rint((parcel[:, :3] * inv_celllength) + 0.5).astype(np.int32)
    ijk = np.rint(parcel[:, :3] * inv_celllength).astype(np.int32)
    parcel[:, 6:9] = ijk

    return parcel

@jit(nopython=True)
def update_parcel_nolength(parcel):
    # 更新位置：parcel[:, :3] 为位置，parcel[:, 3:6] 为速度
    parcel[:, :3] += parcel[:, 3:6]

    # 计算新的 ijk 值并将其直接赋值到 parcel 的第 6、7、8 列
    ijk = np.rint(parcel[:, :3]).astype(np.int32)
    parcel[:, 6:9] = ijk

    return parcel