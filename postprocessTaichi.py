import numpy as np
import os

try:
    import taichi as ti
except Exception:
    ti = None


def voxels_to_bool(voxels, material_index=None, threshold=0.5):
    v = np.asarray(voxels)
    if v.ndim == 4:
        if material_index is None:
            v = np.sum(v, axis=-1)
        else:
            v = v[:, :, :, material_index]
    if v.dtype == np.bool_:
        return v
    if np.issubdtype(v.dtype, np.floating):
        return v > threshold
    return v != 0


def _init_taichi(arch_preference='gpu'):
    if ti is None:
        raise ImportError('Taichi is required: pip install taichi')
    # Some Taichi versions do not expose `is_initialized`. Attempt to init and
    # ignore errors if it's already initialized.
    try:
        if arch_preference == 'gpu':
            ti.init(arch=ti.gpu)
        else:
            ti.init(arch=ti.cpu)
    except Exception:
        try:
            ti.init(arch=ti.cpu)
        except Exception:
            # already initialized or no suitable backend; continue
            pass


def render_axis_projection_taichi(voxels, axis='z', window_size=(800, 800), cmap=None):
    """Render a simple axis-aligned front-projection of boolean voxels with Taichi GUI.

    This is a lightweight, fast renderer that finds the first occupied voxel along
    the chosen axis and paints the pixel with a color. It is not a full raymarcher
    but is GPU-accelerated and interactive for large volumes.

    Args:
        voxels: 3D boolean ndarray (X,Y,Z) or 4D (X,Y,Z,C) — will be converted.
        axis: 'x'|'y'|'z' view direction (camera looks along negative axis).
        window_size: (w,h) in pixels.
        cmap: dict mapping material index -> (r,g,b) floats 0..1; unused for boolean.
    """
    _init_taichi()

    vox_bool = voxels_to_bool(voxels)
    nx, ny, nz = vox_bool.shape

    # choose projection dims
    if axis == 'z':
        sx, sy, sz = nx, ny, nz
        proj_shape = (sx, sy)
    elif axis == 'y':
        sx, sy, sz = nx, nz, ny
        proj_shape = (sx, sy)
    elif axis == 'x':
        sx, sy, sz = ny, nz, nx
        proj_shape = (sx, sy)
    else:
        raise ValueError('axis must be x/y/z')

    pixels_w, pixels_h = window_size

    # Taichi image field
    img = ti.Vector.field(3, dtype=ti.f32, shape=(pixels_w, pixels_h))

    # create a Taichi ndarray from the numpy occupancy to use inside kernels
    occ_np = np.ascontiguousarray(vox_bool.astype(np.uint8))
    occ = ti.field(dtype=ti.u8, shape=occ_np.shape)
    # populate field from numpy
    try:
        occ.from_numpy(occ_np)
    except Exception:
        # fallback: manual copy (slower) if from_numpy not available
        @ti.kernel
        def copy_from_np():
            for i, j, k in ti.ndrange(occ_np.shape[0], occ_np.shape[1], occ_np.shape[2]):
                occ[i, j, k] = int(occ_np[i, j, k])
        copy_from_np()

    @ti.kernel
    def render():
        for i, j in img:
            # map pixel to volume coords
            vx = (i * proj_shape[0]) // pixels_w
            vy = (j * proj_shape[1]) // pixels_h
            color = ti.Vector([0.0, 0.0, 0.0])
            if axis == 'z':
                for kk in range(sz):
                    k = sz - 1 - kk
                    if occ[vx, vy, k] != 0:
                        color = ti.Vector([1.0, 0.6, 0.2])
                        break
            elif axis == 'y':
                for kk in range(sz):
                    k = sz - 1 - kk
                    if occ[vx, k, vy] != 0:
                        color = ti.Vector([1.0, 0.6, 0.2])
                        break
            else:  # axis == 'x'
                for kk in range(sz):
                    k = sz - 1 - kk
                    if occ[k, vx, vy] != 0:
                        color = ti.Vector([1.0, 0.6, 0.2])
                        break
            img[i, j] = color

    gui = ti.GUI('Taichi Voxel Viewer', res=window_size)
    while gui.running:
        render()
        gui.set_image(img.to_numpy())
        if gui.get_event(gui.ESCAPE):
            break
        if gui.get_event(gui.SPACE):
            # toggle: flip axis order as simple interaction
            axis = 'y' if axis == 'z' else 'z'
        gui.show()


def save_projection_image(voxels, out_path, axis='z', window_size=(800, 800)):
    _init_taichi()
    vox_bool = voxels_to_bool(voxels)
    nx, ny, nz = vox_bool.shape
    sx, sy = (nx, ny) if axis == 'z' else ((nx, nz) if axis == 'y' else (ny, nz))
    img = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
    for ix in range(window_size[0]):
        for iy in range(window_size[1]):
            vx = (ix * sx) // window_size[0]
            vy = (iy * sy) // window_size[1]
            found = False
            if axis == 'z':
                for k in range(nz - 1, -1, -1):
                    if vox_bool[vx, vy, k]:
                        img[iy, ix] = np.array([255, 153, 51], dtype=np.uint8)
                        found = True
                        break
            elif axis == 'y':
                for k in range(ny - 1, -1, -1):
                    if vox_bool[vx, k, vy]:
                        img[iy, ix] = np.array([255, 153, 51], dtype=np.uint8)
                        found = True
                        break
            else:
                for k in range(nx - 1, -1, -1):
                    if vox_bool[k, vx, vy]:
                        img[iy, ix] = np.array([255, 153, 51], dtype=np.uint8)
                        found = True
                        break
    from imageio import imwrite
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    imwrite(out_path, img)


if __name__ == '__main__':
    print('Taichi voxel renderer module. Use render_axis_projection_taichi(voxels) or save_projection_image(...)')


def render_voxel_raymarch(voxels, window_size=(800, 600), fov=45.0, rotate=True):
    """Simple perspective raymarch renderer for boolean voxels using Taichi.

    - voxels: 3D boolean ndarray (nx,ny,nz)
    - window_size: (w,h)
    - fov: vertical field-of-view in degrees
    - rotate: if True, slowly rotate camera around scene
    """
    _init_taichi()

    vox_bool = voxels_to_bool(voxels)
    nx, ny, nz = vox_bool.shape

    # convert to taichi field
    occ_np = np.ascontiguousarray(vox_bool.astype(np.uint8))
    occ = ti.field(dtype=ti.u8, shape=occ_np.shape)
    try:
        occ.from_numpy(occ_np)
    except Exception:
        @ti.kernel
        def copy_from_np():
            for i, j, k in ti.ndrange(occ_np.shape[0], occ_np.shape[1], occ_np.shape[2]):
                occ[i, j, k] = int(occ_np[i, j, k])
        copy_from_np()

    W, H = window_size
    img = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

    # camera params
    cam_dist = max(nx, ny, nz) * 1.8
    theta = 0.0
    center = ti.Vector([nx / 2.0, ny / 2.0, nz / 2.0])

    @ti.func
    def sample_occ(ix, iy, iz):
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            return occ[ix, iy, iz] != 0
        return False

    @ti.kernel
    def render_kernel(cam_pos: ti.types.vector(3, ti.f32), cam_dir: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32)):
        aspect = W / H
        tan_fov = ti.tan(fov * 0.5 * 3.1415926 / 180.0)
        right = cam_dir.cross(up).normalized()
        for i, j in img:
            u = (2 * ((i + 0.5) / W) - 1) * aspect * tan_fov
            v = (1 - 2 * ((j + 0.5) / H)) * tan_fov
            rd = (cam_dir + u * right + v * up).normalized()

            color = ti.Vector([0.0, 0.0, 0.0])
            t = 0.0
            t_max = cam_dist * 3.0
            step = 0.7
            hit = False
            for _ in range(512):
                if t > t_max:
                    break
                p = cam_pos + rd * t
                ix = int(p.x + 0.5)
                iy = int(p.y + 0.5)
                iz = int(p.z + 0.5)
                if sample_occ(ix, iy, iz):
                    # compute normal by central differences
                    nx_s = 0.0
                    ny_s = 0.0
                    nz_s = 0.0
                    if sample_occ(ix + 1, iy, iz):
                        nx_s += 1.0
                    if sample_occ(ix - 1, iy, iz):
                        nx_s -= 1.0
                    if sample_occ(ix, iy + 1, iz):
                        ny_s += 1.0
                    if sample_occ(ix, iy - 1, iz):
                        ny_s -= 1.0
                    if sample_occ(ix, iy, iz + 1):
                        nz_s += 1.0
                    if sample_occ(ix, iy, iz - 1):
                        nz_s -= 1.0
                    nrm = ti.Vector([nx_s, ny_s, nz_s])
                    if nrm.norm() == 0:
                        nrm = ti.Vector([0.0, 0.0, 1.0])
                    else:
                        nrm = nrm.normalized()
                    # simple lambertian shading
                    light_dir = ti.Vector([0.5, 0.7, 0.2]).normalized()
                    diff = nrm.dot(light_dir)
                    diff = (diff + 1.0) * 0.5
                    base = ti.Vector([0.9, 0.4, 0.2])
                    color = base * diff
                    # add depth fog
                    fog = ti.exp(-t * 0.02)
                    color = color * fog
                    hit = True
                    break
                t += step
            if not hit:
                color = ti.Vector([0.03, 0.05, 0.08])  # background
            img[i, j] = color

    gui = ti.GUI('Taichi Voxel Raymarch', res=window_size)
    frame = 0
    # spherical camera params
    theta = 0.0
    phi = 0.2
    depth_mode = False
    base_color = ti.Vector([0.9, 0.4, 0.2])
    bg_color = ti.Vector([0.03, 0.05, 0.08])

    while gui.running:
        # keyboard interaction: rotate/pan/zoom
        if gui.is_pressed(ti.GUI.LEFT) or gui.is_pressed('a'):
            theta -= 0.04
        if gui.is_pressed(ti.GUI.RIGHT) or gui.is_pressed('d'):
            theta += 0.04
        if gui.is_pressed(ti.GUI.UP) or gui.is_pressed('w'):
            phi = min(phi + 0.04, 1.4)
        if gui.is_pressed(ti.GUI.DOWN) or gui.is_pressed('s'):
            phi = max(phi - 0.04, -1.4)
        if gui.is_pressed('+') or gui.is_pressed('='):
            cam_dist *= 0.95
        if gui.is_pressed('-') or gui.is_pressed('_'):
            cam_dist *= 1.05
        if gui.is_pressed('r'):
            theta = 0.0; phi = 0.2; cam_dist = max(nx, ny, nz) * 1.8
        if gui.is_pressed(' '):
            rotate = not rotate
        if gui.is_pressed('t'):
            depth_mode = not depth_mode

        if rotate:
            theta += 0.01

        # spherical -> cartesian camera position
        cam_x = center[0] + cam_dist * ti.cos(theta) * ti.cos(phi)
        cam_y = center[1] + cam_dist * ti.sin(phi)
        cam_z = center[2] + cam_dist * ti.sin(theta) * ti.cos(phi)
        cam_pos = ti.Vector([cam_x, cam_y, cam_z])
        cam_dir = (center - cam_pos).normalized()
        up = ti.Vector([0.0, 1.0, 0.0])

        render_kernel(cam_pos, cam_dir, up)

        # if depth_mode, tint by depth: we can approximate by blending with bg
        out_img = img.to_numpy()
        if depth_mode:
            # apply gamma and slight boost for visualization
            out_img = np.clip(out_img ** 0.8 * 1.1, 0.0, 1.0)

        gui.set_image(out_img)
        if gui.get_event(ti.GUI.ESCAPE):
            break
        gui.show()
        frame += 1
