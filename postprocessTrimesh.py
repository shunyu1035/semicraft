import os
import tempfile
import numpy as np

try:
    from skimage import measure
except Exception as e:  # pragma: no cover - graceful fallback
    measure = None

try:
    import trimesh
except Exception as e:  # pragma: no cover
    trimesh = None


def voxels_to_bool(voxels, threshold=0.5):
    """Convert numeric voxel arrays to boolean occupancy.

    Args:
        voxels: ndarray, can be int/float; shape (X,Y,Z) or (X,Y,Z,C) — if C present sum over last axis.
        threshold: float threshold applied to values normalized to [0,1] or absolute for integers.

    Returns:
        boolean ndarray of shape (X,Y,Z)
    """
    vox = np.asarray(voxels)
    if vox.ndim == 4:
        vox = np.sum(vox, axis=-1)
    if vox.dtype == np.bool_:
        return vox
    # for integer arrays use direct threshold, for float try normalizing if in [0,1]
    if np.issubdtype(vox.dtype, np.floating):
        return vox > threshold
    else:
        return vox > threshold


def voxel_to_mesh(voxels, spacing=(1.0, 1.0, 1.0), level=0.5, downsample=1):
    """Convert voxel occupancy to a triangular mesh using marching cubes.

    Returns a `trimesh.Trimesh` instance (if trimesh available) or (verts, faces, normals).
    """
    if measure is None:
        raise ImportError('scikit-image is required: pip install scikit-image')
    if trimesh is None:
        raise ImportError('trimesh is required: pip install trimesh')

    vox_bool = voxels_to_bool(voxels)

    if downsample > 1:
        # simple block downsample by slicing — fast and memory efficient
        vox_bool = vox_bool[::downsample, ::downsample, ::downsample]
        spacing = (spacing[0] * downsample, spacing[1] * downsample, spacing[2] * downsample)

    # marching_cubes requires float input in recent scikit-image
    verts, faces, normals, values = measure.marching_cubes(vox_bool.astype(np.float32), level=level, spacing=spacing)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    return mesh


def save_mesh(mesh, path):
    """Save a `trimesh.Trimesh` to a file, format inferred from extension.

    Supported formats: ply, stl, glb, obj (delegated to trimesh.export)
    """
    if trimesh is None:
        raise ImportError('trimesh is required to save meshes')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mesh.export(path)


def show_mesh(mesh):
    """Try to display mesh using trimesh's built-in viewer.

    Falls back to exporting a temporary glb and printing path if viewer unavailable.
    """
    if trimesh is None:
        raise ImportError('trimesh is required to view meshes')
    try:
        mesh.show()
    except Exception:
        # fallback: export to temp glb and tell user where it is
        fd, tmp = tempfile.mkstemp(suffix='.glb')
        os.close(fd)
        mesh.export(tmp)
        print('Viewer failed; exported temporary glTF to:', tmp)


def film_to_voxels(film, material_index=0):
    """Convert a `film` array (X,Y,Z,C) to boolean voxel occupancy for `material_index`.

    `film` is expected like the project's convention where nonzero indicates presence.
    """
    film = np.asarray(film)
    if film.ndim != 4:
        raise ValueError('film must be 4D array (X,Y,Z,C)')
    # occupancy where the selected channel > 0
    return film[:, :, :, material_index] != 0


if __name__ == '__main__':
    # quick smoke test / example
    try:
        import trimesh as _t
        from skimage import measure as _m
    except Exception:
        print('Example requires scikit-image and trimesh. Skipping demo.')
    else:
        # small synthetic sphere example
        x, y, z = np.indices((80, 80, 80))
        c = np.array([40, 40, 40])
        r = 20
        vox = ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) < r * r
        print('Creating mesh from synthetic voxels...')
        m = voxel_to_mesh(vox, spacing=(1, 1, 1), level=0.5)
        print('Vertices:', len(m.vertices), 'Faces:', len(m.faces))
        try:
            m.show()
        except Exception:
            print('Show failed; exportable via save_mesh()')
