import numpy as np
import open3d as o3d


def inflation(mesh, dis):
    """
    Inflation the mesh

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh to be inflated
    dis : float
        The distance to inflate

    Returns
    -------
    open3d.geometry.TriangleMesh
        The inflated mesh
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    triangles = np.asarray(mesh.triangles)
    points = np.asarray(mesh.vertices)

    remove_ids = []
    for i, point in enumerate(points):
        boolean_index = np.argwhere(triangles == i)[:, 0]
        if len(boolean_index) < 3:
            remove_ids.append(i)
    mesh.remove_vertices_by_index(remove_ids)

    points = np.asarray(mesh.vertices)
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    mesh.orient_triangles()
    triangles = np.asarray(mesh.triangles)
    new_mesh = o3d.geometry.TriangleMesh()
    new_points = []
    for i, point in enumerate(points):
        boolean_index = np.argwhere(triangles == i)[:, 0]
        normal = normals[boolean_index]
        d = np.ones(len(normal)) * dis
        new_point = np.linalg.lstsq(normal, d, rcond=None)[0].squeeze()
        new_point = point + new_point
        if np.linalg.norm(new_point - point) > dis * 2:
            # TODO : Find a better way to solve the bad inflation
            new_point = point + dis * normal.mean(axis=0)

        new_points.append(new_point)

    new_points = np.array(new_points)
    new_mesh.vertices = o3d.utility.Vector3dVector(new_points)
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    new_mesh.remove_duplicated_vertices()
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_duplicated_triangles()
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_triangle_normals()
    return new_mesh


def sample_points(mesh, density=1, mode="uniform", seed=-1):
    """
    Sample points from the mesh

    Parameters
    ----------

    mesh : open3d.geometry.TriangleMesh
        The mesh to be sampled

    density : int
        The density of points to be sampled

    mode : str
        The mode of sampling, can be "uniform" or "poisson_disk"

    seed : int
        The seed of random number generator

    Returns
    -------
    open3d.geometry.PointCloud
    """
    total_points = int(mesh.get_surface_area() * density)
    if mode == "uniform":
        pc = mesh.sample_points_uniformly(total_points, seed=seed)
    elif mode == "poisson_disk":
        pc = mesh.sample_points_poisson_disk(total_points, seed=seed)
    else:
        pass
    return pc


def inflation_sample(
    mesh,
    dis=(float, list, np.ndarray),
    density=(float, list, np.ndarray),
    mode="uniform",
    seed=-1,
):
    """
    Inflation the mesh and sample points from the inflated mesh

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh

    dis : float or list or np.ndarray
        The distance to inflate

    density : float or list or np.ndarray
        The density of points to be sampled

    mode : str
        The mode of sampling, can be "uniform" or "poisson_disk"

    seed : int
        The seed of random number generator

    Returns
    -------
    open3d.geometry.PointCloud
    """
    if isinstance(dis, (float, int)):
        dis = [dis]
    if isinstance(density, (float, int)):
        density = [density]
    if len(dis) != len(density):
        raise ValueError("The length of dis and density must be equal")
    new_mesh = mesh
    pcs = []
    for d, n in zip(dis, density):
        new_mesh = inflation(new_mesh, d)
        pc = sample_points(new_mesh, n, mode=mode, seed=seed)
        pc.paint_uniform_color(np.random.random(3))
        pcs.append(pc)
    return pcs
