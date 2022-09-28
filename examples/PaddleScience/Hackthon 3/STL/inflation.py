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


def offset(mesh, dis):
    """
    Offset the 2D mesh

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh to be offset

    dis : float
        The distance to offset

    Returns
    -------
    open3d.geometry.TriangleMesh
    """

    # check if the mesh is 2D
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    if not np.allclose(normals[:, :-1], 0):
        raise ValueError("The mesh is not 2D")

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles)

    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = set(map(tuple, edges))
    edges = np.array(list(edges))

    vertices = np.asarray(mesh.vertices)[:, :-1]
    edges_in_triangle = np.array(
        [
            np.intersect1d(
                np.argwhere(triangles == edge[0])[:, 0],
                np.argwhere(triangles == edge[1])[:, 0],
            )
            for edge in edges
        ],
        dtype=object,
    )
    surface_edges = edges[[len(i) == 1 for i in edges_in_triangle]]
    edges_in_triangle = [i for i in edges_in_triangle if len(i) == 1]

    edges_normals = []
    for edge, triangle in zip(surface_edges, edges_in_triangle):
        triangle = triangles[triangle].squeeze()
        other_point = vertices[np.setdiff1d(triangle, edge)].squeeze()
        edge = vertices[edge]
        u = (other_point[0] - edge[0][0]) * (edge[0][0] - edge[1][0]) + (
            other_point[1] - edge[0][1]
        ) * (edge[0][1] - edge[1][1])
        u = u / np.sum((edge[0] - edge[1]) ** 2)
        edge_normal = edge[0] + u * (edge[0] - edge[1])
        edge_normal = edge_normal - other_point
        edges_normals.append(edge_normal)

    edges_normals = np.array(edges_normals)
    edges_normals = edges_normals / np.linalg.norm(edges_normals, axis=1)[:, None]

    new_mesh = o3d.geometry.TriangleMesh()
    new_vertices = []
    for point in set(surface_edges.reshape(-1)):
        index = np.argwhere(surface_edges == point)[:, 0]
        normal = edges_normals[index]
        d = np.ones(len(index)) * dis
        new_point = np.linalg.lstsq(normal, d, rcond=None)[0]
        new_point = vertices[point] + new_point
        new_vertices.append(new_point)

    new_vertices = np.hstack((np.array(new_vertices), np.zeros((len(new_vertices), 1))))
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    new_mesh.compute_triangle_normals()
    new_mesh.compute_vertex_normals()
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


def edge_sample(mesh, density=1, mode="uniform", seed=-1):
    """
    Sample points from the 2D mesh edges

    Parameters
    ----------
    mehs : open3d.geometry.TriangleMesh
        The mesh to be sampled

    density : int
        The density of points to be sampled

    mode : str
        The mode of sampling, can be "uniform" or "grid"

    seed : int
        The seed of random number generator
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles)

    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = set(map(tuple, edges))
    edges = np.array(list(edges))

    vertices = np.asarray(mesh.vertices)
    edges_in_triangle = np.array(
        [
            np.intersect1d(
                np.argwhere(triangles == edge[0])[:, 0],
                np.argwhere(triangles == edge[1])[:, 0],
            )
            for edge in edges
        ],
        dtype=object,
    )
    surface_edges = edges[[len(i) == 1 for i in edges_in_triangle]]
    edges_points = vertices[surface_edges]
    vlist = []
    for edge in edges_points:
        dx = edge[1][0] - edge[0][0]
        dy = edge[1][1] - edge[0][1]
        lenght = np.sqrt(dx**2 + dy**2)
        if np.isclose(dx, 0):
            y = sample(1, int(lenght * density), edge[0][1], edge[1][1], type=mode)
            x = np.full_like(y, edge[0][0])
        elif np.isclose(dy, 0):
            x = sample(1, int(lenght * density), edge[0][0], edge[1][0], type=mode)
            y = np.full_like(x, edge[0][1])
        else:
            k = dy / dx
            b = edge[0][1] - k * edge[0][0]
            x = sample(1, int(lenght * density), edge[0][0], edge[1][0], type=mode)
            y = k * x + b
        vlist.append(np.hstack([x, y, np.zeros_like(x)]))

    vlist = np.vstack(vlist)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(vlist)
    return pc


def sample(ndim, n, l_bound, u_bound, type="uniform"):
    if l_bound > u_bound:
        l_bound, u_bound = u_bound, l_bound
    if type == "uniform":
        return np.random.uniform(l_bound, u_bound, (n, ndim))
    elif type == "grid":
        return np.linspace(l_bound, u_bound, n).reshape(-1, 1)
    else:
        raise ValueError("type must be uniform or grid")


def inflation_sample(
    mesh,
    dis=(float, list, np.ndarray),
    density=(float, list, np.ndarray),
    dim=3,
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

    dim : int
        The dimension of the mesh, can be 2 or 3

    mode : str
        The mode of sampling, can be "uniform" or "poisson_disk"

    seed : int
        The seed of random number generator

    Returns
    -------
    open3d.geometry.PointCloud
    """
    assert dim in [2, 3], "The dimension of the mesh can only be 2 or 3"
    if isinstance(dis, (float, int)):
        dis = [dis]
    if isinstance(density, (float, int)):
        density = [density]
    if len(dis) != len(density):
        raise ValueError("The length of dis and density must be equal")
    new_mesh = mesh
    pcs = []
    for d, n in zip(dis, density):
        if dim == 2:
            new_mesh = offset(new_mesh, d)
            pc = edge_sample(new_mesh, n, mode, seed)
        else:
            new_mesh = inflation(new_mesh, d)
            pc = sample_points(new_mesh, n, mode=mode, seed=seed)
        pc.paint_uniform_color(np.random.random(3))
        pcs.append(pc)
    return pcs
