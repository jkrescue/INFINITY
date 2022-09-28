import open3d as o3d
import inflation as inf
import numpy as np

mesh = o3d.io.read_triangle_mesh("./example/wing.stl")


pcs = inf.inflation_sample(
    mesh,
    dis=np.ones(4) * 0.1,
    density=np.linspace(50, 10, 4),
    mode="uniform",
    seed=0,
)

o3d.visualization.draw_geometries(pcs)
