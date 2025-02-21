from typing import *
import numpy as np
import trimesh
from ..representations import MeshExtractResult

def to_trimesh(
    mesh: MeshExtractResult,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        mesh (MeshExtractResult): Extracted mesh.
    """
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    vertex_colors = mesh.vertex_attrs[:, : 3].cpu().numpy()

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    mesh = trimesh.Trimesh(vertices, faces, vertex_colors = vertex_colors)
    return mesh
