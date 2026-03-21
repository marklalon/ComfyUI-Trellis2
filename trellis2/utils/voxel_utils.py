"""
Voxelization utilities with GPU acceleration support.

This module provides a wrapper around o_voxel.convert.mesh_to_flexible_dual_grid
that supports GPU acceleration via a pure PyTorch implementation.
"""

import torch
from typing import Union, Tuple
import numpy as np
import time

__all__ = [
    "mesh_to_flexible_dual_grid",
]


def mesh_to_flexible_dual_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_size = None,
    grid_size = None,
    aabb = None,
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
    regularization_weight: float = 0.1,
    timing: bool = False,
    use_gpu: bool = False,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Voxelize a mesh into a sparse voxel grid with optional GPU acceleration.
    
    Args:
        vertices (torch.Tensor): The vertices of the mesh.
        faces (torch.Tensor): The faces of the mesh.
        voxel_size: The size of each voxel.
        grid_size: The size of the grid.
        aabb: The axis-aligned bounding box of the mesh.
        face_weight (float): Weight for face constraints in QEF.
        boundary_weight (float): Weight for boundary constraints in QEF.
        regularization_weight (float): Weight for regularization in QEF.
        timing (bool): Whether to print timing information.
        use_gpu (bool): Whether to use GPU implementation (pure PyTorch).
        chunk_size (int): Number of triangles to process per chunk (GPU only).
        
    Returns:
        Tuple of (voxel_indices, dual_vertices, intersected_flag)
    """
    if use_gpu:
        # Import GPU implementation (local copy to avoid o_voxel update issues)
        try:
            from .flexible_dual_grid_gpu import mesh_to_flexible_dual_grid_gpu
        except ImportError:
            print("[Warning] GPU voxelization requested but flexible_dual_grid_gpu not found. Falling back to CPU.")
            use_gpu = False
    
    if use_gpu:
        # Prepare parameters for GPU implementation
        # Convert vertices and faces to proper format
        vertices = vertices.float()
        faces = faces.int()
        
        # Process voxel_size
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        
        # Process grid_size
        if grid_size is not None:
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32)
        
        # Process aabb
        if aabb is not None:
            if isinstance(aabb, (list, tuple)):
                aabb = np.array(aabb)
            if isinstance(aabb, np.ndarray):
                aabb = torch.tensor(aabb, dtype=torch.float32)
        
        # Auto adjust aabb if not provided
        if aabb is None:
            min_xyz = vertices.min(dim=0).values
            max_xyz = vertices.max(dim=0).values
            
            if voxel_size is not None:
                padding = torch.ceil((max_xyz - min_xyz) / voxel_size) * voxel_size - (max_xyz - min_xyz)
                min_xyz -= padding * 0.5
                max_xyz += padding * 0.5
            if grid_size is not None:
                padding = (max_xyz - min_xyz) / (grid_size - 1)
                min_xyz -= padding * 0.5
                max_xyz += padding * 0.5
            
            aabb = torch.stack([min_xyz, max_xyz], dim=0).float()
        
        # Fill voxel_size or grid_size
        if voxel_size is None:
            voxel_size = (aabb[1] - aabb[0]) / grid_size
        if grid_size is None:
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        
        # Prepare for GPU
        vertices = vertices - aabb[0].reshape(1, 3)
        grid_range = torch.stack([torch.zeros_like(grid_size), grid_size], dim=0).int()
        
        # Ensure tensors are on CUDA
        vertices_g = vertices.cuda() if not vertices.is_cuda else vertices
        faces_g = faces.cuda() if not faces.is_cuda else faces
        voxel_size_g = voxel_size.cuda() if not voxel_size.is_cuda else voxel_size
        grid_range_g = grid_range.cuda() if not grid_range.is_cuda else grid_range
        
        # Call GPU implementation
        voxel_indices, dual_vertices, intersected_flag = mesh_to_flexible_dual_grid_gpu(
            vertices_g,
            faces_g,
            voxel_size_g,
            grid_range_g,
            face_weight,
            boundary_weight,
            regularization_weight,
            timing,
            chunk_size,
        )
        
        return voxel_indices, dual_vertices, intersected_flag
    else:
        # Use original CPU implementation
        import o_voxel.convert
        if timing:
            print("[CPU] Starting mesh_to_flexible_dual_grid_cpu")
            cpu_start = time.perf_counter()

        result = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=vertices,
            faces=faces,
            voxel_size=voxel_size,
            grid_size=grid_size,
            aabb=aabb,
            face_weight=face_weight,
            boundary_weight=boundary_weight,
            regularization_weight=regularization_weight,
            timing=timing,
        )

        if timing:
            voxel_indices = result[0]
            cpu_total = time.perf_counter() - cpu_start
            print(f"[CPU] Total time: {cpu_total:.3f}s, {voxel_indices.shape[0]} voxels")

        return result
