"""
Pure PyTorch GPU implementation of mesh_to_flexible_dual_grid.

This module provides a GPU-accelerated alternative to the CPU-bound
_C.mesh_to_flexible_dual_grid_cpu() function.
"""

from typing import Union, List, Tuple, Optional
import numpy as np
import torch
import time

__all__ = [
    "mesh_to_flexible_dual_grid_gpu",
]


def _triangle_aabb_intersect_batched(
    tri_verts: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> torch.Tensor:
    """
    Test triangle-AABB intersection using Separating Axis Theorem (SAT).

    Tests 13 axes:
    - 3 AABB face normals (x, y, z)
    - 1 triangle face normal
    - 9 cross products of triangle edges with AABB face normals

    Args:
        tri_verts: (M, 3, 3) triangle vertices
        aabb_min: (M, 3) AABB minimum corners
        aabb_max: (M, 3) AABB maximum corners

    Returns:
        (M,) boolean tensor indicating intersection
    """
    M = tri_verts.shape[0]
    device = tri_verts.device

    # AABB center and extents
    aabb_center = (aabb_min + aabb_max) / 2  # (M, 3)
    aabb_extents = (aabb_max - aabb_min) / 2  # (M, 3)

    # Center triangle vertices on AABB center (standard SAT formulation)
    tri_centered = tri_verts - aabb_center.unsqueeze(1)  # (M, 3, 3)

    # Triangle edges (translation-invariant)
    e0 = tri_centered[:, 1] - tri_centered[:, 0]  # (M, 3)
    e1 = tri_centered[:, 2] - tri_centered[:, 1]  # (M, 3)
    e2 = tri_centered[:, 0] - tri_centered[:, 2]  # (M, 3)

    # Triangle normal
    tri_normal = torch.cross(e0, e1, dim=-1)  # (M, 3)

    intersect = torch.ones(M, dtype=torch.bool, device=device)

    # Test 1: AABB face normals (x, y, z axes)
    # Project centered triangle vertices onto each axis
    for axis in range(3):
        tri_proj = tri_centered[:, :, axis]  # (M, 3)
        tri_min = tri_proj.min(dim=1).values
        tri_max = tri_proj.max(dim=1).values
        r = aabb_extents[:, axis]
        no_overlap = (tri_min > r) | (tri_max < -r)
        intersect = intersect & ~no_overlap

    # Test 2: Triangle face normal
    tri_normal_len = tri_normal.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    tri_normal_unit = tri_normal / tri_normal_len
    r = (aabb_extents * tri_normal_unit.abs()).sum(dim=-1)
    # All triangle vertices are coplanar, use vertex 0 projection
    d = (tri_centered[:, 0] * tri_normal_unit).sum(dim=-1)
    no_overlap = d.abs() > r + 1e-6
    intersect = intersect & ~no_overlap

    # Test 3: 9 cross product axes (edge x AABB normal)
    edges = [e0, e1, e2]
    for i in range(3):
        edge = edges[i]
        for j in range(3):
            # Cross product of edge with axis j unit vector
            # edge x e_j has a simple closed form:
            #   j=0: (0, -edge_z, edge_y)
            #   j=1: (edge_z, 0, -edge_x)
            #   j=2: (-edge_y, edge_x, 0)
            if j == 0:
                axis = torch.stack([torch.zeros(M, device=device), -edge[:, 2], edge[:, 1]], dim=-1)
            elif j == 1:
                axis = torch.stack([edge[:, 2], torch.zeros(M, device=device), -edge[:, 0]], dim=-1)
            else:
                axis = torch.stack([-edge[:, 1], edge[:, 0], torch.zeros(M, device=device)], dim=-1)

            axis_len = axis.norm(dim=-1, keepdim=True)
            # Skip degenerate axes (parallel edge and AABB normal)
            degenerate = (axis_len.squeeze(-1) < 1e-10)
            axis_unit = axis / axis_len.clamp_min(1e-10)

            # Project centered triangle vertices onto axis
            tri_proj = (tri_centered * axis_unit.unsqueeze(1)).sum(dim=-1)  # (M, 3)
            tri_min = tri_proj.min(dim=1).values
            tri_max = tri_proj.max(dim=1).values

            # AABB half-extent projection
            r = (aabb_extents * axis_unit.abs()).sum(dim=-1)

            # No overlap if entire triangle projection is outside [-r, r]
            no_overlap = (tri_min > r + 1e-6) | (tri_max < -r - 1e-6)
            # Degenerate axes can't separate, so they always "intersect"
            no_overlap = no_overlap & ~degenerate
            intersect = intersect & ~no_overlap

    return intersect


def _rasterize_triangles_to_voxels(
    tri_verts_voxel: torch.Tensor,
    grid_size: torch.Tensor,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize triangles to voxels using fully vectorized processing.

    For each triangle, compute its integer bounding box in voxel space,
    enumerate candidate voxels, and filter using triangle-AABB intersection test.

    Args:
        tri_verts_voxel: (F, 3, 3) triangle vertices in voxel coordinates
        grid_size: (3,) grid dimensions
        chunk_size: Number of triangles to process per chunk

    Returns:
        pair_indices: (M, 2) tensor of (triangle_idx, voxel_idx)
        voxel_coords_unique: (N, 3) unique voxel coordinates
    """
    tri_verts_voxel = tri_verts_voxel.to(torch.float64)
    device = tri_verts_voxel.device
    F = tri_verts_voxel.shape[0]

    # Compute bboxes for ALL triangles
    bbox_min_all = tri_verts_voxel.min(dim=1).values.floor().long()  # (F, 3)
    bbox_max_all = tri_verts_voxel.max(dim=1).values.ceil().long()   # (F, 3)

    # Clamp to grid bounds
    grid_max = grid_size.long() - 1
    grid_size_long = grid_size.long()
    is_pow2_grid = bool(((grid_size_long & (grid_size_long - 1)) == 0).all().item())
    if is_pow2_grid:
        bbox_min_all = bbox_min_all.clamp(min=0)
    else:
        bbox_min_all = torch.min(bbox_min_all.clamp(min=0), grid_max.unsqueeze(0).expand(F, -1))
    bbox_max_all = torch.min(bbox_max_all, grid_max.unsqueeze(0).expand(F, -1))
    bbox_max_all = bbox_max_all.clamp(min=0)

    # Compute bbox sizes and total candidates
    bbox_size_all = (bbox_max_all - bbox_min_all + 1).clamp(min=0)  # (F, 3)
    total_per_tri = bbox_size_all[:, 0] * bbox_size_all[:, 1] * bbox_size_all[:, 2]  # (F,)

    # Filter valid triangles (with > 0 candidates)
    valid_mask = total_per_tri > 0
    if not valid_mask.any():
        return (torch.empty((0, 2), dtype=torch.long, device=device),
                torch.empty((0, 3), dtype=torch.long, device=device))

    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    v_tri = tri_verts_voxel[valid_idx]        # (V, 3, 3)
    v_bbox_min = bbox_min_all[valid_idx]       # (V, 3)
    v_bbox_size = bbox_size_all[valid_idx]     # (V, 3)
    v_total = total_per_tri[valid_idx]         # (V,)
    V = valid_idx.shape[0]

    all_pairs = []

    # Process in candidate-count-bounded chunks to control memory
    # Target ~500K candidates per batch for SAT test
    max_candidates_per_batch = 500_000

    # Sort by total candidates to batch similarly-sized triangles
    sort_idx = v_total.argsort()
    v_tri = v_tri[sort_idx]
    v_bbox_min = v_bbox_min[sort_idx]
    v_bbox_size = v_bbox_size[sort_idx]
    v_total = v_total[sort_idx]
    valid_idx = valid_idx[sort_idx]

    # Process in batches
    batch_start = 0
    while batch_start < V:
        # Find batch end based on candidate count limit
        cumsum = v_total[batch_start:].cumsum(0)
        batch_count = (cumsum <= max_candidates_per_batch).sum().item()
        batch_count = max(batch_count, 1)  # At least 1 triangle per batch
        batch_end = min(batch_start + batch_count, V)

        b_tri = v_tri[batch_start:batch_end]            # (B, 3, 3)
        b_bbox_min = v_bbox_min[batch_start:batch_end]   # (B, 3)
        b_bbox_size = v_bbox_size[batch_start:batch_end]  # (B, 3)
        b_total = v_total[batch_start:batch_end]          # (B,)
        b_global_idx = valid_idx[batch_start:batch_end]   # (B,) global triangle indices
        B = b_tri.shape[0]

        total_candidates = b_total.sum().item()
        if total_candidates == 0:
            batch_start = batch_end
            continue

        # Vectorized candidate generation using repeat_interleave
        # tri_local_idx[k] = which triangle (local index 0..B-1) candidate k belongs to
        tri_local_idx = torch.repeat_interleave(
            torch.arange(B, device=device, dtype=torch.long), b_total
        )  # (total_candidates,)

        # Compute local linear offset within each triangle's bbox
        offsets_cumsum = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long), b_total.cumsum(0)
        ])
        local_offset = torch.arange(total_candidates, device=device, dtype=torch.long) - offsets_cumsum[tri_local_idx]

        # Convert linear offset to (dx, dy, dz) within each bbox
        exp_size = b_bbox_size[tri_local_idx]  # (total_candidates, 3)
        sz = exp_size[:, 2]
        sy = exp_size[:, 1]

        dz = local_offset % sz
        remainder = local_offset // sz
        dy = remainder % sy
        dx = remainder // sy

        # Compute actual voxel coordinates
        candidates = b_bbox_min[tri_local_idx] + torch.stack([dx, dy, dz], dim=1)  # (total_candidates, 3)

        # Run SAT intersection test
        aabb_min_f = candidates.to(dtype=tri_verts_voxel.dtype)
        aabb_max_f = aabb_min_f + 1.0
        tri_expanded = b_tri[tri_local_idx]  # (total_candidates, 3, 3)

        hits = _triangle_aabb_intersect_batched(tri_expanded, aabb_min_f, aabb_max_f)

        if hits.any():
            hit_cands = candidates[hits]  # (H, 3)
            hit_global_tri = b_global_idx[tri_local_idx[hits]]  # (H,)
            all_pairs.append(torch.cat([hit_global_tri.unsqueeze(1), hit_cands], dim=1))

        # Free intermediate tensors
        del tri_local_idx, local_offset, candidates, aabb_min_f, aabb_max_f, tri_expanded, hits

        batch_start = batch_end

    if not all_pairs:
        return (torch.empty((0, 2), dtype=torch.long, device=device),
                torch.empty((0, 3), dtype=torch.long, device=device))

    # Concatenate all pairs
    all_pairs = torch.cat(all_pairs, dim=0)  # (M, 4): [tri_idx, x, y, z]

    # Extract unique voxels
    voxel_coords = all_pairs[:, 1:4]  # (M, 3)
    unique_voxels, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)

    # Create pair indices: (triangle_idx, voxel_idx)
    pair_indices = torch.stack([all_pairs[:, 0], inverse], dim=1)

    return pair_indices, unique_voxels


def _accumulate_qef(
    tri_verts: torch.Tensor,
    tri_normals: torch.Tensor,
    pair_indices: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: torch.Tensor,
    num_voxels: int,
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate QEF (Quadratic Error Function) terms for each voxel.

    For each triangle-voxel pair, we compute:
    - ATA += face_weight * outer(normal, normal)
    - ATb += face_weight * dot(normal, p_local) * normal

    where p_local is the point on the triangle plane closest to voxel center.

    Args:
        tri_verts: (F, 3, 3) triangle vertices (in world space)
        tri_normals: (F, 3) triangle normals (normalized)
        pair_indices: (M, 2) tensor of (triangle_idx, voxel_idx)
        voxel_coords: (N, 3) voxel coordinates (integer)
        voxel_size: (3,) voxel size
        num_voxels: N, number of unique voxels
        face_weight: Weight for face constraints
        boundary_weight: Weight for boundary constraints

    Returns:
        ATA: (N, 9) flattened 3x3 matrices
        ATb: (N, 3) right-hand side vectors
    """
    device = tri_verts.device
    M = pair_indices.shape[0]

    # Initialize accumulators
    ATA = torch.zeros(num_voxels, 9, device=device, dtype=torch.float32)
    ATb = torch.zeros(num_voxels, 3, device=device, dtype=torch.float32)

    if M == 0:
        return ATA, ATb

    # Get triangle and voxel data for each pair
    tri_idx = pair_indices[:, 0]
    voxel_idx = pair_indices[:, 1]

    pair_tri_verts = tri_verts[tri_idx]  # (M, 3, 3)
    pair_tri_normals = tri_normals[tri_idx]  # (M, 3)
    pair_voxel_coords = voxel_coords[voxel_idx]  # (M, 3)

    # Compute voxel center in world space
    voxel_center = (pair_voxel_coords.float() + 0.5) * voxel_size.unsqueeze(0)  # (M, 3)

    # Find closest point on triangle to voxel center
    # Project voxel center onto triangle plane
    tri_v0 = pair_tri_verts[:, 0]  # (M, 3)
    tri_v1 = pair_tri_verts[:, 1]
    tri_v2 = pair_tri_verts[:, 2]

    # Vector from triangle origin to voxel center
    v = voxel_center - tri_v0

    # Project onto triangle plane
    dist = (v * pair_tri_normals).sum(dim=-1, keepdim=True)
    p_proj = voxel_center - dist * pair_tri_normals  # (M, 3)

    # Clip to triangle using barycentric coordinates
    edge0 = tri_v1 - tri_v0
    edge1 = tri_v2 - tri_v0

    # Compute barycentric coordinates from the projected point on triangle plane
    v_proj = p_proj - tri_v0

    # Compute barycentric coordinates
    d00 = (edge0 * edge0).sum(dim=-1)
    d01 = (edge0 * edge1).sum(dim=-1)
    d11 = (edge1 * edge1).sum(dim=-1)
    d20 = v_proj[:, 0] * edge0[:, 0] + v_proj[:, 1] * edge0[:, 1] + v_proj[:, 2] * edge0[:, 2]
    d21 = v_proj[:, 0] * edge1[:, 0] + v_proj[:, 1] * edge1[:, 1] + v_proj[:, 2] * edge1[:, 2]

    denom = d00 * d11 - d01 * d01
    denom = denom.clamp_min(1e-10)

    v_bary = (d11 * d20 - d01 * d21) / denom
    w_bary = (d00 * d21 - d01 * d20) / denom
    u_bary = 1.0 - v_bary - w_bary

    # Clamp barycentric coordinates
    u_bary = u_bary.clamp(0, 1)
    v_bary = v_bary.clamp(0, 1)
    w_bary = w_bary.clamp(0, 1)

    # Renormalize
    total = u_bary + v_bary + w_bary
    u_bary = u_bary / total
    v_bary = v_bary / total
    w_bary = w_bary / total

    # Compute closest point on triangle
    closest_point = u_bary.unsqueeze(-1) * tri_v0 + \
                    v_bary.unsqueeze(-1) * tri_v1 + \
                    w_bary.unsqueeze(-1) * tri_v2  # (M, 3)

    # Convert to voxel-local coordinates
    # p_local should be in [0, 1] within the voxel
    p_local = (closest_point / voxel_size.unsqueeze(0)) - pair_voxel_coords.float()  # (M, 3)
    p_local = p_local.clamp(0, 1)

    # Compute QEF terms
    n = pair_tri_normals  # (M, 3)
    d = (n * p_local).sum(dim=-1, keepdim=True)  # (M, 1)

    # ATA += outer(n, n)
    # Flattened as [n0*n0, n0*n1, n0*n2, n1*n0, n1*n1, n1*n2, n2*n0, n2*n1, n2*n2]
    outer_n = (n.unsqueeze(-1) * n.unsqueeze(-2)).view(M, 9)  # (M, 9)

    # ATb += d * n
    atb = d * n  # (M, 3)

    # Scatter add face terms to voxels
    ATA.scatter_add_(0, voxel_idx.unsqueeze(-1).expand(-1, 9),
                     face_weight * outer_n)
    ATb.scatter_add_(0, voxel_idx.unsqueeze(-1).expand(-1, 3),
                     face_weight * atb)

    if boundary_weight > 0:
        for axis in range(3):
            n_axis = torch.zeros((1, 3), device=device, dtype=torch.float32)
            n_axis[0, axis] = 1.0
            outer_axis = (n_axis.unsqueeze(-1) * n_axis.unsqueeze(-2)).view(1, 9)

            is_min = voxel_coords[:, axis] == 0
            if is_min.any():
                min_idx = is_min.nonzero(as_tuple=True)[0]
                ATA[min_idx] += boundary_weight * outer_axis
                # No ATb for min boundary — pulls toward 0

            is_max = voxel_coords[:, axis] == (grid_size[axis] - 1)
            if is_max.any():
                max_idx = is_max.nonzero(as_tuple=True)[0]
                ATA[max_idx] += boundary_weight * outer_axis
                # No ATb for max boundary — matches CPU behavior (pulls toward 0)

    return ATA, ATb


def _accumulate_intersect_qef(
    tri_verts: torch.Tensor,
    pair_indices: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    num_voxels: int,
    intersect_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate QEF terms using explicit edge-triangle intersections.
    """
    device = tri_verts.device
    ATA = torch.zeros(num_voxels, 9, device=device, dtype=torch.float32)
    ATb = torch.zeros(num_voxels, 3, device=device, dtype=torch.float32)

    if pair_indices.shape[0] == 0 or intersect_weight <= 0:
        return ATA, ATb

    tri_idx = pair_indices[:, 0]
    voxel_idx = pair_indices[:, 1]
    pair_tri_verts = tri_verts[tri_idx]
    pair_voxel_coords = voxel_coords[voxel_idx]
    voxel_origin = pair_voxel_coords.float() * voxel_size.unsqueeze(0)

    edge_dirs = torch.diag(voxel_size.float()).to(device)

    # Edge origin offsets from voxel corner for each axis:
    # x-edge at (i, j+1, k+1)*vs, y-edge at (i+1, j, k+1)*vs, z-edge at (i+1, j+1, k)*vs
    vs = voxel_size.float()
    edge_offsets = torch.zeros(3, 3, device=device, dtype=torch.float32)
    edge_offsets[0] = torch.tensor([0, vs[1], vs[2]], device=device)      # x-axis
    edge_offsets[1] = torch.tensor([vs[0], 0, vs[2]], device=device)      # y-axis
    edge_offsets[2] = torch.tensor([vs[0], vs[1], 0], device=device)      # z-axis

    v0 = pair_tri_verts[:, 0]
    v1 = pair_tri_verts[:, 1]
    v2 = pair_tri_verts[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0

    eps = 1e-6

    for axis in range(3):
        ray_origin = voxel_origin + edge_offsets[axis].unsqueeze(0)
        ray_dir = edge_dirs[axis].unsqueeze(0).expand(pair_indices.shape[0], -1)
        h = torch.cross(ray_dir, edge2, dim=-1)
        a = (edge1 * h).sum(dim=-1)
        parallel = a.abs() < 1e-10
        safe_a = torch.where(parallel, torch.ones_like(a), a)
        f = 1.0 / safe_a

        s = ray_origin - v0
        u = f * (s * h).sum(dim=-1)
        q = torch.cross(s, edge1, dim=-1)
        v = f * (ray_dir * q).sum(dim=-1)
        t = f * (edge2 * q).sum(dim=-1)

        valid = (~parallel) & (u >= -eps) & (u <= 1 + eps) & (v >= -eps) & (v <= 1 + eps) & ((u + v) <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)
        if not valid.any():
            continue

        hit_voxel_idx = voxel_idx[valid]
        hit_t = t[valid].clamp(0, 1)

        n_axis = torch.zeros((hit_t.shape[0], 3), device=device, dtype=torch.float32)
        n_axis[:, axis] = 1.0
        d = hit_t.unsqueeze(-1)

        outer_n = (n_axis.unsqueeze(-1) * n_axis.unsqueeze(-2)).view(-1, 9)
        atb = d * n_axis

        ATA.scatter_add_(0, hit_voxel_idx.unsqueeze(-1).expand(-1, 9), intersect_weight * outer_n)
        ATb.scatter_add_(0, hit_voxel_idx.unsqueeze(-1).expand(-1, 3), intersect_weight * atb)

    return ATA, ATb


def _solve_qef_batched(
    ATA: torch.Tensor,
    ATb: torch.Tensor,
    regularization_weight: float = 0.1,
) -> torch.Tensor:
    """
    Solve QEF for each voxel using batched matrix operations.

    Args:
        ATA: (N, 9) flattened 3x3 matrices
        ATb: (N, 3) right-hand side vectors
        regularization_weight: Weight for regularization term

    Returns:
        dual_vertices: (N, 3) dual vertex positions in [0, 1]
    """
    device = ATA.device
    N = ATA.shape[0]

    # Reshape ATA to (N, 3, 3)
    ATA_mat = ATA.view(N, 3, 3)

    # Add regularization: ATA += reg * I
    reg_eye = regularization_weight * torch.eye(3, device=device).unsqueeze(0)
    ATA_reg = ATA_mat + reg_eye

    # Add regularization to ATb: ATb += reg * 0.5 (pull toward center)
    ATb_reg = ATb + regularization_weight * 0.5

    # Solve using torch.linalg.solve
    try:
        dual_vertices = torch.linalg.solve(ATA_reg, ATb_reg)
    except RuntimeError:
        # Fallback for singular matrices: use pseudo-inverse
        ATA_pinv = torch.linalg.pinv(ATA_reg)
        dual_vertices = torch.bmm(ATA_pinv, ATb_reg.unsqueeze(-1)).squeeze(-1)

    # Clamp to [0, 1]
    dual_vertices = dual_vertices.clamp(0, 1)

    return dual_vertices


def _detect_edge_intersections(
    tri_verts: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: torch.Tensor,
    pair_indices: torch.Tensor,
    num_voxels: int,
) -> torch.Tensor:
    """
    Detect which voxel edges intersect with triangles.

    For each occupied voxel, test 3 axis-aligned edges (+x, +y, +z).
    Uses Moller-Trumbore ray-triangle intersection.

    CPU convention: t in [0, 1) (inclusive start, exclusive end), and
    edges at the grid boundary in perpendicular axes are excluded
    (they can't form valid quads in dual contouring).

    Args:
        tri_verts: (F, 3, 3) triangle vertices
        voxel_coords: (N, 3) voxel coordinates
        voxel_size: (3,) voxel size
        grid_size: (3,) grid dimensions
        pair_indices: (M, 2) triangle-voxel pairs
        num_voxels: N

    Returns:
        intersected: (N, 3) boolean tensor, True if edge is intersected
    """
    tri_verts = tri_verts.to(torch.float64)
    voxel_size = voxel_size.to(torch.float64)
    device = tri_verts.device
    N = num_voxels

    # Initialize intersection flags
    intersected = torch.zeros(N, 3, dtype=torch.bool, device=device)

    if pair_indices.shape[0] == 0:
        return intersected

    voxel_idx = pair_indices[:, 1]
    tri_idx = pair_indices[:, 0]

    # Edge segment vectors (+x, +y, +z) with voxel edge lengths
    edge_dirs = torch.diag(voxel_size).to(device)

    # Edge origin offsets from voxel corner for each axis:
    # In dual contouring, the x-edge of voxel (i,j,k) is at node (i, j+1, k+1)
    # y-edge at node (i+1, j, k+1), z-edge at node (i+1, j+1, k)
    edge_offsets = torch.zeros(3, 3, device=device, dtype=voxel_size.dtype)
    edge_offsets[0] = torch.tensor([0, voxel_size[1], voxel_size[2]], device=device, dtype=voxel_size.dtype)
    edge_offsets[1] = torch.tensor([voxel_size[0], 0, voxel_size[2]], device=device, dtype=voxel_size.dtype)
    edge_offsets[2] = torch.tensor([voxel_size[0], voxel_size[1], 0], device=device, dtype=voxel_size.dtype)

    M = pair_indices.shape[0]

    # Get data for each pair
    pair_tri_verts = tri_verts[tri_idx]  # (M, 3, 3)
    pair_voxel_coords = voxel_coords[voxel_idx]  # (M, 3)

    # Voxel corner in world space (the corner with smallest coordinates)
    voxel_corner = pair_voxel_coords.to(voxel_size.dtype) * voxel_size.unsqueeze(0)  # (M, 3)

    # Perpendicular axis pairs for each edge direction
    # x-edge needs y < gs-1 and z < gs-1
    # y-edge needs x < gs-1 and z < gs-1
    # z-edge needs x < gs-1 and y < gs-1
    perp_axes = [(1, 2), (0, 2), (0, 1)]

    # Test each edge direction
    for axis in range(3):
        # Exclude edges at grid boundary in perpendicular axes
        # (these can't form valid quads in dual contouring)
        pa, pb = perp_axes[axis]
        edge_valid_mask = (
            (pair_voxel_coords[:, pa] < grid_size[pa] - 1) &
            (pair_voxel_coords[:, pb] < grid_size[pb] - 1)
        )
        if not edge_valid_mask.any():
            continue

        # Edge origin is offset from voxel corner
        ray_origin = voxel_corner + edge_offsets[axis].unsqueeze(0)
        ray_dir = edge_dirs[axis].unsqueeze(0).expand(M, -1)

        # Moller-Trumbore ray-triangle intersection
        v0 = pair_tri_verts[:, 0]  # (M, 3)
        v1 = pair_tri_verts[:, 1]
        v2 = pair_tri_verts[:, 2]

        edge1 = v1 - v0
        edge2 = v2 - v0

        h = torch.cross(ray_dir, edge2, dim=-1)
        a = (edge1 * h).sum(dim=-1)

        # Parallel test
        parallel = a.abs() < 1e-10

        safe_a = torch.where(parallel, torch.ones_like(a), a)
        f = 1.0 / safe_a
        s = ray_origin - v0
        u = f * (s * h).sum(dim=-1)

        q = torch.cross(s, edge1, dim=-1)
        v = f * (ray_dir * q).sum(dim=-1)
        t = f * (edge2 * q).sum(dim=-1)

        eps = 1e-6
        # t in [0, 1): inclusive start, exclusive end (CPU convention)
        # u/v relaxed to include triangle edge/vertex hits
        valid = (edge_valid_mask &
                 ~parallel &
                 (u >= -eps) & (u <= 1 + eps) &
                 (v >= -eps) & (v <= 1 + eps) &
                 ((u + v) <= 1 + eps) &
                 (t >= -eps) & (t < 1 - eps))

        # Scatter results to voxels
        hit_voxels = voxel_idx[valid]
        intersected[hit_voxels, axis] = True

    return intersected


def mesh_to_flexible_dual_grid_gpu(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_range: torch.Tensor,
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
    regularization_weight: float = 0.1,
    timing: bool = False,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch GPU implementation of mesh_to_flexible_dual_grid.

    Voxelize a mesh into a sparse voxel grid with dual vertices computed
    via Quadratic Error Function minimization.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) mesh faces (vertex indices)
        voxel_size: (3,) size of each voxel
        grid_range: (2, 3) grid bounds [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        face_weight: Weight for face constraints in QEF
        boundary_weight: Weight for boundary constraints in QEF
        regularization_weight: Weight for regularization in QEF
        timing: Whether to print timing information
        chunk_size: Number of triangles to process per chunk

    Returns:
        voxel_indices: (N, 3) indices of occupied voxels
        dual_vertices: (N, 3) dual vertex positions within each voxel [0, 1]
        intersected_flag: (N, 3) boolean flags for edge intersections
    """
    device = vertices.device

    if timing:
        start_time = time.time()
        print("[GPU] Starting mesh_to_flexible_dual_grid_gpu")

    # Step 1: Triangle preparation
    tri_verts = vertices[faces.long()]  # (F, 3, 3)

    # Compute triangle normals
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]
    edge0 = v1 - v0
    edge1 = v2 - v0
    tri_normals = torch.cross(edge0, edge1, dim=-1)
    tri_normal_len = tri_normals.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    tri_normals = tri_normals / tri_normal_len  # Normalize

    # Convert to voxel coordinates
    tri_verts_voxel = tri_verts / voxel_size.unsqueeze(0).unsqueeze(0)

    grid_size = (grid_range[1] - grid_range[0]).long()

    if timing:
        torch.cuda.synchronize()
        step1_time = time.time()
        print(f"[GPU] Step 1 (Triangle prep): {step1_time - start_time:.3f}s, {tri_verts.shape[0]} triangles")

    # Step 2: Rasterize triangles to voxels
    pair_indices, voxel_coords = _rasterize_triangles_to_voxels(
        tri_verts_voxel, grid_size, chunk_size
    )

    num_voxels = voxel_coords.shape[0]

    if timing:
        torch.cuda.synchronize()
        step2_time = time.time()
        print(f"[GPU] Step 2 (Rasterization): {step2_time - step1_time:.3f}s, {num_voxels} voxels, {pair_indices.shape[0]} pairs")

    if num_voxels == 0:
        return (
            torch.empty((0, 3), dtype=torch.long, device=device),
            torch.empty((0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 3), dtype=torch.bool, device=device),
        )

    # Step 3: Face + Boundary QEF accumulation
    ATA, ATb = _accumulate_qef(
        tri_verts, tri_normals, pair_indices, voxel_coords,
        voxel_size, grid_size, num_voxels, face_weight, boundary_weight
    )

    if timing:
        torch.cuda.synchronize()
        step3_time = time.time()
        print(f"[GPU] Step 3 (Face+Boundary QEF): {step3_time - step2_time:.3f}s")

    # Step 4: Intersect QEF accumulation
    ATA_i, ATb_i = _accumulate_intersect_qef(
        tri_verts,
        pair_indices,
        voxel_coords,
        voxel_size,
        num_voxels,
        intersect_weight=face_weight,
    )
    ATA = ATA + ATA_i
    ATb = ATb + ATb_i

    if timing:
        torch.cuda.synchronize()
        step4_time = time.time()
        print(f"[GPU] Step 4 (Intersect QEF): {step4_time - step3_time:.3f}s")

    # Step 5: QEF Solve
    dual_vertices = _solve_qef_batched(ATA, ATb, regularization_weight)

    if timing:
        torch.cuda.synchronize()
        step5_time = time.time()
        print(f"[GPU] Step 5 (QEF solve): {step5_time - step4_time:.3f}s")

    # Step 6: Edge intersection detection
    intersected_flag = _detect_edge_intersections(
        tri_verts, voxel_coords, voxel_size, grid_size, pair_indices, num_voxels
    )

    if timing:
        torch.cuda.synchronize()
        step6_time = time.time()
        print(f"[GPU] Step 6 (Edge intersection): {step6_time - step5_time:.3f}s")
        print(f"[GPU] Total time: {step6_time - start_time:.3f}s")

    # CPU returns dual vertices as global normalized coordinates:
    # dv = (voxel_coord + local_pos) / grid_size, where local_pos is in [0, 1].
    # QEF solve produces local_pos, so transform to match CPU format.
    dual_vertices = (voxel_coords.float() + dual_vertices) / grid_size.float().unsqueeze(0)
    dual_vertices = dual_vertices.clamp(0, 1)

    # Convert voxel_coords to the expected output format
    voxel_indices = voxel_coords.long()

    return voxel_indices, dual_vertices, intersected_flag
