import torch

def sign(p1, p2, p3):
    return (p1[:,0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:,1] - p3[1])

def point_in_triangle(pts, v1, v2, v3, tol=1e-5):
    b1 = sign(pts, v1, v2) < tol
    b2 = sign(pts, v2, v3) < tol
    b3 = sign(pts, v3, v1) < tol

    # Check if points are inside the triangle
    inside = ((b1 == b2) & (b2 == b3))

    # Check if points are on the edges (within a tolerance)
    zero_tol = torch.tensor(tol, dtype=pts.dtype, device=pts.device)
    on_edges = torch.logical_or(torch.isclose(sign(pts, v1, v2), torch.tensor(0.0), atol=zero_tol),
                                torch.isclose(sign(pts, v2, v3), torch.tensor(0.0), atol=zero_tol)) | \
               torch.isclose(sign(pts, v3, v1), torch.tensor(0.0), atol=zero_tol)
    
    # get rid of "on_edges" points that are outside the triangle 
    # "edges" are per definition infinite lines, no? So we have to cut them off at the triangle
    x_max = torch.max(v1[0], torch.max(v2[0], v3[0])) + tol
    y_max = torch.max(v1[1], torch.max(v2[1], v3[1])) + tol
    x_min = torch.min(v1[0], torch.min(v2[0], v3[0])) - tol
    y_min = torch.min(v1[1], torch.min(v2[1], v3[1])) - tol
    outside = torch.logical_or(pts[:,0] > x_max, pts[:,0] < x_min) | torch.logical_or(pts[:,1] > y_max, pts[:,1] < y_min)
    on_edges = torch.logical_and(on_edges, ~outside)

    # Include points on edges in the inside result
    inside |= on_edges

    return inside

"""
# Example usage
triangle = torch.tensor([(0, 0), (4, 0), (2, 4)])  # Define triangle by its corner points
grid = regular_2D_mesh(100, 100, on_boundary=True, scale_x=4, scale_y=4)
grid_size = grid.size()
points = grid.view(2, -1).t()
result = point_in_triangle(points, *triangle)
result = result.view(grid_size[1], grid_size[2])
print(result)
"""