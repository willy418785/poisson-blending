import numpy as np
import copy
from operator import add

def implant_ROI_of_src_in_tar(src, tar, start_coor, end_coor, mask=None):
    src_copy = copy.copy(src)
    tar_copy = copy.copy(tar)
    src_height, src_width, src_channels = src.shape
    tar_height, tar_width, tar_channels = tar.shape
    assert src_channels == tar_channels
    assert (src_height, src_width) <= (tar_height, tar_width)
    assert start_coor >= (0, 0)
    assert end_coor <= (tar_width, tar_height)
    assert tuple([end - start for start, end in zip(start_coor, end_coor)]) == (src_width, src_height)
    if mask is None:
        tar_copy[start_coor[1]:end_coor[1], start_coor[0]:end_coor[0]] = src_copy
        return tar_copy
    else:
        assert src.shape == mask.shape
        src_copy[mask != 255] = tar_copy[start_coor[1]:end_coor[1], start_coor[0]:end_coor[0]][mask != 255]
        tar_copy[start_coor[1]:end_coor[1], start_coor[0]:end_coor[0]] = src_copy
        return tar_copy

def get_boundary_point_on_tar_given_src_w_offset(src, tar, offset):
    src_height, src_width, src_channels = src.shape
    tar_height, tar_width, tar_channels = tar.shape
    start_x, start_y = offset
    end_x, end_y = start_x + src_width, start_y + src_height
    if start_x < 0:
        start_x = 0
        end_x = src_width
    elif end_x > tar_width:
        end_x = tar_width
        start_x = tar_width - src_width
    if start_y < 0:
        start_y = 0
        end_y = src_height
    elif end_y > tar_height:
        end_y = tar_height
        start_y = tar_height - src_height
    return (start_x, start_y), (end_x, end_y)

def get_four_neighbors(center, lower_bound, upper_bound):
    four_neighbors = [tuple(map(add, center, (1, 0))), tuple(map(add, center, (-1, 0))), tuple(map(add, center, (0, 1))), tuple(map(add, center, (0, -1)))]
    for neighbor in four_neighbors:
        if any([i >= bound for i, bound in zip(neighbor, upper_bound)]):
            four_neighbors.remove(neighbor)
        if any([i < bound for i, bound in zip(neighbor, lower_bound)]):
            four_neighbors.remove(neighbor)
    return four_neighbors

def gauss_seidel(A, b, tolerance, max_iterations, x):
    # x is the initial condition
    iter1 = 0
    # Iterate
    for k in range(max_iterations):
        iter1 = iter1 + 1
        print("The solution vector in iteration", iter1, "is:", x)
        x_old = x.copy()

        # Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i + 1):], x_old[(i + 1):])) / A[i, i]

        # Stop condition
        # LnormInf corresponds to the absolute value of the greatest element of the vector.

        LnormInf = max(abs((x - x_old))) / max(abs(x_old))
        print("The L infinity norm in iteration", iter1, "is:", LnormInf)
        if LnormInf < tolerance:
            break

    return x