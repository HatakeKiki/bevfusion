import os

import numpy as np
import torch

__all__ = ["load_augmented_point_cloud", "reduce_LiDAR_beams"]


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    # NOTE: path definition different from Tianwei's implementation.
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    tokens = path.split("/")
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + "_VIRTUAL",
        tokens[-1] + ".pkl.npy",
    )
    if not os.path.exists(seg_path):
        print(seg_path)
        assert False
        
    data_dict = np.load(seg_path, allow_pickle=True).item()
    
    # assert os.path.exists(wt_path)
    # wt_dir = "_VIRTUAL_WEIGHTS"
    # wt_path = os.path.join(
    #     *tokens[:-3],
    #     "virtual_points",
    #     tokens[-3],
    #     tokens[-2] + wt_dir,
    #     tokens[-1] + ".pkl.npy",
    # )
    # with open(wt_path, 'rb') as file:
    #     weight_dict = np.load(file, allow_pickle=True)
    #     weight_dict = weight_dict.reshape(1, -1)[0, 0]

    
    if len(data_dict["real_points_indice"]) > 0:
        average_time = points[data_dict["real_points_indice"]][:, 4].mean()
        # dims: 16
        # x, y, z, i, c*10, s
        # x, y, z, i, average_time, c*10, s, 1, 0
        virtual_points1 = np.concatenate(
            [
                data_dict["real_points"][:, :4],
                np.ones([data_dict["real_points"].shape[0], 1]) * average_time,
                data_dict["real_points"][:, 4:],
                # extra dim for gaussian weight
                # weight_dict['weights_painted'][:, 0].reshape(-1, 1),
                np.ones([data_dict["real_points"].shape[0], 1]),
                np.zeros([data_dict["real_points"].shape[0], 1]),
            ],
            axis=-1,
        )
        # virtual_points1[:, -3] *= weight_dict['weights_painted']
        # NOTE: add zero reflectance to virtual points instead of removing them from real points
        # dims: 15
        # x, y, z, 0(i), c*10, s
        # x, y, z, 0(i), average_time, c*10, s, 0, 0
        virtual_points2 = np.concatenate(
            [
                data_dict["virtual_points"][:, :3],
                np.zeros([data_dict["virtual_points"].shape[0], 1]),
                np.ones([data_dict["virtual_points"].shape[0], 1]) * average_time,
                data_dict["virtual_points"][:, 3:],
                # extra dim for gaussian weight
                # weight_dict['weights_virtual'][:, 0].reshape(-1, 1),
                np.zeros([data_dict["virtual_points"].shape[0], 2]),
            ],
            axis=-1,
        )
        # virtual_points2[:, -3] *= weight_dict['weights_virtual']
    # dims: 18
    # x, y, z, i, t, 0*10, 0(s), 0, 1
    points = np.concatenate(
        [
            points,
            np.zeros([points.shape[0], 10 + 1]),
            # extra dim for gaussian weight
            # np.zeros([points.shape[0], 1]),
            np.zeros([points.shape[0], 1]),
            np.ones([points.shape[0], 1]),
        ],
        axis=1,
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)

    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()

