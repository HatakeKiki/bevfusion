from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix 
from pyquaternion import Quaternion 
import numpy as np 
from functools import reduce

CAM_CHANS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

# ref_sample: sample data token, that is sample["data"]["LIDAR_TOP"] # pointsensor: lidar_top sample data, that is # nusc.get("sample_data", curr_sd_rec["prev"]) -> data token of a lidar top sample 
def find_closest_camera_tokens(nusc, pointsensor, ref_sample):
    lidar_timestamp = pointsensor["timestamp"]

    min_cams = {} 

    for chan in CAM_CHANS:
        camera_token = ref_sample['data'][chan]

        cam = nusc.get('sample_data', camera_token)
        min_diff = abs(lidar_timestamp - cam['timestamp'])
        min_cam = cam

        for i in range(6):  # nusc allows at most 6 previous camera frames 
            if cam['prev'] == "":
                break 

            cam = nusc.get('sample_data', cam['prev'])
            cam_timestamp = cam['timestamp']

            diff = abs(lidar_timestamp - cam_timestamp)

            if (diff < min_diff):
                min_diff = diff 
                min_cam = cam 
            
        min_cams[chan] = min_cam 

    return min_cams
