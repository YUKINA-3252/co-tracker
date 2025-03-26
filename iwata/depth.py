# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import open3d as o3d
from PIL import Image
import yaml

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

from geometry_msgs.msg import TransformStamped
from image_geometry.cameramodels import PinholeCameraModel
import rosbag
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import CameraInfo
import tf.transformations as tft
from tf2_msgs.msg import TFMessage
import tf2_ros

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# camera info
camera_info_file_path = "camera_info.txt"
with open(camera_info_file_path, "r") as file:   
 camera_info_lines = file.readlines()
camera_K_values = []
for line in camera_info_lines:
    if line.strip().startswith("K:"):
        camera_K_values = eval(line.split("K:")[1].strip())
        break
fx = camera_K_values[0]
fy = camera_K_values[4]
cx = camera_K_values[2]
cy = camera_K_values[5]

def get_average_depth(depth_img, x, y, window_size=10):
    half_size = window_size // 2
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, depth_img.shape[1])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, depth_img.shape[0])

    region = depth_img[y_min:y_max, x_min:x_max]
    valid_region = region[np.isfinite(region) & (region > 0)]

    return np.mean(valid_region)

def coords_to_depth(depth_img, img_x, img_y):
    x = (img_x - cx) / fx
    y = (img_y - cy) / fy
    z = depth_img[int(img_y.round()), int(img_x.round())]
    x *= z
    y *= z
    return (x, y, z)

def transform_to_base_coords(points):
    points_ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, points_ones])
    points_base_coords_homogeneous = (base_to_camera_transformation @ points_homogeneous.T).T
    points_base_coords = points_base_coords_homogeneous[:, :3]
    points_base_coords = points_base_coords[~np.isnan(points_base_coords).any(axis=1)]
    return points_base_coords

def calculate_line_intersection(p1, p2, q1, q2):
    p = np.array(p1, dtype=np.float32)
    r = np.array(p2, dtype=np.float32) - p
    q = np.array(q1, dtype=np.float32)
    s = np.array(q2, dtype=np.float32) - q

    rxs = np.cross(r, s)
    q_p = q - p
    q_pxr = np.cross(q_p, r)

    if np.isclose(rxs, 0):
        return None

    t = np.cross(q_p, s) / rxs
    u = q_pxr / rxs

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = p + t * r
        return intersection

    return None

def draw_dotted_line(image, point1, point2, color, thickness=1, gap=10):
    point1 = np.array(point1, dtype=np.float32)
    point2 = np.array(point2, dtype=np.float32)
    line_length = np.linalg.norm(point2 - point1)

    num_points = int(line_length // gap)

    for i in range(num_points + 1):
        t = i / num_points
        x = int((1 - t) * point1[0] + t * point2[0])
        y = int((1 - t) * point1[1] + t * point2[1])

        cv2.circle(image, (x, y), thickness, color, -1)

if __name__ == "__main__":
    rosbag_directory_path = "rosbag"
    rosbag_files = [os.path.join(rosbag_directory_path, f) for f in os.listdir(rosbag_directory_path) if os.path.isfile(os.path.join(rosbag_directory_path, f))]
    rosbag_latest_file = max(rosbag_files, key=os.path.getmtime)
    rosbag_latest_file_name = os.path.basename(rosbag_latest_file)
    rosbag_file = os.path.join(rosbag_directory_path, f'{rosbag_latest_file_name}')

    depth_image_topic = '/head_camera/depth/image_rect_raw'

    bridge = CvBridge()

    with rosbag.Bag(rosbag_file, 'r') as bag:
        end_time = bag.get_end_time()
        end_time_ros = rospy.Time(end_time)

        for topic, msg, t in bag.read_messages(topics=[depth_image_topic]):
            if t.to_sec() >= end_time_ros.to_sec() - 130:
                depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if msg.encoding == '16UC1':
                    depth_image = np.asarray(depth_image, dtype=np.float32)
                    depth_image /= 1000.0

    # transformation
    base_to_camera_transformation_translation = np.asarray([0.099087, 0.020357, 0.56758])
    base_to_camera_transformation_rotation = np.asarray([[0.017396, -0.896664, 0.442385], [-0.999853, 0.014582, -0.009762], [0.002302, -0.442487, -0.89678]])
    base_to_camera_transformation = np.eye(4)
    base_to_camera_transformation[:3, :3] = base_to_camera_transformation_rotation
    base_to_camera_transformation[:3, 3] = base_to_camera_transformation_translation

    # points = np.asarray([[305, 167],  [298, 354]])
    points = np.asarray([[376, 405], [384, 298]])
    point = np.mean(points, axis=0)
    point_3d = coords_to_depth(depth_image, point[0], point[1])
    points_3d = []
    points_3d.append(coords_to_depth(depth_image, point[1], point[0]))
    points_3d.append(coords_to_depth(depth_image, point[1], point[0]))
    points_3d = np.array(points_3d)
    result = transform_to_base_coords(points_3d)[0]
    print(result)
