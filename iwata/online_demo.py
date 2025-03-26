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
    directory_path = "assets"
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    latest_file = max(files, key=os.path.getmtime)
    latest_file_name = os.path.basename(latest_file)
    latest_file_path = os.path.join(directory_path, f'{latest_file_name}')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default=latest_file_path,
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=200, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument("--target_object", type=str, default="box", help="Target Object to track")

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame, segm_mask):
        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            segm_mask=segm_mask
        )

    # # process rosbag
    # rosbag_directory_path = "rosbag"
    # rosbag_files = [os.path.join(rosbag_directory_path, f) for f in os.listdir(rosbag_directory_path) if os.path.isfile(os.path.join(rosbag_directory_path, f))]
    # rosbag_latest_file = max(rosbag_files, key=os.path.getmtime)
    # rosbag_latest_file_name = os.path.basename(rosbag_latest_file)
    # rosbag_file = os.path.join(rosbag_directory_path, f'{rosbag_latest_file_name}')

    depth_image_topic = '/head_camera/depth/image_rect_raw'
    camera_info_topic = '/head_camera/depth/camera_info'
    camera_info_msg = None

    bridge = CvBridge()

    # transformation
    base_to_camera_transformation_translation = np.asarray([0.099087, 0.020357, 0.56758])
    base_to_camera_transformation_rotation = np.asarray([[0.017396, -0.896664, 0.442385], [-0.999853, 0.014582, -0.009762], [0.002302, -0.442487, -0.89678]])
    base_to_camera_transformation = np.eye(4)
    base_to_camera_transformation[:3, :3] = base_to_camera_transformation_rotation
    base_to_camera_transformation[:3, 3] = base_to_camera_transformation_translation

    if args.target_object == "box":
        input_mask_list = ['/home/iwata/co-tracker/iwata/edge_image/box_mask.png']
    elif args.target_object == "paper":
        input_mask_list = ['/home/iwata/co-tracker/iwata/edge_image/only_paper_mask.png']
    elif args.target_object == "top_paper":
        input_mask_list = ['/home/iwata/co-tracker/iwata/edge_image/top_paper_mask.png']
        # input_mask_list = ['/home/iwata/Grounded-Segment-Anything/co-tracker/mask_top_paper_1.png']

    for idx, input_mask in enumerate(input_mask_list):

        # segmentation mask
        input_mask = input_mask
        segm_mask = np.array(Image.open(input_mask))

        save_video_name = args.target_object
        meta_data_fps = iio.immeta(args.video_path)["fps"]
        print(f"video's fps is {meta_data_fps}")

        # Iterating over video frames, processing one window at a time:
        is_first_step = True
        for i, frame in enumerate(
            iio.imiter(
                args.video_path,
                plugin="FFMPEG",
            )
        ):
            if i % model.step == 0 and i != 0:
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                    grid_size=args.grid_size,
                    grid_query_frame=args.grid_query_frame,
                    segm_mask=torch.from_numpy(segm_mask)[None, None]
                )
                is_first_step = False
            window_frames.append(frame)
        # Processing the final video frames in case video length is not a multiple of model.step
        pred_tracks, pred_visibility = _process_step(
            window_frames[-(i % model.step) - model.step - 1 :],
            is_first_step,
            grid_size=args.grid_size,
            grid_query_frame=args.grid_query_frame,
            segm_mask=torch.from_numpy(segm_mask)[None, None]
        )

        print("Tracks are computed")

        # save a video with predicted tracks
        # seq_name = os.path.splitext(args.video_path.split("/")[-1])[0]
        video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]

        # file_name = "tmp.png"
        # B, T, C, H, W = video.shape
        # image = np.zeros((H, W), dtype=np.uint8)
        # for i in range(pred_tracks.shape[2]):
        #     x,y = int(pred_tracks[0][0][i][1].round()), int(pred_tracks[0][0][i][0].round())
        #     image[x, y] = 255
        #     x,y = int(pred_tracks[0][-1][i][1].round()), int(pred_tracks[0][-1][i][0].round())
        #     image[x, y] = 255
        # cv2.imwrite(file_name, image)

        # k = 90
        # # start 3d points
        # start_pred_tracks_points = []
        # with rosbag.Bag(rosbag_file, 'r') as bag:
        #     start_time = bag.get_start_time()
        #     start_time_ros = rospy.Time(start_time)
        #     for topic, msg, t in bag.read_messages(topics=[depth_image_topic]):
        #         if t.to_sec() >= start_time_ros.to_sec() + 3:
        #             depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #             if msg.encoding == '16UC1':
        #                 depth_image = np.asarray(depth_image, dtype=np.float32)
        #                 depth_image /= 1000.0
        #             elif msg.encoding != '32FC1':
        #                 rospy.logerr('Unsupported depth encoding: %s' % msg.encoding)
        #             for i in range(pred_tracks.shape[2]):
        #                 point = coords_to_depth(depth_image, pred_tracks[0][0][i][0], pred_tracks[0][0][i][1])
        #                 if pred_visibility[0][k][i]:
        #                     start_pred_tracks_points.append(point)
        #             break
        # start_pred_tracks_points = torch.tensor(start_pred_tracks_points)
        # start_pred_tracks_points = start_pred_tracks_points.numpy()

        # # end 3d points
        # end_pred_tracks_points = []
        # with rosbag.Bag(rosbag_file, 'r') as bag:
        #     end_time = bag.get_end_time()
        #     end_time_ros = rospy.Time(end_time)
        #     for topic, msg, t in bag.read_messages(topics=[depth_image_topic]):
        #         if t.to_sec() >= end_time_ros.to_sec() - 10:
        #             depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #             if msg.encoding == '16UC1':
        #                 depth_image = np.asarray(depth_image, dtype=np.float32)
        #                 depth_image /= 1000.0
        #             elif msg.encoding != '32FC1':
        #                 rospy.logerr('Unsupported depth encoding: %s' % msg.encoding)
        #             for i in range(pred_tracks.shape[2]):
        #                 point = coords_to_depth(depth_image, pred_tracks[0][k][i][0], pred_tracks[0][k][i][1])
        #                 if pred_visibility[0][k][i]:
        #                     end_pred_tracks_points.append(point)
        #             break

        # paper_edge = []
        # for i in range(pred_tracks.shape[2]):
        #     if not(pred_visibility[0][-210][i]):
        #         paper_edge.append([pred_tracks[0][-210][i][0], pred_tracks[0][-210][i][1]])
        # org_image = cv2.imread("edge_image/output.png", cv2.IMREAD_COLOR)

        # edge_region_image = np.zeros((org_image.shape[0], org_image.shape[1]), dtype=np.uint8)
        # for i in range(len(paper_edge)):
        #     edge_region_image[int(paper_edge[i][1].round()), int(paper_edge[i][0].round())] = 255
        #     cv2.circle(edge_region_image, (int(paper_edge[i][0].round()), int(paper_edge[i][1].round())), 5, (255, 255, 255), -1)
        # cv2.imwrite("edge_image/result.png", edge_region_image)

        # mask = cv2.imread('edge_image/result.png', cv2.IMREAD_GRAYSCALE)
        # _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # # threshold = 3
        # # filtered = np.where(dist_transform > threshold, 255, 0).astype(np.uint8)
        # kernel = np.ones((3, 3), np.uint8)
        # dilated = cv2.dilate(binary, kernel, iterations=3)
        # # filtered = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # cv2.imwrite('edge_image/dilated.png', dilated)
        # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # max_contour = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(max_contour)
        # result = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # epsilon = 0.02 * cv2.arcLength(max_contour, True)
        # approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
        # moments = cv2.moments(approx_contour)
        # if moments['m00'] != 0:
        #     cx = int(moments['m10'] / moments['m00'])
        #     cy = int(moments['m01'] / moments['m00'])

        # right_side_lines = []
        # for i in range(len(approx_contour)):
        #     p1 = approx_contour[i][0]
        #     p2 = approx_contour[(i + 1) % len(approx_contour)][0]

        #     midpoint_x = (p1[0] + p2[0]) / 2
        #     midpoint_y = (p1[1] + p2[1]) / 2
        #     length = np.linalg.norm(p2 - p1)
        #     if midpoint_x > cx:
        #         right_side_lines.append((p1, p2, length))

        # if right_side_lines:
        #     longest_line = max(right_side_lines, key=lambda line: line[2])
        #     p1, p2, _ = longest_line

        # # left_side_lines = []
        # # for i in range(len(approx_contour)):
        # #     p1 = approx_contour[i][0]
        # #     p2 = approx_contour[(i + 1) % len(approx_contour)][0]

        # #     midpoint_x = (p1[0] + p2[0]) / 2
        # #     midpoint_y = (p1[1] + p2[1]) / 2
        # #     length = np.linalg.norm(p2 - p1)
        # #     if midpoint_x < cx:
        # #         left_side_lines.append((p1, p2, length))

        # # if left_side_lines:
        # #     longest_line = max(left_side_lines, key=lambda line: line[2])
        # #     p1, p2, _ = longest_line

        # cv2.line(org_image, tuple(p1), tuple(p2), (255, 0, 0), 7)
        # cv2.imwrite('edge_image/line.png', org_image)

        # output = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(output, [approx_contour], -1, (255, 0, 0), 7)
        # cv2.circle(output, (cx, cy), 10, (0, 0, 255), -1)
        # cv2.imwrite('edge_image/contours.png', output)

        # # expand line
        # paper_mask = cv2.imread('edge_image/top_paper_mask_2.png', cv2.IMREAD_GRAYSCALE)
        # contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # max_contour = max(contours, key=cv2.contourArea)
        # line_direction = p1 - p2
        # line_direction = line_direction / np.linalg.norm(line_direction)
        # extended_start = p1 - 1000 * line_direction
        # extended_end = p2 + 1000 * line_direction
        # intersections = []
        # for i in range(len(max_contour)):
        #     contour_start = max_contour[i][0]
        #     contour_end = max_contour[(i + 1) % len(max_contour)][0]

        #     intersect = calculate_line_intersection(
        #         extended_start, extended_end,
        #         contour_start, contour_end
        #     )

        #     if intersect is not None:
        #         intersections.append(intersect)
        # expand_line_image = cv2.imread("edge_image/output.png", cv2.IMREAD_COLOR)
        # for point in intersections:
        #     point = tuple(map(int, point))
        #     cv2.circle(expand_line_image, point, 20, (0, 0, 255), -1)
        # # print(tuple(p1), tuple(map(int, intersections[0])))
        # draw_dotted_line(expand_line_image, tuple(p1), tuple(map(int, intersections[1])), (0, 255, 0), thickness=7, gap=10)
        # draw_dotted_line(expand_line_image, tuple(p2), tuple(map(int, intersections[0])), (0, 255, 0), thickness=7, gap=10)
        # cv2.line(expand_line_image, tuple(p1), tuple(p2), (255, 0, 0), 7)
        # print(tuple(p1), tuple(p2))
        # cv2.imwrite('edge_image/expand_line.png', expand_line_image)
        # cv2.rectangle(org_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # end_pred_tracks_points = torch.tensor(end_pred_tracks_points)
        # end_pred_tracks_points = end_pred_tracks_points.numpy()

        # start_pred_tracks_points_base_coords = transform_to_base_coords(start_pred_tracks_points)
        # end_pred_tracks_points_base_coords = transform_to_base_coords(end_pred_tracks_points)

        # start_point_cloud = o3d.geometry.PointCloud()
        # start_point_cloud.points = o3d.utility.Vector3dVector(start_pred_tracks_points_base_coords)
        # start_plane_model, start_inliers = start_point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        # end_point_cloud = o3d.geometry.PointCloud()
        # end_point_cloud.points = o3d.utility.Vector3dVector(end_pred_tracks_points_base_coords)
        # end_plane_model, end_inliers = end_point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

        # common_inliers = np.intersect1d(start_inliers, end_inliers)
        # start_plane_points = start_point_cloud.select_by_index(common_inliers)
        # start_pred_tracks_points_base_coords = np.asarray(start_plane_points.points)
        # end_plane_points = end_point_cloud.select_by_index(common_inliers)
        # end_pred_tracks_points_base_coords = np.asarray(end_plane_points.points)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # start_pred_tracks_points_base_coords = start_pred_tracks_points_base_coords - np.asarray([0.099, 0.020, 0.568])
        # end_pred_tracks_points_base_coords = end_pred_tracks_points_base_coords - np.asarray([0.099, 0.020, 0.568])
        # x = start_pred_tracks_points_base_coords[:, 0]
        # y = start_pred_tracks_points_base_coords[:, 1]
        # z = start_pred_tracks_points_base_coords[:, 2]
        # X = end_pred_tracks_points_base_coords[:, 0]
        # Y = end_pred_tracks_points_base_coords[:, 1]
        # Z = end_pred_tracks_points_base_coords[:, 2]
        # ax.scatter(x, y, z, c='b', marker='o')
        # ax.scatter(X, Y, Z, c='r', marker='o')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.axis('equal')
        # plt.show()

        # # estimate rigid transform
        # p_start = np.mean(start_pred_tracks_points_base_coords, axis=0)
        # p_end = np.mean(end_pred_tracks_points_base_coords, axis=0)
        # start_prime = start_pred_tracks_points_base_coords - p_start
        # end_prime = end_pred_tracks_points_base_coords - p_end
        # H = start_prime.T @ end_prime
        # U, _, Vt = np.linalg.svd(H)
        # R = Vt.T @ U.T
        # t = p_end - R @ p_start
        # # M = cv2.estimateAffine3D(start_pred_tracks_points_base_coords, end_pred_tracks_points_base_coords)
        # print(R, t)
        # start_centroid = np.mean(start_pred_tracks_points_base_coords, axis=0)
        # end_centroid = np.mean(end_pred_tracks_points_base_coords, axis=0)
        # start_prime = start_pred_tracks_points_base_coords - start_centroid
        # end_prime = end_pred_tracks_points_base_coords - end_centroid
        # H = np.dot(start_prime.T, end_prime)
        # U, _, Vt = np.linalg.svd(H)
        # R = np.dot(Vt.T, U.T)
        # if np.linalg.det(R) < 0:
        #     Vt[-1, :] *= -1
        # R = np.dot(Vt.T, U.T)
        # t = end_centroid - np.dot(R, start_centroid)

        vis = Visualizer(save_dir="saved_videos", pad_value=120, linewidth=3)
        vis.visualize(video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=save_video_name)

        if args.target_object == "box":
            data = {"R": R.tolist(),
                    "t": t.tolist(),
            }
            with open(os.path.join("/home/iwata/Grounded-Segment-Anything/ros/fold", "updated_box_coords.yaml"), "w") as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)
        if args.target_object == "paper":
            data = {"R": R.tolist(),
                    "t": t.tolist(),
            }
            with open(os.path.join("/home/iwata/Grounded-Segment-Anything/ros/fold", "updated_paper_coords.yaml"), "w") as yaml_file:
                yaml.dump(data, yaml_file, default_flow_style=False)
