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
from PIL import Image

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

from image_geometry.cameramodels import PinholeCameraModel
import rosbag
from cv_bridge import CvBridge
import rospy

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def coords_to_depth(depth_img, track_x, track_y):
    height, width, _ = depth_img.shape
    cameramodel = PinholeCameraModel()
    x = (track_x - cameramodel.cx()) / cameramodel.fx()
    y = (track_y - cameramodel.cy()) / cameramodel.fy()
    z = depth_img.reshape(-1)[track_y * width + track_x]
    x *= z
    y *= z
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="../assets/hiro_demo_2024-09-13-13-43-10_cut.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=50, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

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

    # process rosbag
    bag_file = '../assets/hiro_demo_2024-09-27-11-33-18.bag'
    depth_image_topic = '/head_camera/depth/image_raw'

    bridge = CvBridge()

    with rosbag.Bag(bag_file, 'r') as bag:
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        start_time_ros = rospy.Time(start_time)
        end_time_ros = rospy.Time(end_time)
        for topic, msg, t in bag.read_messages(topics=[depth_image_topic]):
            if t >= start_time_ros:
                depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv2.imwrite('depth_image_at_start_time.png', depth_image)
                break
        for topic, msg, t in bag.read_messages(topics=[depth_image_topic]):
            if t.to_sec() >= end_time_ros.to_sec() - 1:
                depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv2.imwrite('depth_image_at_end_time.png', depth_image)
                break

    input_mask_list = ['../assets/paper_mask_for_co_tracker.png']

    for idx, input_mask in enumerate(input_mask_list):

        # segmentation mask
        input_mask = input_mask
        segm_mask = np.array(Image.open(input_mask))

        save_video_name = f'../saved_videos/video{idx}'
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

        file_name = "tmp.png"
        B, T, C, H, W = video.shape
        image = np.zeros((H, W), dtype=np.uint8)
        for i in range(pred_tracks.shape[1]):
            x,y = int(pred_tracks[0][i][0][1].round()), int(pred_tracks[0][i][0][0].round())
            image[x, y] = 255
        cv2.imwrite(file_name, image)

        vis = Visualizer(save_dir="../saved_videos", pad_value=120, linewidth=3)
        vis.visualize(video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=save_video_name)
