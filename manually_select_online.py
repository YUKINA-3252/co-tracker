# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

# disable cudnn
# torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/apple_mask.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker_stride_4_wind_8.pth",
        help="cotracker model",
    )
    parser.add_argument("--grid_size", type=int, default=0, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame ",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    args = parser.parse_args()


    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebook/research/co-tracker", "cotracker2_online")

    # load the input video frame by frame
    video = read_video_from_path(args.video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    segm_mask = torch.from_numpy(segm_mask)[None, None]

    model = model.to(DEFAULT_DEVICE)

    windows_frames = []

    queries = torch.tensor([
        [0., 400., 350.]
    ])
    queries = queries.to(DEFAULT_DEVICE)


    def _process_step(window_frames, is_first_step, queries):
        video_chunk = (
            torch.tensor(np.stack(window_frames[model.step * 2:]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        ) # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None]
        )

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
                queries=queries[None],
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_trackes, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        queries = queries[None],
    )

    print("test")

    pred_tracks, pred_visibility = model(
        video,
        queries=queries[None]
    )
    print("computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(video, pred_tracks, pred_visibility)
