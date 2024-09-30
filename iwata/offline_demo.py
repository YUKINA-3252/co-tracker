import numpy as np
import os
from PIL import Image
import torch

from base64 import b64encode
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path

video = read_video_from_path('../assets/hiro_demo_2024-09-13-13-43-10.mp4')
# video = read_video_from_path('../assets/apple.mp4')
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

model = CoTrackerPredictor(
    checkpoint=os.path.join(
        '../checkpoints/cotracker2.pth'
        )
    )

if torch.cuda.is_available():
    model = model.cuda()
    video = video.cuda()

input_mask = '../assets/mask_for_co_tracker.png'
# input_mask = '../assets/apple_mask.png'
segm_mask = np.array(Image.open(input_mask))

grid_size = 30
pred_tracks, pred_visibility = model(video, grid_size=grid_size, segm_mask=torch.from_numpy(segm_mask)[None, None])
vis = Visualizer(
    save_dir='./videos',
    pad_value=100,
    linewidth=2,
    )
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='segm_grid')
