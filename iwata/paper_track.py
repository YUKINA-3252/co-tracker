import os
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
from IPython.display import HTML


video = read_video_from_path('../assets/tmp.mp4')
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

model = CoTrackerPredictor(
    checkpoint=os.path.join(
        '../checkpoints/cotracker2.pth'
        )
    )

if torch.cuda.is_available():
    model = model.cuda()
    video = video.cuda()

pred_tracks, pred_visibility = model(video, grid_size=30)

vis = Visualizer(save_dir='./videos', pad_value=100)
vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser')

queries = torch.tensor([
    [0, 400, 350],
    [10, 600, 500],
    [20, 750, 600],
    [30, 900, 200]
    ])
if torch.cuda.is_available():
    queries = queries.cuda()

grid_size = 30
grid_query_frame = 20

pred_tracks, pred_visivbility = model(video, grid_size=grid_size, grid_query_frame=grid_query_frame, backward_tracking=True)

vis = Visualizer(
    save_dir='./videos',
    pad_value=100
    )
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='grid_query_20')
