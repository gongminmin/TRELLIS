import os
os.environ['ATTN_BACKEND'] = 'xformers'     # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
images = [
    Image.open("assets/example_multi_image/character_1.png"),
    Image.open("assets/example_multi_image/character_2.png"),
    Image.open("assets/example_multi_image/character_3.png"),
]

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs: a list of meshes

video_color = render_utils.render_video(outputs[0])['color']
video_normal = render_utils.render_video(outputs[0])['normal']
video = [np.concatenate([frame_color, frame_normal], axis=1) for frame_color, frame_normal in zip(video_color, video_normal)]
imageio.mimsave("sample_multi.mp4", video, fps=30)
