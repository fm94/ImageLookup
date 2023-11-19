from dataclasses import dataclass, field
from pathlib import Path
import argparse

import cv2
import numpy as np

from placeholders import ImageLookupConfig, Mode
  

def rescale_image(image: np.array, scale_percent: int = 50) -> np.array:
    """ a utility function that rescales an images given a factor in % """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    rescaled_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return rescaled_image

def load_images_from_disk(path: str | Path, extension: str = "*.png") -> list[Path]:
    """ a utility function that is used to get all relevant images in a directory """
    print('>> Loading images...')
    if isinstance(path, str):
        path = Path(path)
    return list(path.glob(extension))

def parse_args() -> ImageLookupConfig:
    parser = argparse.ArgumentParser(description="ImageLookupConfig argument parser")
    parser.add_argument("--source_dir", default="../sources", type=str)
    parser.add_argument("--query_dir", default="../queries", type=str)
    parser.add_argument("--gt_file", default="../gt.txt", type=str)
    parser.add_argument("--output_dir", default="../drawings", type=str)
    parser.add_argument("--mode", default="SNIPPETS", choices=["SNIPPETS", "FULL"])
    parser.add_argument("--contour_threshold", type=int, default=250)
    parser.add_argument("--area_threshold", type=int, default=2000)
    parser.add_argument("--rescaling_factor", type=int, default=64)
    parser.add_argument("--use_clustering", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--n_feature_source", type=int)
    parser.add_argument("--n_feature_query", type=int, default=32)
    parser.add_argument("--source_detector", type=str)
    parser.add_argument("--query_detector", type=str)
    parser.add_argument("--matcher", type=str)
    parser.add_argument("--distance_factor", type=float, default=0.4)

    args = parser.parse_args()
    arg_dict = vars(args)
    arg_dict['mode'] = Mode.FULL if arg_dict['mode'] == 'FULL' else Mode.SNIPPETS
    return ImageLookupConfig(**arg_dict)