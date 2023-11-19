from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2

class Mode(IntEnum):
    """ this enum class is used to run the algorithm in two modes:
    full: using entire images even if they contain multiple subfigures 
    snippets: first split all the images into sub figures and then run the algorithm
    the second option is better. """
    FULL = 0
    SNIPPETS = 1

@dataclass
class ImageLookupConfig:
    """ a class that stores all relevant configs
    numbers are obtained by trial and error - they can be imporved """
    # source images directory
    source_dir: str 
    # query images directory
    query_dir: str
    # ground truth file path
    gt_file: str
    # path where drawings will be saved to
    output_dir: str | None = None
    # run on snippet or full images
    mode: Mode = Mode.SNIPPETS
    # when extracting sub figures, how to binarize the images
    contour_threshold: int | None = 250
    # drop too small subfigures
    area_threshold: int | None = 2000
    # rescaling factor in %
    rescaling_factor: int = 64
    # whether to use clustering to reduce the number of index images
    use_clustering: bool = False
    # if the above is true, how many labels to use
    n_clusters: int = 3
    # number of features used for source images
    n_feature_source: int | None = None
    # number of features used for query images, typically less
    n_feature_query: int | None = 32
    # the SIFT detector for source images
    source_detector: Any = None
    # the SIFT detector for query images
    query_detector: Any = None
    # the matcher used for the feature search
    matcher: Any = None
    # distance factor used to filter out bad matches
    distance_factor: float = 0.4

    def __post_init__(self):
        # setup all models
        if self.n_feature_source is not None and self.source_detector is None:
            self.source_detector = cv2.SIFT_create(self.n_feature_source)
        else:
            self.source_detector = cv2.SIFT_create()
        if self.n_feature_query is not None and self.query_detector is None:
            self.query_detector = cv2.SIFT_create(self.n_feature_query)
        else:
            self.query_detector = cv2.SIFT_create()
        if self.matcher is None:
            index_params = dict(algorithm=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

@dataclass
class Image:
    """ a class that stores a single image and all relevant infos """
    # the actual image data
    data: np.array
    # the path of the image or parent image if this is a subfigure
    path: str | Path
    # array of descriptors
    descriptors: Any = None
    # array of keypoints
    keypoints: Any = None
    # label, used for clustering. Visually similar features have similar label
    label: int | None = None
    # if this is a subfigure, then coordinates of the subfigure in the original figure
    x: int | None = None
    y: int | None = None
    w: int | None = None
    h: int | None = None


@dataclass
class Source:
    """ a class that stores a single source image used for indexing """
    # path of the image
    path: str | Path
    # the image stored as an Image object
    image: Image
    # if it contains subfigures then they are stored here
    sub_images: list[Image] | None = None