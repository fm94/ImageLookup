from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from tqdm import tqdm

from utils import load_images_from_disk, rescale_image, parse_args
from placeholders import Mode, ImageLookupConfig, Image, Source
from gt import GT
from pre_selector import PreSelector

# TODO: potential improvements
# store source descriptors to disk, load only needed ones
# instead of loop inference in the classifier, use vectorization
# add some sort of confidence to avoid drawing random polygons when nothing is found

class ImageLookup:
    """ main class of the ImageLookup 'clone' algorithm """
    
    def __init__(self, config: ImageLookupConfig) -> None:
        # load the config
        self.config = config
        # all sources will be saved here
        self.sources: list[Source] | None = None
        
        # create a source preselector if needed
        self._pre_selector: PreSelector | None = PreSelector(config.n_clusters) if config.use_clustering else None
     
    def index_sources(self, source_path: str | Path) -> None:
        """ a method called to index a list of source images """
        if isinstance(source_path, str):
            source_path = Path(source_path)
        images = load_images_from_disk(source_path)
        # create source placeholders
        self._create_sources(images)
        # build a clustering model for indexing if necessary
        if self.config.use_clustering:
            self._pre_selector.build_index_model()
        # extract SIFT features
        self._extract_descriptors()
        
    def query(self, images: list[str | Path], verbose: bool = True, out_path: str | Path | None = None) -> dict[str, str]:
        """ query method
        :param images: list of image paths to query
        :param verbose: whether to print the found results
        :param out_path: if provided, save images with found polygones drawn on them """
        
        result: dict[str, tuple[str, str]] = {}
        for query in tqdm(images, desc=">> Querying...", total=len(images)):
            img = cv2.imread(str(query), 0)
            scaled_img, desc, kps = self._scale_and_extract_kd(img, False)
            # use a label only if clustering is preused
            label = self._pre_selector.get_label(image=img) if self.config.use_clustering else None
            matches = []
            references = []
            # TODO: this can be improved
            for source in self.sources:
                # the difference between both modes here is which descriptors to use
                # the ones from the original image or the subimages
                # if there's any kind of exceptions then skip the source
                if self.config.mode == Mode.SNIPPETS:
                    for sub_source in source.sub_images:
                        # either there's no clustering, or there is clustering and the label match
                        if sub_source.label == label or not self.config.use_clustering:
                            try:
                                matches.append(self.config.matcher.knnMatch(sub_source.descriptors, desc, k=2))
                                references.append(sub_source)
                            except:
                                continue
                else:
                    try:
                        if source.image.label == label or not self.config.use_clustering:
                            matches.append(self.config.matcher.knnMatch(source.image.descriptors, desc, k=2))
                            references.append(source)
                    except:
                        continue
            # find the best descriptor match
            best_match_index = -1
            min_distance = float('inf')
            best_matches = None
            for img_idx, match_list in enumerate(matches):
                good_matches = []
                for match in match_list:
                    try:
                        m, n = match
                        if m.distance < self.config.distance_factor * n.distance:
                            good_matches.append(m)
                    # sometimes there aren't any matches, capture this with except but can be done differently
                    except:
                        continue
                if len(good_matches) > 0:
                    # the mean distance for every image is used as a metric
                    average_distance = np.mean([match.distance for match in good_matches])
                    if average_distance < min_distance:
                        min_distance = average_distance
                        best_match_index = img_idx
                        best_matches = good_matches
            # save results
            rect_coords = self._find_rect(references[best_match_index], kps, best_matches, scaled_img)
            result[query.stem] = (references[best_match_index].path.stem,
                                  ";".join([f"{int(x)},{int(y)}" for x, y in rect_coords.reshape(-1, 2)]))
            if out_path is not None:
                # draw the rectangle on the source image
                self._draw_rect(references[best_match_index], rect_coords, out_path, query.stem)
        # if verbose then simply print the dictionnary
        if verbose:
            for key, val in dict(sorted(result.items())).items():
                print(f"{key} : {val}")
        return result
    
    def _draw_rect(self, image: Image, coordinates: np.array, out_path: str | Path, query_name: str, color: tuple[int, int, int] = (0, 165, 255), thickness: int = 10) -> None:
        """ used to draw a polygon representing the region of interes on the original source image
        :param image: source image with the best match
        :param coordinates: list of points to draw the polygon
        :param out_path: output directory to store the drawing
        :param query_name: query file name
        :param color: color of the polygon/rectangle, default is orange
        :param tickeness: thickeness of the lines
        
        """
        # read the source image in full color
        data = cv2.imread(str(image.path))
        image_path = Path(image.path)
        filename_without_extension = image_path.stem
        # construct the output filename
        output_path = Path(out_path)  / f"{query_name}_{filename_without_extension}_detection.png"
        # there are two cases, when we have enough points to do a homoraphy and when not
        # if not then we simply draw a rectangle that convers the entire surface, some sort of convexhull
        
        # draw
        cv2.polylines(data, [np.int32(coordinates)], True, color, thickness, cv2.LINE_AA)
        cv2.imwrite(str(output_path), data)

    def _find_rect(self, image: Image, query_kps: np.array, best_matches: list[Any], scaled_image: np.array) -> np.array:
        """ map the found keypoints into a polygon on the original image 
        :param image: source image with the best match
        :param query_kps: query keypoints
        :param best_matches: -
        :param scaled_image: scaled-down query image
        
        """
        try:
            if self.config.mode == Mode.FULL:
                image = image.image
            src_pts = np.float32([image.keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([query_kps[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            # get the transformation matrix
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RHO, 1)
            h, w = scaled_image.shape
            # create dummy points which represent the query
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # transform to the original space
            dst = cv2.perspectiveTransform(pts, M)
            # re-up-scale
            dst = dst / (self.config.rescaling_factor / 100)
            # schift the detection back to the original coordinates if a subfigure is used
            if self.config.mode == Mode.SNIPPETS:
                dst[:,:,0] = dst[:,:,0] + image.x
                dst[:,:,1] = dst[:,:,1] + image.y
            # construct a convex hull to avoid tangled polygones
            hull = np.array(cv2.convexHull(dst))
        except Exception:
            # handle snippets with too few points
            if self.config.mode == Mode.SNIPPETS:
                hull = np.float32([[image.x, image.y], [image.x + image.w, image.y], [image.x + image.w, image.y + image.h], [image.x, image.y + image.h]]).reshape(-1, 1, 2)
            else:
                # handle full images
                try:
                    h, w = image.data.shape
                except:
                    h, w = image.image.data.shape
                h = h / (self.config.rescaling_factor / 100)
                w = w / (self.config.rescaling_factor / 100)
                hull = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        return hull
        
    
    def _extract_descriptors(self) -> None:
        """ use the SIFT algorithm to extract descriptors and keypoints """
        counter = 0
        for src in tqdm(self.sources, desc='>> Extracting descriptors...', total=len(self.sources)):
            # in snippet mode we need to go over all subfigures
            if self.config.mode == Mode.SNIPPETS:
                for sub_img in src.sub_images:
                    sub_img.data, sub_img.descriptors, sub_img.keypoints = self._scale_and_extract_kd(sub_img.data)
                    # get also the label
                    if self.config.use_clustering:
                        sub_img.label = self._pre_selector.get_label(index=counter)
                    counter += 1
            else:
                src.image.data, src.image.descriptors, src.image.keypoints = self._scale_and_extract_kd(src.image.data)
                # get also the label
                if self.config.use_clustering:
                    src.label = self._pre_selector.get_label(index=counter)
                counter += 1
                
    def _scale_and_extract_kd(self, image: np.array, is_source: bool = True) -> tuple[np.array, np.array]:
        """ a helper method used to first rescale the image to improve speed and then extract relevant info """
        image = rescale_image(image, self.config.rescaling_factor)
        # use different detectors based on whether it's a source image or a query
        if is_source:
            kps, des = self.config.source_detector.detectAndCompute(image, None)
        else:
            kps, des = self.config.query_detector.detectAndCompute(image, None)
        return image, np.float32(des) if des is not None else np.array([], dtype=np.float32), kps
    
    def _create_sources(self, images: list[Path]) -> None:
        """ create a list of sources. If necessary also extract subfigures from original images
        using countours """
        self.sources = []
        counter = 0
        for img in tqdm(images, desc='>> Creating sources...', total=len(images)):
            image_data = cv2.imread(str(img), 0)
            sub_images = None
            if self.config.mode == Mode.SNIPPETS:
                # if we are in the snippets mode, execute an opencv algo to find smaller figures
                sub_images = []
                _, thresh = cv2.threshold(image_data, self.config.contour_threshold, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area > self.config.area_threshold:
                        # derive the containing rectanle
                        x, y, w, h = cv2.boundingRect(contour)
                        subfigure = image_data[y:y+h, x:x+w]
                        if self.config.use_clustering:
                            # add to the queue
                            self._pre_selector.put_to_queue(subfigure)
                        sub_images.append(Image(subfigure, img, x=x, y=y, w=w, h=h))
                        counter += 1
            else:
                counter += 1
                if config.use_clustering:
                    # add to the queue
                    self._pre_selector.put_to_queue(image_data)
            # construct the object with all the data
            self.sources.append(Source(img, Image(image_data, img), sub_images))
        print(f'>> {counter} sources created')

    
if __name__ == '__main__':
    print('>> Starting...')
    config = parse_args()
    image_lookup = ImageLookup(config)
    image_lookup.index_sources(config.source_dir)
    results = image_lookup.query(load_images_from_disk(config.query_dir),
                               out_path=config.output_dir)
    gt = GT(config.gt_file)
    acc, correct = gt.evaluate(results)
    print(f'>> Done! Accuracy is {acc:.2f}% ({correct} are correct)')