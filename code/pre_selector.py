from typing import Any

import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from placeholders import Source, Mode

class PreSelector:
    """ this class holds a model that given an image, it gives a label
    similar images shall have the same label thus, it helps reducing the search domain when querying.
    This simply uses clustering in the backend """
    
    # I am using a mobilenetv2 feature extractor, already pretained and needs to be downloaded
    model: Any = MobileNetV2(weights='imagenet', include_top=False)
    
    def __init__(self, n_classes: int = 3):
        # number of classes
        self.n_classes = n_classes
    
        # defined all models required
        self._kmeans: Any = None
        self._cluster_labels: Any = None
        self._classifier: Any = None
        self._pca: Any = None
        self._index_features: list[tf.Tensor] = []
        
    def get_label(self, image = None, index: int | None = None) -> int:
        """ get the label of a single image, stored in a numpy array
        if instead index is used, then refer to the source images. Used for efficiency because they
        are indexed at the beggining """
        if index is not None:
            features = self._index_features[index]
        elif image is not None:
            features = self._inference(image)
        else:
            raise ValueError
        features_reduced = self._pca.transform([features])
        predicted_cluster = self._classifier.predict(features_reduced)
        return predicted_cluster[0]
    
    def put_to_queue(self, image: np.array, is_source: bool = True) -> None:
        """ send a source image to the queue for the following tasks """
        image_features = self._inference(image)
        if is_source:
            self._index_features.append(image_features)
    
    def build_index_model(self) -> None:
        """" create a clustering model. first get good centroids, reduce dims and then cluster """
        print('>> Building Index Clustering Models...')
        self._find_centroids()
        reduced_features = self._reduce_dims()
        self._train_classifier(reduced_features)
    
    def _inference(self, image: np.array) -> tf.Tensor:
        """" run an image through mobilenet to get the latent space representation """
        image = cv2.resize(image, (224, 224))
        img_rgb = np.stack((image, image, image), axis=-1)
        img_batched = img_rgb[np.newaxis, :, :, :]
        image = preprocess_input(img_batched)
        features = self.model.predict(image)
        return features.flatten()
        
    def _find_centroids(self) -> None:
        """ find suitable centroids given a dataset """
        print('>> Finding Centroids...')
        self._kmeans = KMeans(n_clusters=self.n_classes)
        self._cluster_labels = self._kmeans.fit_predict(self._index_features)
        
    def _reduce_dims(self) -> np.array:
        """ use PCA for dim reduction """
        print('>> Reducing Dimentionality...')
        self._pca = PCA(n_components=50)
        return self._pca.fit_transform(self._index_features)
    
    def _train_classifier(self, reduced_features: np.array) -> None:
        """ train a lazy knn classifier """
        print('>> Training the Clustering Model...')
        self._classifier = KNeighborsClassifier()
        self._classifier.fit(reduced_features, self._cluster_labels)