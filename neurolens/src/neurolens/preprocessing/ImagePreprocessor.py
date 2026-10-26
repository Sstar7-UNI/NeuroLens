from .ImageOperator import ImageOperator
from ..data_management.ImageRecord import ImageRecord
from typing import List
import tensorflow as tf
from tensorflow.keras import layers

class ImagePreprocessor:
    def __init__(self,width:int =224, height:int = 224):
        self._height = height
        self._width = width
        self._pipeline: List[ImageOperator] = []
        self._normalization = layers.Normalization(axis=-1)
        self._rotation = RandomRotation(0.05, fill_mode="constant")
        self._translation = layers.RandomTranslation(0.05,0.05,fill_mode = "constant")
        self._zoom = layers.RandomZoom((-0.1,0.1),fill_mode="constat")

    def decode(self, path:str) -> tf.Tensor:
        contents = tf.io.read_file(path)
        image = tf.io.decode_png(contents, channels=1,expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize_with_pad(image, self._height, self._width)
    def preprocess(self, path: str) -> tf.Tensor:
        image = self.decode(path)
        for op in self._pipeline:
            image = op.apply(image)
        return tf.squeeze(self._normalization(tf.expand_dims(image, 0)), 0)
    def fitNormalization(self,samples:List[ImageRecord]) -> None:
        paths = [r.imagePath for r in samples]
        ds = tf.data.Dataset.from_tensor_slices(paths).map(self.decode).batch(32)
        self._normalization.adapt(ds)

    def augment(self,image:tf.Tensor,copies:Integer) -> List[tf.Tensor]:
        out = []
        for _ in range(copies):
            x = tf.expand_dims(image,0)
            x = self._rotation(x,training=True)
            x = self._zoom(x,training=True)
            out.append(tf.squeeze(x,0))
        return out
    def buildDataset(self,samples:List[ImageRecord]) -> Dataset:
        paths = [r.imagePath for r in samples]
        labels = [int(r.label) for r in samples]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        return ds.map(lambda p,y: (self.preprocess(p),tf.one_hot(y,4)))