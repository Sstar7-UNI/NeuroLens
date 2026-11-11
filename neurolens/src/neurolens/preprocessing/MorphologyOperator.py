import tensorflow as tf

from neurolens.preprocessing.ImageOperator import ImageOperator
from neurolens.preprocessing.Connectivity import Connectivity
from neurolens.preprocessing.RegionOfInterest import RegionOfInterest

class MorphologyOperator(ImageOperator):

    def __init__(self,connectivity: Connectivity) -> None:
        super().__init__()
        self._connectivity = connectivity

    def dilate(self, mask: tf.Tensor) -> tf.Tensor:
        if self._connectivity == Connectivity.FOUR:
            kernel = tf.constant([[0, 1, 0],[1, 1, 1],[0, 1, 0]],dtype=mask.dtype)
        else:
            kernel = tf.constant([[1, 1, 1],[1, 1, 1],[1, 1, 1]],dtype=mask.dtype)
        imageBatch = tf.expand_dims(mask, axis=0)
        filterTensor = tf.reshape(kernel, (3, 3, 1, 1))
        neighborCounts = tf.nn.conv2d(imageBatch,filterTensor,strides=[1, 1, 1, 1],padding="SAME")
        dilated = neighborCounts > 0
        dilated = tf.squeeze(dilated, axis=0)
        return tf.cast(dilated, mask.dtype)

    def erode(self, mask: tf.Tensor) -> tf.Tensor:
        if self._connectivity == Connectivity.FOUR:
            kernel = tf.constant([[0, 1, 0],[1, 1, 1],[0, 1, 0]],dtype=mask.dtype)
        else:
            kernel = tf.constant([[1, 1, 1],[1, 1, 1],[1, 1, 1]],dtype=mask.dtype)
        imageBatch = tf.expand_dims(mask, axis=0)
        filterTensor = tf.reshape(kernel, (3, 3, 1, 1))
        neighborCounts = tf.nn.conv2d(imageBatch,filterTensor,strides=[1, 1, 1, 1],padding="SAME")
        if self._connectivity == Connectivity.FOUR:
            eroded = neighborCounts == 5
        elif self._connectivity == Connectivity.EIGHT:
            eroded = neighborCounts == 9
        eroded = tf.squeeze(eroded, axis=0)
        return tf.cast(eroded, mask.dtype)

    def open(self, mask: tf.Tensor) -> tf.Tensor:
        erodedMask = self.erode(mask)
        newMask = self.dilate(erodedMask)
        return newMask

    def close(self, mask: tf.Tensor) -> tf.Tensor:
        dilatedMask = self.dilate(mask)
        newMask = self.erode(dilatedMask)
        return newMask

    def reconstructByDilation(self, marker: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        current=marker*mask
        changed=tf.constant(True)
        def condition(current,changed):
            return changed
        def body(current,changed):
            dilated=self.dilate(current)
            nextMask=dilated*mask
            hasChanged=tf.reduce_any(nextMask!=current)
            return nextMask,hasChanged
        result, _ = tf.while_loop(condition,body,loop_vars=(current, changed))
        return result

    def fillHoles(self, mask: tf.Tensor) -> tf.Tensor:
        background = 1 - mask
        height = tf.shape(mask)[0]
        width = tf.shape(mask)[1]
        rows = tf.range(height)[:, tf.newaxis]
        columns = tf.range(width)[tf.newaxis, :]
        border = ((rows == 0)| (rows == height - 1)| (columns == 0)| (columns == width - 1))
        marker = tf.cast(tf.expand_dims(border, axis=-1),mask.dtype) * background
        exterior = self.reconstructByDilation(marker, background)
        return 1 - exterior

    def apply(self, input: tf.Tensor) -> tf.Tensor:
        closedMask = self.close(input)
        filledMask = self.fillHoles(closedMask)
        return filledMask

    def connectedComponents(self, mask: tf.Tensor) -> list[RegionOfInterest]:
        regions = []
        remaining = tf.identity(mask)
        while bool(tf.reduce_any(remaining > 0).numpy()):
            positions = tf.where(remaining > 0)
            seed = positions[:1]
            marker = tf.scatter_nd(
                indices=seed,
                updates=tf.ones((1,), dtype=mask.dtype),
                shape=tf.shape(remaining, out_type=seed.dtype)
            )
            component = self.reconstructByDilation(marker, remaining)
            regions.append(RegionOfInterest(component))
            remaining = remaining * (1 - component)
        return regions