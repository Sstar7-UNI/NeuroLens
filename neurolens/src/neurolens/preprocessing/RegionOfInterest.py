import tensorflow as tf

class RegionOfInterest:

    def __init__(self, mask: tf.Tensor) -> None:
        self._mask = mask

    def boundingBox(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        positions = tf.where(tf.cast(self._mask, tf.bool)) # avem h w ch deci o sa avem y x nu x y
        def emptyBox():
            zero = tf.constant(0, dtype=positions.dtype)
            return zero, zero, zero, zero
        def nonEmptyBox():
            ys = positions[:, 0]
            xs = positions[:, 1]
            xMin = tf.reduce_min(xs)
            xMax = tf.reduce_max(xs)
            yMin = tf.reduce_min(ys)
            yMax = tf.reduce_max(ys)
            width = xMax - xMin + 1
            height = yMax - yMin + 1
            return xMin, yMin, width, height
        return tf.cond(tf.shape(positions)[0] > 0,nonEmptyBox,emptyBox)

    def applyMask(self, image: tf.Tensor) -> tf.Tensor:
        castedMask = tf.cast(self._mask,image.dtype)
        return castedMask*image

    def cropToBoundingBox(self, image: tf.Tensor) -> tf.Tensor:
        x, y, width, height = self.boundingBox()
        return image[y:y + height, x:x + width, :]