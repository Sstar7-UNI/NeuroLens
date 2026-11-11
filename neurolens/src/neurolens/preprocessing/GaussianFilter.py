from neurolens.preprocessing.ImageOperator import ImageOperator
from neurolens.preprocessing.PaddingMode import PaddingMode
import tensorflow as tf
import math

class GaussianFilter(ImageOperator):

    def __init__(self, sigma: float, kernelSize: int, padding: PaddingMode) -> None:
        if sigma<=0 or not math.isfinite(sigma):
            raise ValueError("sigma must be positive and finite")
        if not kernelSize>0 or kernelSize%2==0:
            raise ValueError("kernelSize must be an odd and positive number")
        if not isinstance(padding,PaddingMode):
            raise ValueError("padding must a PaddingMode")
        self._sigma = sigma
        self._kernelSize = kernelSize
        self._padding = padding

    def buildKernel1D(self) -> list[float]:
        radius = self._kernelSize//2
        wx = []
        result = []
        for i in range(-radius,radius+1):
            aux=math.exp(-(i**2)/(self._sigma**2*2))
            wx.append(aux)
        suma=sum(wx)
        for x in wx:
            x=x/suma
            result.append(x)
        return result

    def apply(self, input: tf.Tensor) -> tf.Tensor:
        g=tf.constant(self.buildKernel1D(),dtype=tf.float32)
        column=tf.reshape(g,(self._kernelSize,1))
        row=tf.reshape(g,(1,self._kernelSize))
        kernel = column * row
        radius = self._kernelSize//2
        paddings = [ [radius, radius],[radius, radius],[0, 0]]
        if self._padding == PaddingMode.ZERO:
            padded = tf.pad(input, paddings, constant_values=0.0)
        elif self._padding == PaddingMode.REFLECT:
            padded = tf.pad(input, paddings, mode="REFLECT")
        elif self._padding == PaddingMode.REPLICATE:
            top = tf.repeat(input[:1, :, :], repeats=radius, axis=0)
            bottom = tf.repeat(input[-1:, :, :], repeats=radius, axis=0)
            padded = tf.concat([top, input, bottom], axis=0)
            left = tf.repeat(padded[:, :1, :], repeats=radius, axis=1)
            right = tf.repeat(padded[:, -1:, :], repeats=radius, axis=1)
            padded = tf.concat([left, padded, right], axis=1)
        image_batch = tf.expand_dims(padded, axis=0)
        filter_tensor = tf.reshape(kernel,(self._kernelSize, self._kernelSize, 1, 1))
        filtered = tf.nn.conv2d(image_batch,filter_tensor,strides=[1, 1, 1, 1],padding="VALID")
        return tf.squeeze(filtered, axis=0)