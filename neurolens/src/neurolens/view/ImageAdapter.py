import numpy as np
from PySide6 import QtGui
from PySide6.QtGui import QImage, QPixmap

class ImageAdapter:

    def tensorToArray(self, tensor):
        array = tensor.numpy()
        array = np.squeeze(array,axis=-1)#asta elimina ultima dimensiune, in cazul asta numarul de canale pe care il are tf
        return array

    def arrayToUint8(self, array):
        array = np.clip(array,0,1)#aduce valorile in interval
        new_array = np.uint8(array*255)
        return new_array

    def arrayToQImage(self, array):
        height, width = array.shape
        bytesPerLine = array.strides[0] # nr octeti dintre doua randuri consecutive
        image=QImage(array.data,width,height,bytesPerLine,QImage.Format.Format_Grayscale8).copy() # copy ca sa nu ramana pe memoria array
        return image

    def qImageToPixmap(self,image):
        pixmap = QPixmap.fromImage(image)
        return pixmap

    def tensorToPixmap(self, tensor):
        array = self.tensorToArray(tensor)
        uint8array = self.arrayToUint8(array)
        qimage = self.arrayToQImage(uint8array)
        pixmap=self.qImageToPixmap(qimage)
        return pixmap