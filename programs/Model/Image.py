import cv2
import numpy as np
import copy
import os

class Image:
    def __init__(self, path = None, read_flag = cv2.IMREAD_UNCHANGED):
        self._data = None
        self._mask = None
        print(os.getcwd())
        if path is not None:
            self._data = cv2.imread(path, read_flag)
            self._mask = np.zeros(self._data.shape, dtype='uint8')
            if read_flag == cv2.IMREAD_GRAYSCALE:
                self._data = np.expand_dims(self._data, -1)
                self._mask = np.expand_dims(self._mask, -1)

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    #def get_masked_data(self):
    #    return cv2.bitwise_and(self._data, self._data, mask = self._mask)

    def get_mask(self):
        return self._mask

    def set_mask(self, mask):
        self._mask = mask

    def get_color_type(self):
        if self._data is not None:
            channels = self._data.shape[-1]
            if channels == 1:
                return 'grey'
            elif channels == 3:
                return 'rgb'
            elif channels == 4:
                return 'alpha'
            else:
                return None
        else:
            return None

    def get_shape(self):
        if self._data is not None:
            return self._data.shape
        else:
            return None