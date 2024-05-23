import cv2
import copy
import numpy as np
import Controller.utils as utils
MIN_PIXELS_OF_IMAGE = 10

class MaskMover:
    def __init__(self, src_w_msk, tar):
        self._src = copy.deepcopy(src_w_msk)
        self._tar = copy.deepcopy(tar)
        self._offset_x = 0
        self._offset_y = 0
        self._delta_x = 0
        self._delta_y = 0
        self._scale_ratio = 1
        self._is_mouse_left_down = False

        img_height, img_width, _ = src_w_msk.get_shape()
        bg_height, bg_width, _ = tar.get_shape()
        # calculate the maximum scaling ratio under the restriction that scaled src image cannot be bigger than tar image at any dimension
        self._scale_ratio_upperbound = min((bg_height / img_height, bg_width / img_width))
        if self._scale_ratio_upperbound < self._scale_ratio:
            # if the src image is bigger than target image, set the scale ratio to be less than 1.0 (default = maximun scaling ratio)
            self._scale_ratio = self._scale_ratio_upperbound
        # calculate the minimum scaling ratio under the restriction that scaled src image cannot be bigger than MIN_PIXELS_OF_IMAGE at any dimension
        self._scale_ratio_lowerbound = max((MIN_PIXELS_OF_IMAGE / img_height, MIN_PIXELS_OF_IMAGE / img_width))

    def _reset_all_members(self):
        self._offset_x = 0
        self._offset_y = 0
        self._delta_x = 0
        self._delta_y = 0
        self._scale_ratio = 1
        self._is_mouse_left_down = False

    def _blend(self, mode='msk'):
        scaled_height, scaled_width, channels = self._src.get_shape()
        scaled_height = int(scaled_height * self._scale_ratio)
        scaled_width = int(scaled_width * self._scale_ratio)
        img = cv2.resize(self._src.get_data(), (scaled_width, scaled_height))
        mask = cv2.resize(self._src.get_mask(), (scaled_width, scaled_height))
        if channels == 1:
            img = np.expand_dims(img, -1)
            mask = np.expand_dims(mask, -1)
        background = copy.copy(self._tar.get_data())
        (start_x, start_y), (end_x, end_y) = utils.get_boundary_point_on_tar_given_src_w_offset(img, background, (self._offset_x, self._offset_y))
        self._offset_x, self._offset_y = start_x, start_y
        if mode == 'img':
            # draw content of ROI in src on background
            background = utils.implant_ROI_of_src_in_tar(img, background, (start_x, start_y), (end_x, end_y), mask)
        elif mode == 'msk':
            # draw mask of ROI in src on background
            background = utils.implant_ROI_of_src_in_tar(mask, background, (start_x, start_y), (end_x, end_y), mask)
        return background

    def _event_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._is_mouse_left_down = True
            self._delta_x = x
            self._delta_y = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._is_mouse_left_down:
                self._offset_x += (x - self._delta_x)
                self._offset_y += (y - self._delta_y)
                self._delta_x = x
                self._delta_y = y
        elif event == 10:   #mouse scroll event
            if flags>0:     #scroll up
                self._scale_ratio += 0.05
                if self._scale_ratio > self._scale_ratio_upperbound:
                    self._scale_ratio = self._scale_ratio_upperbound
            else:           #scroll down
                self._scale_ratio -= 0.05
                if self._scale_ratio < self._scale_ratio_lowerbound:
                    self._scale_ratio = self._scale_ratio_lowerbound
        elif event == cv2.EVENT_LBUTTONUP:
            self._is_mouse_left_down = False
            self._delta_x = 0
            self._delta_y = 0

    def _key_handler(self, key):
        if key == ord('q') or key == ord('Q'):
            return 'quit'
        elif key == ord('s') or key == ord('S'):
            return 'save'
        elif key == ord('r') or key == ord('R'):
            return 'reset'
        else:
            return None

    def edit(self, painter, window_name):
        while True:
            painter.initialize_window(window_name, self._event_handler)
            key = painter.paint(window_name, self._blend('img'), 1)
            action = self._key_handler(key)
            if action == 'quit':
                exit(0)
            elif action == 'save':
                break
            elif action == 'reset':
                self._reset_all_members()
        print('Press any key to start blending...\n')
        painter.paint(window_name, self._blend('msk'), 0)
        painter.erase(window_name)
        return (self._offset_x, self._offset_y), self._scale_ratio