import cv2
import copy
from operator import add
import numpy as np
from Model.Image import Image
import Controller.utils as utils
import time

class PoissonBlender:
    def __init__(self, src_w_msk, tar, offset, scale_ratio, mode = 'default'):
        scaled_height, scaled_width, channels = src_w_msk.get_shape()
        scaled_height = int(scaled_height * scale_ratio)
        scaled_width = int(scaled_width * scale_ratio)
        src_img = cv2.resize(src_w_msk.get_data(), (scaled_width, scaled_height))
        src_img_msk = cv2.resize(src_w_msk.get_mask(), (scaled_width, scaled_height))
        if channels == 1:
            # append channel dim if src image is grayscale image
            src_img = np.expand_dims(src_img, -1)
            src_img_msk = np.expand_dims(src_img_msk, -1)
        self._background = copy.deepcopy(tar)
        self._src = Image()
        self._src.set_data(src_img)
        self._src.set_mask(src_img_msk)
        self._img_w_msk = Image()
        self._offset = offset
        self.set_div_of_guidance_field(mode)

    def _get_div_of_guidance_field(self, src, tar, mode='default'):
        assert src.shape == tar.shape
        laplacian_kernal = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        x_diff_kernal = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        y_diff_kernal = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        x_second_diff_kernal = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
        y_second_diff_kernal = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        height, width, channels = src.shape
        guidance_field = cv2.filter2D(src.astype(np.float32), -1, kernel=laplacian_kernal)
        if mode=='default':
            # By default, div. of guidance field is set to be the second derivative of src
            pass
        elif mode=='mix':
            # Guidance field is defined as the maximum of absolute value of the first derivatives of src and tar
            src_diff_x = cv2.filter2D(src.astype(np.float32), -1, kernel=x_diff_kernal)
            src_diff_y = cv2.filter2D(src.astype(np.float32), -1, kernel=y_diff_kernal)
            tar_diff_x = cv2.filter2D(tar.astype(np.float32), -1, kernel=x_diff_kernal)
            tar_diff_y = cv2.filter2D(tar.astype(np.float32), -1, kernel=y_diff_kernal)

            if channels == 1:
                # append channel dim if image is grayscale
                src_diff_x = np.expand_dims(src_diff_x, -1)
                src_diff_y = np.expand_dims(src_diff_y, -1)
                tar_diff_x = np.expand_dims(tar_diff_x, -1)
                tar_diff_y = np.expand_dims(tar_diff_y, -1)

            # get the index where the absolute value of src diff. is less than the one of tar diff. on X direction
            x_where_src_less_than_tar = np.sum(np.absolute(src_diff_x), -1) < np.sum(np.absolute(tar_diff_x), -1)
            x_where_src_less_than_tar = np.expand_dims(x_where_src_less_than_tar, -1)
            x_where_src_less_than_tar = np.repeat(x_where_src_less_than_tar, channels, -1)

            # get the index where the absolute value of src diff. is less than the one of tar diff. on Y direction
            y_where_src_less_than_tar = np.sum(np.absolute(src_diff_y), -1) < np.sum(np.absolute(tar_diff_y), -1)
            y_where_src_less_than_tar = np.expand_dims(y_where_src_less_than_tar, -1)
            y_where_src_less_than_tar = np.repeat(y_where_src_less_than_tar, channels, -1)

            # replace the diff. value with the one of maximum value between src and tar
            src_diff_x[x_where_src_less_than_tar] = tar_diff_x[x_where_src_less_than_tar]
            src_diff_y[y_where_src_less_than_tar] = tar_diff_y[y_where_src_less_than_tar]

            # get the second derivative on both direction
            src_second_diff_x = cv2.filter2D(src_diff_x, -1, kernel=x_second_diff_kernal)
            src_second_diff_y = cv2.filter2D(src_diff_y, -1, kernel=y_second_diff_kernal)

            # sum them up to get the div. of the given vector field
            guidance_field = src_second_diff_x + src_second_diff_y
        elif mode == 'average':
            # The div. of guidance field is simply average over the second derivatives of src and tar
            tar_dual_prime = cv2.filter2D(tar.astype(np.float32), -1, kernel=laplacian_kernal)
            guidance_field = (guidance_field + tar_dual_prime) / 2

        if channels == 1:
            # append channel dim if image is grayscale
            guidance_field = np.expand_dims(guidance_field, -1)
        return guidance_field

    def set_div_of_guidance_field(self, mode='default'):
        (start_x, start_y), (end_x, end_y) = utils.get_boundary_point_on_tar_given_src_w_offset(self._src.get_data(),
                                                                                                self._background.get_data(), self._offset)

        mask = np.zeros(self._background.get_data().shape, np.uint8)
        mask = utils.implant_ROI_of_src_in_tar(self._src.get_mask(), mask, (start_x, start_y), (end_x, end_y))

        div = self._get_div_of_guidance_field(self._src.get_data(), self._background.get_data()[start_y:end_y, start_x:end_x], mode)
        background = utils.implant_ROI_of_src_in_tar(div, self._background.get_data().astype(np.float32), (start_x, start_y), (end_x, end_y),
                                                     self._src.get_mask())
        self._img_w_msk.set_data(background)
        self._img_w_msk.set_mask(mask)

    def blend(self, painter, window_name):
        print('Blending...')
        print('I: This might take a while depending on the number of pixels inside the mask\n')
        painter.initialize_window(window_name)
        results = copy.copy(self._img_w_msk.get_data())
        ROIs = copy.copy(self._img_w_msk.get_mask())
        img_height, img_width, channel = results.shape
        t_start = time.time()
        for channel_num in range(channel):
            # for each channel, built up A matrix and b vector
            result = results[:, :, channel_num]
            ROI = ROIs[:, :, channel_num]
            ROI_index = [tuple(i) for i in np.column_stack(np.where(ROI == 255))]
            A = np.zeros((len(ROI_index), len(ROI_index)), np.float32)
            b = np.array([result[i] for i in ROI_index], np.float32)
            t = time.time()
            for index, value in enumerate(ROI_index):
                # construct every row vector in A corresponding to each pixel in ROI
                row_vec = np.zeros((len(ROI_index)), np.float32)
                four_neighbors = utils.get_four_neighbors(value, (0, 0), (img_height, img_width))
                # row_vec[index] = 4 if the center pixel is not on the boundary of given image
                row_vec[index] = len(four_neighbors)
                for neighbor in four_neighbors:
                    if ROI[neighbor] != 255:
                        # the given neighbor point is out of mask region
                        b[index] += result[neighbor]
                    else:
                        # the given neighbor point is in mask region
                        row_vec[ROI_index.index(neighbor)] = -1
                A[index] = row_vec
            print('Generating A and b on channel ' + str(channel_num) + ' takes', str(time.time() - t), 'seconds')
            t = time.time()
            # solve Ax = b
            x = np.linalg.solve(A, b)
            print('Solving x on channel ' + str(channel_num) + ' takes', str(time.time() - t), 'seconds\n')
            # t = time.time()
            # x, _, _, _ = np.linalg.lstsq(A, b)
            # print('solve x itratively: ', str(time.time() - t))
            #x_0 = np.ones(len(ROI_index), np.float32) * np.mean(self._background.get_data()[:,:,channel][ROI == 255])
            #x = utils.gauss_seidel(A, b, 0.001, 200, x_0)
            # clipping every value that is out of the uint8 range
            x[x <= 0] = 0
            x[x >= 255] = 255
            for index, value in enumerate(ROI_index):
                # fit x into ROI region by index (1D -> 2D)
                result[value] = x[index]
        print('It takes ' + str(time.time() - t_start) + ' seconds in total to finish interpolation\n')
        results = results.astype(np.uint8)
        print('Press any key to save the result (as '+ window_name + '.jpg in working dir)\n')
        painter.paint(window_name, results, 0)
        cv2.imwrite(window_name + '.jpg', results)
        return results