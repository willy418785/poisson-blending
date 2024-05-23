import cv2
import copy

class Masker:
    def __init__(self, img):
        self._ori_img = img
        self._img = copy.deepcopy(img)
        self._is_mouse_left_down = False
        self._marker_radius = 3

    def _event_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # draw a circle with radius of self._marker_radius on mask and image when mouse left button pressed down
            self._is_mouse_left_down = True
            self._img.set_data(cv2.circle(self._img.get_data(), (x, y), self._marker_radius, (0, 255, 0), thickness=-1))
            self._img.set_mask(
                cv2.circle(self._img.get_mask(), (x, y), self._marker_radius, (255, 255, 255), thickness=-1))
        elif event == cv2.EVENT_MOUSEMOVE:
            # draw a circle with radius of self._marker_radius on mask and image when mouse moved
            if self._is_mouse_left_down:
                self._img.set_data(cv2.circle(self._img.get_data(), (x, y), self._marker_radius, (0, 255, 0), thickness=-1))
                self._img.set_mask(cv2.circle(self._img.get_mask(), (x, y), self._marker_radius, (255, 255, 255), thickness=-1))
        elif event == 10:
            # mouse scroll event
            if flags>0:     #scroll up
                # increase the radius of circle to be drawn when scrolling up
                self._marker_radius += 1
            else:           #scroll down
                self._marker_radius -= 1
                # decrease the radius of circle to be drawn when scrolling down
                if self._marker_radius < 1:
                    self._marker_radius = 1
        elif event == cv2.EVENT_LBUTTONUP:
            # stop drawing when mouse left button released
            self._is_mouse_left_down = False

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
            #opencv imshow loop
            painter.initialize_window(window_name, self._event_handler)
            key = painter.paint(window_name, self._img.get_data(), 1)
            action = self._key_handler(key)
            if action == 'quit':
                # quit drawing mask
                exit(0)
            elif action == 'save':
                # stop to finish drawing mask
                break
            elif action == 'reset':
                # reset mask
                self._img = copy.deepcopy(self._ori_img)
        print('Press any key to finish masking...\n')
        painter.paint(window_name, self._img.get_mask(), 0)
        painter.erase(window_name)
        self._img.set_data(self._ori_img.get_data())
        return self._img