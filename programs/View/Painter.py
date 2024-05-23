import cv2

class Painter:
    def __init__(self):
        self.windows_list = []

    def initialize_window(self, window_name, event_handler=None):
        if window_name not in self.windows_list:
            self.windows_list.append(window_name)
        cv2.namedWindow(window_name)
        if event_handler is not None:
            cv2.setMouseCallback(window_name,
                                event_handler)

    def paint(self, window_name, img, duration):
        cv2.imshow(window_name, img)
        key = cv2.waitKey(duration)
        return key

    def erase(self, window_name):
        self.windows_list.remove(window_name)
        cv2.destroyWindow(window_name)

    def erase_all(self):
        self.windows_list = []
        cv2.destroyAllWindows()