from threading import Thread
from MaskApi import Mask_Api
import queue
import cv2
import time

Dll = Mask_Api()
Dll.Load()


class VideoCapture():

    def __init__(self, id):
        self.cap = cv2.VideoCapture(id)
        self.q = queue.Queue()
        t = Thread(target=self._Read_)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _Read_(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


cap = VideoCapture(0)


def PicFront(img):
    return cv2.flip(cv2.transpose(img), 0)


if __name__ == "__main__":
    time.sleep(0.1)
    while(1):
        jsondata = Dll.Check(PicFront(cap.read()))
        print(jsondata)
        
