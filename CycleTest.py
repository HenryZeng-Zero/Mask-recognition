from threading import Thread
from MaskApi import Mask_Api
from sound import Play
import queue
import cv2
import time
import os

Dll = Mask_Api()
Dll.Load()

player = Play(5, "zh")


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


def Check(facedata):
    count = 0
    for face in facedata["Data"]:
        if face["width"] * face["height"] < 10000:
            return None

        if face["prob"] < 0.8:
            count += 1

    if facedata["Num"] == 0:
        return None

    return count


def CheckList(n):

    count = 0
    for i in range(0, n):
        facedata = Dll.Check(PicFront(cap.read()))
        dat = Check(facedata)
        if dat == 0:
            count += 1
        elif dat == None:
            return None

    return count / n


if __name__ == "__main__":
    time.sleep(0.1)
    out = CheckList(20)

    if out == None:
        key = False
    else:
        key = out > 0.8

    while(True):

        out = CheckList(20)
        print(out, time.time())

        if out == None:
            continue

        if out <= 0.9 and key == False:
            # player.Play("您没有戴口罩，请戴好口罩!")
            os.system('mplayer /home/pi/Mask/nomask.mp3')
            key = True

        elif out > 0.9 and key == True:
            # player.Play("您已戴好口罩，欢迎进入!")
            os.system('mplayer /home/pi/Mask/mask.mp3')
            key = False