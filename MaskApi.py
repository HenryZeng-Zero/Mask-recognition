import cv2
from ctypes import POINTER, c_char_p, CDLL, c_ubyte, c_int
import json



class Mask_Api():
    def __init__(self, cdll_file = "./build/libMask.so"):
        # cdll_file = "/home/pi/Mask/build/libMask.so"
        self.LibMask = CDLL(cdll_file)

        self.LibMask.VC.argtypes = (POINTER(c_ubyte), c_int, c_int,)
        self.LibMask.VC.restype = c_char_p

    def Load(self, detect_model_file = './Data/pyramidbox_lite.nb', classify_model_file = './Data/mask_detector.nb'):
        self.detect_model_file = detect_model_file.encode('utf-8')
        self.classify_model_file = classify_model_file.encode('utf-8')

        self.LibMask.LoadModel(self.detect_model_file,self.classify_model_file)

    def Check(self,Img):
        UcharImg = Img.ctypes.data_as(POINTER(c_ubyte))

        Returns = self.LibMask.VC(UcharImg, rows, cols)

        Data = str(Returns,encoding="utf-8")[:-2]

        Jsons = json.loads(Data)

        print(Jsons)
