import cv2   
from ctypes import POINTER, c_char_p,CDLL,c_ubyte,c_int
import numpy as np
 
LibMask = CDLL("/home/pi/Mask/build/libMask.so")

# LibMask.VC.argtypes(POINTER(c_ubyte),c_int,c_int)

# LibMask.LoadModel.argtypes(c_char_p,c_char_p)
# ========================================================
# LibMask.LoadModel("/home/pi/Mask/Data/mask_detector.nb".encode("utf-8"),"/home/pi/Mask/Data/pyramidbox_lite.nb".encode('utf-8'))



# # UcharImg = Img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))


# LibMask.VC.argtypes = (POINTER(c_ubyte),c_int,c_int,)
# LibMask.VC.restype = (ctypes.c_char,)

# ret = LibMask.VC(UcharImg,rows,cols)

LibMask.LoadModel("/home/pi/Mask/Data/mask_detector.nb".encode("utf-8"),"/home/pi/Mask/Data/pyramidbox_lite.nb".encode('utf-8'))


Img = cv2.imread("/home/pi/Mask/Data/test_img.jpg")
cols = Img.shape[1]
rows = Img.shape[0]

UcharImg = Img.ctypes.data_as(POINTER(c_ubyte))

LibMask.VC.argtypes = (POINTER(c_ubyte),c_int,c_int,)
LibMask.VC.restype = c_char_p

print(LibMask.VC(UcharImg,rows,cols))