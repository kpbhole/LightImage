import numpy as np
import cv2
import sys
from collections import namedtuple

#brange brightness range
#bval brightness value
BLevel = namedtuple("BLevel", ['brange', 'bval'])

#all possible levels
_blevels = [
    BLevel(brange=range(0, 24), bval=0),
    BLevel(brange=range(23, 47), bval=1),
    BLevel(brange=range(46, 70), bval=2),
    BLevel(brange=range(69, 93), bval=3),
    BLevel(brange=range(92, 116), bval=4),
    BLevel(brange=range(115, 140), bval=5),
    BLevel(brange=range(139, 163), bval=6),
    BLevel(brange=range(162, 186), bval=7),
    BLevel(brange=range(185, 209), bval=8),
    BLevel(brange=range(208, 232), bval=9),
    BLevel(brange=range(231, 256), bval=10),
]


def detect_level(h_val):
    h_val = int(h_val)
    for blevel in _blevels:
        if h_val in blevel.brange:
            return blevel.bval
    raise ValueError("Brightness Level Out of Range")


def get_img_avg_brightness(img):
    #img = cv2.imread("/Users/kalpeshbhole/Desktop/7.jpg",cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return int(np.average(v.flatten()))


img = cv2.imread("/Users/kalpeshbhole/Desktop/new.jpg",cv2.IMREAD_COLOR)
new_image = img
val = detect_level(get_img_avg_brightness(img))
print(val)
alpha = 1.0 # Simple contrast control
beta = 25    # Simple brightness control
if val < 9:
    while val < 9:
        try:
            if beta in [29,33,37] and beta <= 40 and alpha < 3.0:
                alpha += 0.65
            new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            #hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)[0][0]
            val = detect_level(get_img_avg_brightness(new_image))
            beta += 0.8
            print(val)
        except Exception as e:
            print(e)
else:
    while val > 9:
        beta -= 3
        new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        val = detect_level(get_img_avg_brightness(new_image))
        if val < 9:
            break
        print(val)

#cv2.imshow('Original Image', im)
try:
    cv2.imwrite('/Users/kalpeshbhole/Desktop/new.jpg', new_image)
    cv2.waitKey()
except Exception as e:
    cv2.imwrite('/Users/kalpeshbhole/Desktop/new.jpg', img)
    cv2.waitKey()
    print("No Change!")

print("The image brightness level is: {0}".format(val))
print("Done ")
