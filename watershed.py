import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread ("3.jpg")

def CalcOfDamageLeaf (image_name):
    image = cv.imread(image_name)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    image_erode = cv.erode(image, kernel)
    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)
    markers = np.zeros((image.shape[0], image.shape[1]), dtype="int32")
    markers[90:140, 90:140] = 225
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[236:255, 236:255] = 1
    area_BGR = cv.watershed(image_erode, markers)
    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    sick_part = area_BGR - healthy_part

    mask = np.zeros_like(image, np.uint8)
    mask[area_BGR > 1] = (255, 0, 255)
    mask[sick_part > 1] = (0, 0, 255)
    return mask

filt1 = CalcOfDamageLeaf("3.jpg")

bilateral = cv.bilateralFilter(img, 5, 60, 60)
cv.imwrite("3_bilateral.jpg", bilateral)

filt2 = CalcOfDamageLeaf("1_bilateral.jpg")

GaussianBlur = cv.GaussianBlur(img, (5,5), 0)
cv.imwrite("3_GausseanBlur.jpg", GaussianBlur)

filt3 = CalcOfDamageLeaf("3_GausseanBlur.jpg")

Median = cv.medianBlur(img, ksize = 5)
cv.imwrite("Median.jpg", Median)

filt4 = CalcOfDamageLeaf("Median.jpg")

plt.imshow(filt1)
plt.show()

plt.imshow(filt2)
plt.show()

plt.imshow(filt3)
plt.show()

plt.imshow(filt4)
plt.show()