import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def onMouse(event, x, y, flags, param) -> None:
    if event == cv.EVENT_LBUTTONDOWN:
        # draw circle here (etc...)
        print('x = %d, y = %d' % (x, y))


base_image = cv.imread('./images/room.jpg')
base_image = cv.cvtColor(base_image, cv.COLOR_BGR2RGB)
print("im here")


cv.imshow('room', base_image)
cv.setMouseCallback('image', onMouse)
while(True):
    pressedkey = cv.waitKey(0)
    # Wait for ESC key to exit
    if pressedkey == 27:
        cv.destroyAllWindows()
        break
