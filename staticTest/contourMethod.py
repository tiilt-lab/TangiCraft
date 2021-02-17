# Frame by frame change simulation + exploratory analysis of ripped code base to see if useful
# Source: https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

# import the necessary packages
import argparse
import imutils
import cv2
from pprint import pprint


# Ripped class from source. Probably don't need tbh.
class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


# This is the arrays used to analyze current structure, as described on the doc
centers = [[(9 + j * 18, 9 + i * 18) for j in range(300 // 18)] for i in range(225 // 18)]
top = [[0 for j in range(300 // 18)] for i in range(225 // 18)]
board = [[(0, 0) for j in range(300 // 18)] for i in range(225 // 18)]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

# For loop is used to simulate frame by frame changes
for i in range(5):
    # "there" array used to check if blocks have gotten removed.
    # 18 -- average width of block from abitrary height used when exploring
    # 225, 300 -- x and y dimensions used when looking at resized image
    there = [[False for j in range(300 // 18)] for l in range(225 // 18)]
    
    # Current image/frame looking at.
    args = "File_00" + str(i) + ".jpeg"

    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    image = cv2.imread(args)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Exploring edge detection and seeing if it could help
    sharp = cv2.Laplacian(blurred, cv2.CV_64F)

    # Change threshold (150) to account for different lighting. Can check
    # and confirm with the imshow below. Your target should be white while
    # everything else is black (ideally).
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image and initialize the
    # shape detector

    # Used to check if contours are really there
    #cv2.imshow("sharp", sharp)
    #cv2.imshow("blur", blurred)
    cv2.imshow("thres", thresh)
    cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    # loop over the contours
    for c in cnts:
        #if c.shape[0] > 20:
        #    continue
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        #cX = int((M["m10"] / M["m00"]) * ratio)
        #cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)

        # Only look for the blocks.
        if not (shape == 'rectangle' or shape == 'square'):
            continue

        x, y, w, h = cv2.boundingRect(c)
        print(x, y, w, h)


        # Update the board when new blocks come up
        for p in range(len(centers)):
            for q in range(len(centers[0])):
                # Determine where the block is based on which center it's closest to
                # Center is like a "drop point" -- see doc
                center = centers[p][q]
                x_diff = abs(center[0] - (x + (w // 2)))
                y_diff = abs(center[1] - (y + (h // 2)))

                # If within half of the width of the bounding box, then it is the closest point
                if x_diff <= 9 and y_diff <= 9:
                    # No block at that point case
                    if top[p][q] == 0:
                        top[p][q] += 1
                        board[p][q] = (w, h)
                    # Block is there, but width and height is greater (due to it being
                    # elevated -- blah blah depth perception on camera etc)
                    # Instead, come up with a threshold of what is different enough to be
                    # determined to be an added block on top. Need to talk about it
                    # Explain the theory in meeting to see if it makes sense.
                    elif board[p][q][0] < w and board[p][q][1] < h:
                        top[p][q] += 1
                        board[p][q] = (w, h)
                    # Conversely, width and height is smaller so block is removed.
                    elif board[p][q][0] > w and board[p][q][1] < h:
                        top[p][q] -= 1
                        board[p][q] = (w, h)
                    # There was a block there, so don't remove when we check later.
                    there[p][q] = True
                    break
            else:
                continue
            break

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        resized = imutils.resize(image, width=300)
        cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
        #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # show the output image
        cv2.imshow("Image", resized)
        cv2.waitKey(0)

    # Check if there was a block at a given drop point. If so, there at that point should be true
    # and just leave it be bc it's already been processed accordingly in above
    # code. If not, then that means any block at that point has been completely removed.
    for p in range(len(there)):
        for q in range(len(there[0])):
            if not there[p][q]:
                top[p][q] = 0

    pprint(board)
    pprint(top)
