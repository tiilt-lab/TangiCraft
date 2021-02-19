from videoTest import shapeDetection
import cv2
import imutils
import numpy as np


class Board:

    def __init__(self, img):
        self.side_length = 0

        self.sd = shapeDetection.ShapeDetector()

        self.set_side_length(img)

        if self.side_length is None:
            self.side_length = 36

        # TODO: Remove when standard height is set and workflow for finding side length occurs
        self.side_length = 36

        self.width, self.height = img.shape[0], img.shape[1]

        self.centers = [[(self.side_length // 2 + j * self.side_length, # min(self.height, self.side_length // 2 + j * self.side_length),
                          self.side_length // 2 + i * self.side_length) # min(self.width, self.side_length // 2 + i * self.side_length))
                         for j in range(self.height // self.side_length + 1)] for i in range(self.width // self.side_length + 1)]
        self.top = [[0 for j in range(self.height // self.side_length + 1)] for i in range(self.width // self.side_length + 1)]

        self.there = [[False for j in range(self.height // self.side_length + 1)] for l in range(self.width // self.side_length + 1)]

    def set_side_length(self, img):
        self.side_length = self.get_side_length(img)

    def get_side_length(self, img):
        cnts = get_contours(img)

        for c in cnts:
            shape = self.sd.detect(c)

            if not (shape == 'rectangle' or shape == 'square'):
                continue

            x, y, w, h = cv2.boundingRect(c)

            return w

    def tc_to_center(self, x, y, w, h):
        return (x + (w // 2)), (y + (h // 2))

    def get_center(self, x, y):
        # Update the board when new blocks come up
        for p in range(len(self.centers)):
            for q in range(len(self.centers[0])):
                # Determine where the block is based on which center it's closest to
                # Center is like a "drop point" -- see doc
                center = self.centers[p][q]
                x_diff = abs(center[0] - x)
                y_diff = abs(center[1] - y)

                # If within half of the width of the bounding box, then it is the closest point
                if x_diff <= self.side_length // 2 and y_diff <= self.side_length // 2:
                    return p, q

    # Check if a block exists there
    def is_exist(self, x, y):
        p, q = self.get_center(x, y)
        return self.top[p][q] == 0

    def remove_single(self, x, y):
        p, q = self.get_center(x, y)

        if self.is_exist(x, y):
            print("Error: Removing something that isn't there.")
        else:
            self.top[p][q] -= 1

    def add_single(self, x, y, low_layer=False):
        p, q = self.get_center(x, y)

        # No block at that point case
        if not low_layer:
            self.top[p][q] += 1
        elif self.top[p][q] == 0:
            self.top[p][q] += 1

        if low_layer:
            # There was a block there, so don't remove when we check later.
            self.there[p][q] = True

    # This is the adding blocks to the top part for the lowest layer if there aren't already blocks labeled there.
    def add_low_layer(self, img):
        cnts = get_contours(img)
        for c in cnts:
            shape = self.sd.detect(c)

            if not (shape == 'rectangle' or shape == 'square'):
                continue

            # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            # # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # # show the output image
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)

            x, y, w, h = cv2.boundingRect(c)
            x, y = self.tc_to_center(x, y, w, h)

            # TODO: decide to replace below line with this chunk. Code works on test case.
            p, q = self.get_center(x, y)
            if p > 2 and q > 2:
                # Add single block to topology
                self.add_single(x, y, low_layer=True)

            # # Add single block to topology
            # self.add_single(x, y, low_layer=True)

    # Does initial block checking at the first level (make sure the val is non-zero if at least a block is there)
    # Meant to cover the "sliding" that a user can do.
    # Also helps for low level error correction if caught early on. (Someone placed a block on the low level and didn't
    # build up and our initial code didn't detect it)
    # Also accounts for the clear behavior one might do where they just slide all the blocks out of the workspace
    def surface_level(self, img):
        self.there = [[False for j in range(self.height // self.side_length + 1)] for l in range(self.width // self.side_length + 1)]

        self.add_low_layer(img)

        # Determine cleared blocks based on there
        self.clear_blocks()

    def clear_blocks(self):
        # Check if there was a block at a given drop point. If so, there at that point should be true
        # and just leave it be bc it's already been processed accordingly in above
        # code. If not, then that means any block at that point has been completely removed.
        for p in range(len(self.there)):
            for q in range(len(self.there[0])):
                if not self.there[p][q]:
                    self.top[p][q] = 0

    # Function to use if user decides when to build
    def build_activated(self, log, img):
        for x, y, release in log:
            if release:
                self.remove_single(x, y)
            else:
                self.add_single(x, y)

        self.surface_level(img)


def get_dimensions(img):
    # percent by which the image is resized
    scale_percent = 50

    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    return dsize


def get_contours(img):
    dsize = get_dimensions(img)

    # resize image
    resized = cv2.resize(img, dsize)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.bilateralFilter(gray, -10, 25, 10)

    edges = cv2.Canny(gray, 100, 200, apertureSize=3)

    e_adj = np.absolute(edges)
    edges = np.uint8(e_adj)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength=1, maxLineGap=15)
    for line in lines:
        line = line[0]
        cv2.line(thresh, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 1)

    thresh1 = cv2.subtract(thresh, edges)
    # for i in range(thresh.shape[0]):
    #     for j in range(thresh.shape[1]):
    #         thresh[i][j] = max(0, thresh[i][j] - edges[i][j])

    cv2.imshow("Image", thresh)
    cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts

