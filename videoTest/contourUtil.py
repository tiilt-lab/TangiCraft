from videoTest import shapeDetection
import cv2
import imutils


class Board:

    def __init__(self, img):
        self.side_length = 0

        self.set_side_length(img)

        self.centers = [[(9 + j * 18, 9 + i * 18) for j in range(300 // 18)] for i in range(225 // 18)]
        self.top = [[0 for j in range(300 // 18)] for i in range(225 // 18)]

        self.there = [[False for j in range(300 // 18)] for l in range(225 // 18)]

        self.sd = shapeDetection.ShapeDetector()

    def set_side_length(self, img):
        self.side_length = self.get_side_length(img)

    def get_contours(self, img):
        dsize = get_dimensions(img)

        # resize image
        resized = cv2.resize(img, dsize)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        return cnts

    def get_side_length(self, img):
        cnts = self.get_contours(img)

        for c in cnts:
            shape = self.sd.detect(c)

            if not (shape == 'rectangle' or shape == 'square'):
                continue

            x, y, w, h = cv2.boundingRect(c)

            return w

    def get_center(self, x, y):
        w, h = self.side_length

        # Update the board when new blocks come up
        for p in range(len(self.centers)):
            for q in range(len(self.centers[0])):
                # Determine where the block is based on which center it's closest to
                # Center is like a "drop point" -- see doc
                center = self.centers[p][q]
                x_diff = abs(center[0] - (x + (w // 2)))
                y_diff = abs(center[1] - (y + (h // 2)))

                # If within half of the width of the bounding box, then it is the closest point
                if x_diff <= 9 and y_diff <= 9:
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
        cnts = self.get_contours(img)
        for c in cnts:
            shape = self.sd.detect(c)

            if not (shape == 'rectangle' or shape == 'square'):
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Add single block to topology
            self.add_single(x, y, low_layer=True)

    # Does initial block checking at the first level (make sure the val is non-zero if at least a block is there)
    # Meant to cover the "sliding" that a user can do.
    # Also helps for low level error correction if caught early on. (Someone placed a block on the low level and didn't
    # build up and our initial code didn't detect it)
    # Also accounts for the clear behavior one might do where they just slide all the blocks out of the workspace
    def surface_level(self, img):
        self.there = [[False for j in range(300 // 18)] for l in range(225 // 18)]

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


def get_dimensions(img):
    # percent by which the image is resized
    scale_percent = 50

    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    return dsize