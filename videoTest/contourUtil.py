from videoTest import shapeDetection
import cv2
import imutils
import numpy as np

###############
# DEFINITIONS #
###############

# drop point: preset IRL coordinates that determine where a block is placed. block's location is based on the closest
# drop point the block is located at

# index set: the indices that the drop point has in the 2D array. Maps to point that tells Minecraft where to
# build (pretend the building space in Minecraft is a grid)

#########################
# ADJUSTABLE PARAMETERS #
#########################

# self.side_length
# self.side_deviation_threshold
# self.border_ratio
# function get_contours – every param used in the functions


class Board:

    def __init__(self, img):
        self.side_length = 0

        # The side length of the contour can be up to (self.side_deviation_threshold * 100)% smaller than the side
        # length
        self.side_deviation_threshold = 0.8

        # Defines the border of the board (self.board / self.border_ratio)
        # Anything found at the border is assumed to be held and ready to be placed rather than actually part of the
        # structure
        self.border_ratio = 6

        self.sd = shapeDetection.ShapeDetector()

        self.set_side_length(img)

        if self.side_length is None:
            self.side_length = 36

        # TODO: Remove when standard height is set and workflow for finding side length occurs
        self.side_length = 36

        self.width, self.height = img.shape[0], img.shape[1]

        # Creates an array with all the possible "drop points" on the board
        # Whatever is placed on the board will map to the closest center, to standardize where the blocks will map to in
        # Minecraft, which are the index set for top

        self.centers = [[(self.side_length // 2 + j * self.side_length,
                          self.side_length // 2 + i * self.side_length)
                         for j in range(self.height // self.side_length + 1)] for i in range(self.width // self.side_length + 1)]

        # Creates an array that represents a topological graph.
        # Each point represents the height (in blocks) at that point
        self.top = [[0 for j in range(self.height // self.side_length + 1)] for i in range(self.width // self.side_length + 1)]

        # Array that checks if a block is there. If it is not, yet the topological array says there is, then the block
        # gets removed

        # Essentially a helpful heuristic in case the grab detector didn't pick up on the removed block
        self.there = [[False for j in range(self.height // self.side_length + 1)] for l in range(self.width // self.side_length + 1)]

    def set_side_length(self, img):
        self.side_length = self.get_side_length(img)

    def get_side_length(self, img):
        cnts = get_contours(img)
        ret = 0

        for c in cnts:
            shape = self.sd.detect(c)

            # Only look at contours that are rectangles or squares because the rest probably aren't blocks
            if not (shape == 'rectangle' or shape == 'square'):
                continue

            x, y, w, h = cv2.boundingRect(c)

            # The temporary strategy is to find the maximum width among all quadrilateral contours
            if w > ret:
                ret = w

        # Only return the value if it's non-zero. Otherwise return None
        if ret != 0:
            return ret

    # Convert top corner coordinate to center coordinate
    def tc_to_center(self, x, y, w, h):
        return (x + (w // 2)), (y + (h // 2))

    # Get the index set mapped to the closest "drop point" based on (x, y) coordinates
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

    # Check if a block doesn't exist at a index set
    def is_block_not_at_center(self, x, y):
        # Map coordinates to a index set
        p, q = self.get_center(x, y)

        # If block doesn't exist at the index set
        # 0 means no blocks at that point b/c height is zero
        return self.top[p][q] == 0

    # Remove single block at given coordinates
    def remove_single(self, x, y):
        # Map coordinates to a index set
        p, q = self.get_center(x, y)

        # Check if there even is a block at that index set
        if self.is_block_not_at_center(x, y):
            print("Error: Removing something that isn't there.")
        else:
            self.top[p][q] -= 1

    # Add a single block at given coordinates
    def add_single(self, x, y, low_layer=False):
        # Map coordinates to that index set
        p, q = self.get_center(x, y)

        # low_layer is a flag to see where the first layer of blocks is at (height of at least 1)
        # you don't really want to change anything if the flag is on because it's just trying to read the contours
        # the exception is that it serves as another layer of check in case grab detection doesn't detect something
        # placed at a point
        # "we know at least one thing is there" -- basically a heuristic
        if not low_layer:
            self.top[p][q] += 1
        elif self.top[p][q] == 0:
            self.top[p][q] += 1

        # it's trying to read the contours so that we can clear out any blocks that have been mistakenly placed or not
        # detected as removed when it actually was removed
        if low_layer:
            # There was a block there, so don't remove when we check later.
            self.there[p][q] = True

    # This is the adding blocks to the top part for the lowest layer if there aren't already blocks labeled there.
    # Also checking if blocks exist at a "drop point"
    def add_low_layer(self, img):
        # Get the contours to see where the blocks are located
        cnts = get_contours(img)
        for c in cnts:
            shape = self.sd.detect(c)

            if not (shape == 'rectangle' or shape == 'square'):
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Basically check if the contour you are looking at even remotely is close to the length of the side
            if w / self.side_length <= self.side_deviation_threshold:
                # Convert top corner coordinates to center
                x, y = self.tc_to_center(x, y, w, h)

                # Convert coordinates to index set
                p, q = self.get_center(x, y)

                # Check if index set is part of the border
                if p > len(self.centers) // self.border_ratio and q > len(self.centers) // self.border_ratio:
                    # Add single block to topology
                    self.add_single(x, y, low_layer=True)

    # Does initial block checking at the first level (make sure the val is non-zero if at least a block is there)
    # Meant to cover the "sliding" that a user can do.
    # Also helps for low level error correction if caught early on. (Someone placed a block on the low level and didn't
    # build up and our initial code didn't detect it)
    # Also accounts for the clear behavior one might do where they just slide all the blocks out of the workspace
    def surface_level(self, img):
        # Recreate the there array because we need to scan every time, which requires a clean slate
        self.there = [[False for j in range(self.height // self.side_length + 1)] for l in range(self.width // self.side_length + 1)]

        # Scan contours to get the surface (low layer implies surface because you need a low layer to begin building up
        # and making the surface)
        self.add_low_layer(img)

        # Determine cleared blocks based on there
        self.clear_blocks()

    # Clear any blocks that are in the topography but not determined to be there based on the there array built through
    # scanning the contours
    def clear_blocks(self):
        # Check if there was a block at a given index set. If so, there at that point should be true
        # and just leave it be bc it's already been processed accordingly in above
        # code. If not, then that means any block at that point has been completely removed.
        for p in range(len(self.there)):
            for q in range(len(self.there[0])):
                if not self.there[p][q]:
                    self.top[p][q] = 0

    # Function to use if user decides when to build
    def build_activated(self, log, img):
        # Checks the log file as to where and what operations needs to be done (grab block, release block)
        for x, y, release in log:
            if release:
                self.remove_single(x, y)
            else:
                self.add_single(x, y)

        # heuristic to check for faulty or missed operations
        self.surface_level(img)


def get_contours(img):
    # Process the image to be a sharp black and white contrast to find contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.bilateralFilter(gray, -10, 25, 10)

    # Get the edges via Canny edge detection
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)

    e_adj = np.absolute(edges)
    edges = np.uint8(e_adj)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength=1, maxLineGap=15)

    # Draw the lines onto the b/w image to sharpen the edges in the image
    if lines is not None:
        for line in lines:
            line = line[0]
            cv2.line(thresh, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 1)

    # Get the contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return cnts

