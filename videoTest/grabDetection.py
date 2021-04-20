import cv2
import mediapipe as mp
import math
from copy import deepcopy
from videoTest import contourUtil
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#########################
# ADJUSTABLE PARAMETERS #
#########################

# self.grace_period_timer
# self.grace_period_timer_threshold
# self.grace_period_reset_toggle
# self.stability_timer
# self.stability_timer_threshold
# self.distance_params
# self.grabbing_params
# variable trigger – line 461

# Get euclidean distance of two points
def eud_dist(a_x, a_y, b_x, b_y):
    dist = ((a_x - b_x) ** 2) + ((a_y - b_y) ** 2)
    return math.sqrt(dist)


# Get midpoint of two points
def find_midpoint(a_x, a_y, b_x, b_y):
    return ((a_x + b_x) / 2, (a_y + b_y) / 2)


# Convert normalized hand coordiates to image coordinates
def hlist_to_coords(hlist, dsize):
    ret = []
    for hl in hlist:
        temp1 = []
        for val in hl.landmark._values:
            temp = [val.x * dsize[0], val.y * dsize[1]]
            temp1.append(temp)
        ret.append(temp1)
    return ret


# Get the distance between two fingertips
def finger_to_finger_dist(hl, f1, f2):
    i1 = f1 * 4
    i2 = f2 * 4
    point1 = hl[i1]
    point2 = hl[i2]
    distance = eud_dist(point1[0], point1[1], point2[0], point2[1])
    return distance


# Check if thumb is near finger based on a upper and lower bound
def is_thumb_near_finger(hl, finger, lb, ub):
    return is_finger_near_finger(hl, 1, finger, lb, ub)


# Check if finger is near finger based on a upper and lower bound
def is_finger_near_finger(hl, f1, f2, lb, ub):
    return lb < finger_to_finger_dist(hl, f1, f2) < ub


# Get the half dimensions of a image
def get_half_dimensions(img):
    # percent by which the image is resized
    scale_percent = 50

    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    return dsize


# Draw the grid lines
def drawlines(dim, img, b):
    offset = b.side_length

    # Green color in BGR
    color = (0, 255, 0)

    # Line thickness of 9 px
    thickness = 2

    x_start = 0
    y_start = 0
    while x_start <= dim[0]:
        img = cv2.line(img, (x_start, y_start), (x_start, dim[1]), color, thickness)
        x_start += offset

    x_start = 0
    while y_start <= dim[1]:
        img = cv2.line(img, (x_start, y_start), (dim[0], y_start), color, thickness)
        y_start += offset

    return img


# Get the measurement of the blocks
def prompt_measurement(cap, img):
    txt = 'Put Block Down For Measurement. Press "a" when complete.'
    img_txt = cv2.putText(img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands', img_txt)
    if cv2.waitKey(0) == ord('a'):
        pass

    _, img = cap.read()
    ds = get_half_dimensions(img)

    # resize image
    img = cv2.resize(img, ds)
    board_ret = contourUtil.Board(img)

    txt = 'Remove block. Press "a" when complete.'
    img_txt = cv2.putText(img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands', img_txt)
    if cv2.waitKey(0) == ord('a'):
        pass
    return board_ret


class hand:
    def __init__(self, mhl_val, handedness, curr_board, grabbing=False):
        # Last location of the hand (based on wrist)
        self.last_loc = mhl_val[0]

        # State of whether the hand is moving
        self.moving = False

        # State of whether the hand is grabbing something
        self.grabbing = grabbing

        # Whether it's the left or right hand
        self.handedness = handedness

        # Keep track of the time since a hand appeared
        # The code won't start detecting if the hand is grabbing anything until the grace period is past the
        # maximum grace period set (self.max_gp)
        self.grace_period_timer = 0

        self.grace_period_timer_threshold = 1

        # Avoid instant fluctuations
        # The heuristic is that we are assuming the hand can't actually re-drop or re-grab anything in 6 frames (0.1 of
        # a second), so we want to have the hand stabilize first before determining a drop or grab

        # Similar role to the grace period, but different functionality
        self.stability_timer = 0

        # The threshold where we can conclude that the hand is stablized
        self.stability_timer_threshold = 5

        # The lower and upper bound to what distance a hand travels still qualifies it being the same hand we were
        # already tracking
        # Arbitrary values, need to mess around with
        self.distance_params = [curr_board.side_length / 4, curr_board.side_length]

        # The lower and upper bound to what distance the fingers need to be at to qualify for the fingers to be close
        # enough (grabbing something)
        # Trying to do sqrt(2) more than side length in the case of holding it by the diagonal
        self.grabbing_params = [0, curr_board.side_length * 1.35]

    # String representation of a hand, for debugging purposes
    def __str__(self):
        return str(self.last_loc[0]) + " " + str(self.last_loc[1]) + " " + str(self.handedness) \
               + " " + str(self.grabbing)

    # Debug representation of a hand, for debugging purposes
    def __repr__(self):
        return str(self.last_loc[0]) + " " + str(self.last_loc[1]) + " " + str(self.handedness) \
               + " " + str(self.grabbing)

    # Check if the hand is not moving
    def is_still(self, loc):
        # Get the distance from the current location and the last location
        distance = eud_dist(loc[0], loc[1], self.last_loc[0], self.last_loc[1])
        params = self.distance_params
        # The distance should be less than the moving lower bounds to qualify as not moving
        return distance < params[0]

    # Check if the hand is moving
    def is_moving(self, loc):
        # Get the distance from the current location and the last location
        distance = eud_dist(loc[0], loc[1], self.last_loc[0], self.last_loc[1])
        params = self.distance_params
        # The distance should be within the moving bounds to qualify as moving
        return params[0] < distance < params[1]

    # Return new location and index if there, else return None
    def find_loc(self, mh, mhl):
        for i in range(0, len(mh)):
            # Get the information about the current hand
            handedness = mh[i].classification._values[0].label

            # Get the coordinates of the points of the hand
            mhl_val = mhl[i]

            # Get the wrist coordinates
            loc = mhl_val[0]

            # Check if the hand is moving. If not, then no need to update the information
            if self.is_moving(loc) or self.is_still(loc):
                # Update everything else and return the location of the hand and the index
                self.handedness = handedness
                return loc, i
        # Otherwise, if hand is not found, return None
        return None, None

    # Update the last location of the hand
    # Return True if successful, else False
    def update_loc(self, mh, mhl):
        # Get the location of the hand
        loc, ind = self.find_loc(mh, mhl)

        # Update it if the location was found
        if loc is not None:
            self.last_loc = loc
            return True
        return False

    # Check to see if the hand is grabbing something
    # Return True if grabbing, else False
    def is_grabbing(self, mh, mhl):
        # If it's moving, just return whatever the old value was. We are assuming we can't throw items and we can't
        # pick items up that quickly
        if self.moving:
            return self.grabbing

        # Get location and index of the hand
        loc, ind = self.find_loc(mh, mhl)

        # If hand is not found, then it's not moving
        if loc is None:
            return False

        # Get the coordinates of the points on the hand
        hand_landmarks = mhl[ind]

        # Check if the thumb is open by comparing the tip and the joint before the tip with the joint two before the tip
        pseudoFixKeyPoint = hand_landmarks[2][0]
        thumbIsOpen = hand_landmarks[3][0] < pseudoFixKeyPoint and hand_landmarks[4][0] < pseudoFixKeyPoint

        p = self.grabbing_params

        # Check if the fingers are near the thumb. If so, then the given finger is doing a pinching motion with the
        # thumb
        p2 = is_thumb_near_finger(hand_landmarks, 2, p[0], p[1])

        # Focus on only index finger for now
        pincher = p2
        grab = not thumbIsOpen and pincher

        return grab

    # If the grab state has changed, update it and reset the timer for stability delay
    def update_grabbing(self, mh, mhl):
        new_grab = self.is_grabbing(mh, mhl)
        if self.grabbing != new_grab:
            self.stability_timer = 0
        self.grabbing = new_grab

    # Check if toggled from grabbing to not, and vice versa. Find loc, convert it to pic coordinates, and print it,
    # with the change. Return coordinates.
    def print_toggle(self, mh, mhl, img):
        # If grace period timer less than threshold, still in grace period, so don't check
        if self.grace_period_timer <= self.grace_period_timer_threshold:
            return None, None, None

        # Get location and index of hand
        loc, ind = self.find_loc(mh, mhl)
        # If hand is not found, don't return coordinates
        if loc is not None:
            # Get location of potential drop
            thumb = mhl[ind][4]
            index = mhl[ind][8]
            mid = find_midpoint(thumb[0], thumb[1], index[0], index[1])
            x = mid[0]
            y = mid[1]

            # Check if the hand is stable and has toggled grab state
            grabbing = self.is_grabbing(mh, mhl)
            if grabbing and not self.grabbing and self.stability_timer >= self.stability_timer_threshold:
                print("Grabbed at ({}, {})".format(x, y))
                return x, y, True
            if self.grabbing and not grabbing and self.stability_timer >= self.stability_timer_threshold:
                print("Released at ({}, {})".format(x, y))
                return x, y, False
        return None, None, None

    # Return index. Check if there first, then update grabbing, then update loc.
    # It'll be on the main function to take index and remove available entry or remove hand.
    def update_everything(self, mh, mhl):
        loc, ind = self.find_loc(mh, mhl)
        if loc is None:
            return None

        self.moving = self.is_moving(loc)
        self.update_grabbing(mh, mhl)
        self.update_loc(mh, mhl)

        # Update the times as well
        self.grace_period_timer += 1
        self.stability_timer += 1
        return ind


def main():
    # For debugging purposes
    frame = 0

    # Board for Minecraft conversion
    board = None

    # Initialize hand detection
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

    # For webcam input:
    cap = cv2.VideoCapture("./testVideos/IMG_4362.MOV")

    # List of hands
    loh = []

    # Whether a frame has no hands in it
    no_hands = None

    # Trigger board cleanup when there are no hands for 10 consecutive frames
    trigger = 10

    # Log of all the actions taken
    log = []
    while cap.isOpened():
        # Read the image
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Resize image
        dsize = get_half_dimensions(image)
        image = cv2.resize(image, dsize)

        # Set up board if not set up already
        if board is None:
            # Replace with board prompt
            # board = contourUtil.Board(cv2.resize(image, dsize))
            board = prompt_measurement(cap, image)

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Find the hands in the images
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # List of coordinates where blocks have been grabbed or dropped
        lod = []

        # Get a copy of the hand info
        cmh = deepcopy(results.multi_handedness)

        # If hand info is none then there are no hands in the frame
        if cmh is not None:
            # Convert all hand point normalized coordinates to image coordinates.
            cmhl = hlist_to_coords(results.multi_hand_landmarks, dsize)
            ind = 0

            # Iterate through the hands
            while len(loh) > ind:
                # Get current hand
                temp_hand = loh[ind]

                # See if any grab or drop occurred
                x, y, release = temp_hand.print_toggle(cmh, cmhl, image)

                # If a grab or drop occurred, update the board
                if release is not None:
                    if release:
                        board.remove_single(x, y)
                    else:
                        board.add_single(x, y)

                    # Log used to keep track of what was dropped and picked up
                    log.append([x, y, release])

                # If there was a grab or drop, add it to the list of coordinates to mark the location
                if x is not None:
                    lod.append((x, y))

                # Update all the information about the hand from the last time to this frame
                rem = temp_hand.update_everything(cmh, cmhl)

                # If hand is successfully tracked and updated, remove from hand results for the frame
                # Otherwise remove from list of hands we are tracking
                if rem is not None:
                    cmh.pop(rem)
                    cmhl.pop(rem)
                    ind += 1
                else:
                    loh.pop(ind)

            # Add whatever remaining hands that didn't match any of the tracked hands to the tracked hands list
            # In other words, we begin tracking the "new" hands (might be mistakenly considered new)
            for index in range(0, len(cmh)):
                temp_hand = hand(cmhl[index], cmh[index].classification._values[0].label, board)
                loh.append(temp_hand)
            no_hands = 0
        else:
            # No hands are detected so remove all the currently tracked hands and update the no_hands state
            # No hands state only begins keeping track after hands initially appear
            loh = []
            if no_hands is not None:
                no_hands += 1

        # Draw the points on the hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check if the no_hands state has started keeping track and is triggered
        if no_hands is not None and no_hands > trigger:
            # If triggered, reset the no hands state and clean up the surface
            board.surface_level(image)
            no_hands = 0

        # Draw circles to mark where blocks were placed in this frame
        for d in lod:
            x = int(d[0])
            y = int(d[1])
            image = cv2.circle(image, (x, y), 50, (255, 0, 0), 10)

        # Draw grid lines to the image
        image = drawlines(dsize, image, board)

        # Draw marks on where blocks are located at all times
        for i in range(0, len(board.top)):
            for j in range(0, len(board.top[0])):
                if board.top[i][j] != 0:
                    cx, cy = board.centers[i][j]
                    image = cv2.circle(image, (cx, cy), 30, (0, 255, 0), 10)

        # Display the new augmented frame.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        frame += 1

    # Close the hand and video code after video or stream is over
    hands.close()
    cap.release()


if __name__ == "__main__":
    main()
