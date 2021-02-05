# So far implemented own version of object tracking. Works alright for one hand.
# Only worked on gripping with thumb and index finger. Need to open up for other methods.
# Grip detection is not bad actually. There's probably a better way though.
# Still need to implement object detection in the case of sliding objects? (Unless theres a better way)
# Not quite sure if there is an easier way to do object tracking. Looking into the source code and maybe the nodes can
# help??? Not quite sure how the code in the back works. Don't even know where they're tensors are (maybe that's
# intentional?)
# Not sure if there's a better way of looking into things tbh. Very messy code.

import cv2
import mediapipe as mp
import math
from copy import deepcopy
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def eud_dist(a_x, a_y, b_x, b_y):
    dist = ((a_x - b_x) ** 2) + ((a_y - b_y) ** 2)
    return math.sqrt(dist)


def find_midpoint(a_x, a_y, b_x, b_y):
    return ((a_x + b_x) / 2, (a_y + b_y) / 2)


def is_thumb_near_finger(hl, finger, lb, ub):
    ind = finger * 4
    point1 = hl.landmark._values[4]
    point2 = hl.landmark._values[ind]
    distance = eud_dist(point1.x, point1.y, point2.x, point2.y)
    return lb < distance < ub


def get_dimensions(img):
    # percent by which the image is resized
    scale_percent = 50

    # calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    return dsize


class hand:
    def __init__(self, loc, handedness, grabbing=False):
        self.was_there = False
        self.last_loc = loc
        self.moving = False
        self.grabbing = grabbing
        self.handedness = handedness
        self.grace_period = 0

        # Arbitrary values, need to mess around with
        self.distance_params = [0.020, 0.25]
        self.grabbing_params = [0, 0.08]
        self.max_gp = 1

    def is_still(self, loc):
        distance = eud_dist(loc.x, loc.y, self.last_loc.x, self.last_loc.y)
        params = self.distance_params
        return distance < params[0]

    def is_moving(self, loc):
        distance = eud_dist(loc.x, loc.y, self.last_loc.x, self.last_loc.y)
        params = self.distance_params
        return params[0] < distance < params[1]

    # Return new location and index if there, else return None
    def find_loc(self, mh, mhl):
        for i in range(0, len(mh)):
            handedness = mh[i].classification._values[0].label
            loc = mhl[i].landmark._values[0]

            if self.handedness != handedness:
                return None, None
            if self.is_moving(loc):
                return loc, i
            elif self.is_still(loc):
                return loc, i
        return None, None

    # Return True if successful, else False
    def update_loc(self, mh, mhl):
        loc, ind = self.find_loc(mh, mhl)
        if loc is not None:
            self.last_loc = loc
            return True
        return False

    # Return True if grabbing, else False
    def is_grabbing(self, mh, mhl):
        if self.moving:
            return self.grabbing

        loc, ind = self.find_loc(mh, mhl)
        if loc is None:
            return False
        hand_landmarks = mhl[ind]

        thumbIsOpen = False
        firstFingerIsOpen = False
        secondFingerIsOpen = False
        thirdFingerIsOpen = False
        fourthFingerIsOpen = False

        pseudoFixKeyPoint = hand_landmarks.landmark._values[2].x
        if hand_landmarks.landmark._values[3].x < pseudoFixKeyPoint and hand_landmarks.landmark._values[
            4].x < pseudoFixKeyPoint:
            thumbIsOpen = True

        pseudoFixKeyPoint = hand_landmarks.landmark._values[6].y
        if hand_landmarks.landmark._values[7].y < pseudoFixKeyPoint and hand_landmarks.landmark._values[
            8].y < pseudoFixKeyPoint:
            firstFingerIsOpen = True

        pseudoFixKeyPoint = hand_landmarks.landmark._values[10].y
        if hand_landmarks.landmark._values[11].y < pseudoFixKeyPoint and hand_landmarks.landmark._values[
            12].y < pseudoFixKeyPoint:
            secondFingerIsOpen = True

        pseudoFixKeyPoint = hand_landmarks.landmark._values[14].y
        if hand_landmarks.landmark._values[15].y < pseudoFixKeyPoint and hand_landmarks.landmark._values[
            16].y < pseudoFixKeyPoint:
            thirdFingerIsOpen = True

        pseudoFixKeyPoint = hand_landmarks.landmark._values[18].y
        if hand_landmarks.landmark._values[19].y < pseudoFixKeyPoint and hand_landmarks.landmark._values[
            20].y < pseudoFixKeyPoint:
            fourthFingerIsOpen = True

        p = self.grabbing_params

        p2 = (not firstFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 2, p[0], p[1])
        p3 = (not secondFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 3, p[0], p[1])
        p4 = (not thirdFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 4, p[0], p[1])
        p5 = (not fourthFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 5, p[0], p[1])
        # pincher = p2 or p3 or p4 or p5

        # Focus on only index finger for now
        pincher = p2
        grab = not thumbIsOpen and pincher

        return grab

    def update_grabbing(self, mh, mhl):
        self.grabbing = self.is_grabbing(mh, mhl)

    # Check if toggled from grabbing to not, and vice versa. Find loc, convert it to pic coordinates, and print it,
    # with the change. Return coordinates.
    def print_toggle(self, mh, mhl, img):
        if self.grace_period <= self.max_gp:
            return None, None
        loc, ind = self.find_loc(mh, mhl)
        if loc is not None:
            thumb = mhl[ind].landmark._values[4]
            index = mhl[ind].landmark._values[8]
            mid = find_midpoint(thumb.x, thumb.y, index.x, index.y)
            x = mid[0]
            y = mid[1]
            grabbing = self.is_grabbing(mh, mhl)
            if grabbing and not self.grabbing:
                size = get_dimensions(img)
                x = x * size[0]
                y = y * size[1]
                print("Grabbed at ({}, {})".format(x, y))
                return x, y
            if self.grabbing and not grabbing:
                size = get_dimensions(img)
                x = x * size[0]
                y = y * size[1]
                print("Released at ({}, {})".format(x, y))
                return x, y
        return None, None

    # Return index. Check if there first, then update grabbing, then update loc.
    # It'll be on the main function to take index and remove available entry or remove hand.
    def update_everything(self, mh, mhl):
        loc, ind = self.find_loc(mh, mhl)
        if loc is None:
            return None

        self.moving = self.is_moving(loc)
        self.update_grabbing(mh, mhl)
        self.update_loc(mh, mhl)
        self.grace_period += 1
        return ind


# For webcam input:
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
cap = cv2.VideoCapture("IMG_4343.MOV")
loh = []
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lod = []
    cmh = deepcopy(results.multi_handedness)
    cmhl = deepcopy(results.multi_hand_landmarks)

    if cmh is not None:
        ind = 0
        while len(loh) > ind:
            temp_hand = loh[ind]
            x, y = temp_hand.print_toggle(cmh, cmhl, image)
            if x is not None:
                lod.append((x, y))
            rem = temp_hand.update_everything(cmh, cmhl)
            if rem is not None:
                cmh.pop(rem)
                cmhl.pop(rem)
                ind += 1
            else:
                loh.pop(ind)
        for index in range(0, len(cmh)):
            temp_hand = hand(cmhl[index].landmark._values[0], cmh[index].classification._values[0].label)
            loh.append(temp_hand)
    else:
        loh = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Test if it's moving
    # for val in loh:
    #     if val.moving:
    #         print("moving")

    dsize = get_dimensions(image)

    # resize image
    image = cv2.resize(image, dsize)
    for d in lod:
        x = int(d[0])
        y = int(d[1])
        image = cv2.circle(image, (x, y), 50, (255, 0, 0), 10)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()
