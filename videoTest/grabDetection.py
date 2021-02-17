# So far implemented own version of object tracking. Works alright for one hand.
# Only worked on gripping with thumb and index finger. Need to open up for other methods.
# Grip detection is not bad actually. There's probably a better way though.
# Still need to implement object detection in the case of sliding objects? (Unless theres a better way)
# Not quite sure if there is an easier way to do object tracking. Looking into the source code and maybe the nodes can
# help??? Not quite sure how the code in the back works. Don't even know where they're tensors are (maybe that's
# intentional?)
# Not sure if there's a better way of looking into things tbh. Very messy code.

# Handedness strat:
    # Very messy, so relying on percentage
    # Not remaking hand anymore because switches too frequently. Instead just changing hand.
    # If percentage is greater than some thresh, store it as most likely hand.
    # If the most likely hand is not the current hand, then use most likely hand
    # Update most likely hand if new percentage is greater than some thresh
        # The code can be oddly confident about the wrong things sometimes...

# Tracking works alright for both hands
    # Need to try overlapping hands
    # Need to try picking and moving
    # Need to try jerkiness + overlap
    # One of each and all three

# Track:
    # Hand angle
    # Velocity (px per frame) and determine param for difference
    # Fix jitter confirm check (do arr [t f] + t)
        # Have default be first instance
    # Finger ordering as another heuristic

import cv2
import mediapipe as mp
import math
from copy import deepcopy
from videoTest import contourUtil
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame = 0


def eud_dist(a_x, a_y, b_x, b_y):
    dist = ((a_x - b_x) ** 2) + ((a_y - b_y) ** 2)
    return math.sqrt(dist)


def find_midpoint(a_x, a_y, b_x, b_y):
    return ((a_x + b_x) / 2, (a_y + b_y) / 2)


def finger_to_finger_dist(hl, f1, f2):
    i1 = f1 * 4
    i2 = f2 * 4
    point1 = hl.landmark._values[i1]
    point2 = hl.landmark._values[i2]
    distance = eud_dist(point1.x, point1.y, point2.x, point2.y)
    return distance


def is_thumb_near_finger(hl, finger, lb, ub):
    return is_finger_near_finger(hl, 1, finger, lb, ub)


def is_finger_near_finger(hl, f1, f2, lb, ub):
    return lb < finger_to_finger_dist(hl, f1, f2) < ub


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
    def __init__(self, loc, handedness, percentage, grabbing=False):
        self.was_there = False
        self.last_loc = loc
        self.moving = False
        self.grabbing = grabbing
        self.handedness = handedness
        self.percentage = percentage
        self.grace_period = 0

        # Avoid instant fluctuations
        self.grab_log = []

        self.confirm_toggle = 3

        # Workaround for messed up hands
        if percentage >= 0.7:
            self.best_hand = handedness
        else:
            self.best_hand = None

        # Arbitrary values, need to mess around with
        self.distance_params = [0.020, 0.08]
        self.grabbing_params = [0, 0.08]

        # Only bc my hand did that weird thing was 15 added
        # self.max_gp = 15
        self.max_gp = 1

        # When grabbing changes, the value can't change for another 7 frames
        self.toggle_gp = -3

    def __str__(self):
        size = get_dimensions(image)
        return str(self.last_loc.x * size[0]) + " " + str(self.last_loc.y * size[1]) + " " + str(self.handedness) + " " + str(self.grabbing)

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
            percentage = mh[i].classification._values[0].score
            loc = mhl[i].landmark._values[0]

            # if self.handedness != handedness:
            #     return None, None
            if self.is_moving(loc) or self.is_still(loc):
                if percentage > 0.7:
                    self.best_hand = handedness
                self.handedness = handedness
                self.percentage = percentage
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

        pseudoFixKeyPoint = hand_landmarks.landmark._values[6].x
        if hand_landmarks.landmark._values[7].x < pseudoFixKeyPoint and hand_landmarks.landmark._values[
            8].x < pseudoFixKeyPoint:
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

        # p2 = (not firstFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 2, p[0], p[1])
        p2 = is_thumb_near_finger(hand_landmarks, 2, p[0], p[1])
        p3 = (not secondFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 3, p[0], p[1])
        p4 = (not thirdFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 4, p[0], p[1])
        p5 = (not fourthFingerIsOpen) and is_thumb_near_finger(hand_landmarks, 5, p[0], p[1])
        # pincher = p2 or p3 or p4 or p5

        # Workaround to messed up hand configs
        # alt_pincher = (not fourthFingerIsOpen) and is_finger_near_finger(hand_landmarks, 4, 5, p[0], p[1])
        alt_pincher = is_finger_near_finger(hand_landmarks, 4, 5, p[0], p[1])

        # Focus on only index finger for now
        pincher = p2
        if self.best_hand == self.handedness:
            grab = not thumbIsOpen and pincher
        else:
            grab = not thirdFingerIsOpen and alt_pincher

        return grab

    def update_grabbing(self, mh, mhl):
        new_grab = self.is_grabbing(mh, mhl)
        if self.grabbing != new_grab:
            self.gradient = 0
        self.grabbing = new_grab

    # Add change to log, if true -- for 5 frame delay
    # Update delay incrementer
    # Return the coordinates and print when delay is over, actually update toggle
    # Delete entry when delay is interrupted
    def update_toggle(self, mh, mhl, img):
        info = self.get_grabbing(mh, mhl, img)
        if info is None:
            return None, None

        if len(self.grab_log) > 2:
            print("Error in the grab log.")

        if len(self.grab_log) == 0:
            self.grab_log.append(info)
            self.grabbing = self.grab_log[0]["grabbing"]
        elif len(self.grab_log) == 1 and info["grabbing"] != self.grab_log[0]["grabbing"]:
            self.grab_log.append(info)
        elif len(self.grab_log) == 2 and info["grabbing"] != self.grab_log[1]["grabbing"]:
            self.grab_log.pop(1)

        x, y = None, None
        if len(self.grab_log) == 2 and self.grab_log[1]["change_delay"] >= self.confirm_toggle:
            self.grab_log.pop(0)
            x, y = self.print_toggle()
            self.grabbing = self.grab_log[0]["grabbing"]

        for grab_info in self.grab_log:
            grab_info["change_delay"] += 1

        return x, y

    # Get grabbing information for potential insertion into the toggle log
    def get_grabbing(self, mh, mhl, img):
        if self.grace_period <= self.max_gp:
            return None
        loc, ind = self.find_loc(mh, mhl)
        if loc is not None:
            thumb = mhl[ind].landmark._values[4]
            index = mhl[ind].landmark._values[8]
            mid = find_midpoint(thumb.x, thumb.y, index.x, index.y)
            x = mid[0]
            y = mid[1]
            grabbing = self.is_grabbing(mh, mhl)
            size = get_dimensions(img)
            x = x * size[0]
            y = y * size[1]
            return {"grabbing": grabbing, "coordinates": (x, y), "change_delay": 0}
        return None

    # If toggled. Find loc, convert it to pic coordinates, and print it, with the change. Return coordinates.
    # Introducing a five frame delay to confirm if it actually changed
    def print_toggle(self):
        if len(self.grab_log) > 1:
            print("Error in grabbing log. Shouldn't print if more than one element.")
            return None, None

        (x, y) = self.grab_log[0]["coordinates"]
        grabbing = self.grab_log[0]["grabbing"]

        if grabbing and not self.grabbing:
            print("Grabbed at ({}, {})".format(x, y))
            self.grace_period = self.toggle_gp
            return x, y
        if self.grabbing and not grabbing:
            print("Released at ({}, {})".format(x, y))
            self.grace_period = self.toggle_gp
            return x, y
        print("Error: It should toggle.")
        return None, None

    # Return index. Check if there first, then update grabbing, then update loc.
    # It'll be on the main function to take index and remove available entry or remove hand.
    def update_everything(self, mh, mhl):
        loc, ind = self.find_loc(mh, mhl)
        if loc is None:
            return None

        self.moving = self.is_moving(loc)
        # self.update_grabbing(mh, mhl)
        self.update_loc(mh, mhl)
        self.grace_period += 1
        # self.gradient += 1
        return ind


# For webcam input:
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
cap = cv2.VideoCapture("IMG_4362.MOV")
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

    # print(frame)
    # 137
    if frame == 48:
        stop = 0

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
            x, y = temp_hand.update_toggle(cmh, cmhl, image)
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
            temp_hand = hand(cmhl[index].landmark._values[0], cmh[index].classification._values[0].label, cmh[index].classification._values[0].score)
            loh.append(temp_hand)
    else:
        loh = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for h in range(0, len(loh)):
        # print(str(h) + ": " + str(loh[h]))
        continue

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

    frame += 1
hands.close()
cap.release()
