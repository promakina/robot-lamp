import cv2
import mediapipe as mp
import numpy as np
import logging
import math

from config_loader import load_parameters
rp = load_parameters("robot_parameters.py")

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

tracked_hand_center = None
tracked_label = None


def detect_hand(frame):
    '''
    Received frame, detects landmards, returns offsets
    '''
    global tracked_hand_center, tracked_label

    # Define tags
    INDEX_FINGER = False
    TWO_FINGERS = False
    THREE_FINGERS = False
    SHAKA_DETECTED = False

    selected_index = None
    offset_x = 0
    offset_y = 0
    label = None
    orientation = None
    upside = None

    # Flip and convert the frame for MediaPipe.
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    current_hands = []
  
    if results.multi_hand_landmarks: # If hands are detected
        # Find center of the hand(s)
        for i, this_landmark in enumerate(results.multi_hand_landmarks):
            # Compute the bounding box center.
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in this_landmark.landmark]
            y_coords = [int(lm.y * h) for lm in this_landmark.landmark]
            center = ( (min(x_coords) + max(x_coords)) // 2, (min(y_coords) + max(y_coords)) // 2 )
            current_hands.append((center, i))
            # Draw landmarks.
            mp_draw.draw_landmarks(frame, this_landmark, mp_hands.HAND_CONNECTIONS)
        
        # Find labels
        labels = [is_right_or_left(handedness) for handedness in results.multi_handedness]

        selected_index, tracked_hand_center, tracked_label = (
            whichHandtoTrack(current_hands, labels, selected_index, tracked_hand_center, tracked_label) )

        hand_landmarks = results.multi_hand_landmarks[selected_index]
        landmarks = hand_landmarks.landmark
        #handedness = results.multi_handedness[selected_index]

        #label = is_right_or_left(handedness)
        upside = is_hand_upside(landmarks)
        

        offset_x, offset_y = process_hand_bounding_box(frame, hand_landmarks)
        
        offset_x = -offset_x # Adjust for flipped frame

        INDEX_FINGER = is_index_finger_extended(landmarks, frame.shape)
        TWO_FINGERS = is_two_fingers_extended(landmarks, frame.shape)
        THREE_FINGERS = is_three_fingers_sign(landmarks, frame.shape)
        
        if tracked_label:
            orientation = get_palm_orientation(landmarks, tracked_label)
            if orientation:
                SHAKA_DETECTED = is_shaka_gesture(landmarks, orientation, tracked_label, upside)

        # Display the current state for debugging.
        font_size = rp.FONT_SIZE
        font_thickness = rp.FONT_THICKNESS

        cv2.putText(frame, f"Index: {INDEX_FINGER}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_thickness)
        
        cv2.putText(frame, f"Two Fingers: {TWO_FINGERS}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_thickness )
        
        cv2.putText(frame, f"Three Fingers: {THREE_FINGERS}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_thickness )
        
        cv2.putText(frame, f"Shaka: {SHAKA_DETECTED}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_thickness )
    
    else:
        # Reset values
        tracked_label = None
        tracked_hand_center = None
        selected_index = None

    return frame, offset_x, offset_y, INDEX_FINGER, TWO_FINGERS, THREE_FINGERS, SHAKA_DETECTED 

def whichHandtoTrack(current_hands, labels, selected_index, tracked_hand_center, tracked_label):
    '''
    Determine which hand to track when multiple hands are detected
    
    '''
    threshold = 200 # in pixels

    if tracked_hand_center is not None:
        if tracked_label is not None and 'right' in labels and 'left' in labels:
            # If left and right are detected
            # continue tracking previous hand, match previous label
            selected_index = labels.index(tracked_label)
            tracked_hand_center = current_hands[selected_index][0]
        else:
            # Find the hand whose center is closest to the previously tracked center.
            distances = [math.hypot(c[0]-tracked_hand_center[0], c[1]-tracked_hand_center[1]) for c, _ in current_hands]
            min_index = np.argmin(distances)
            
            # Use a threshold to decide if the previously tracked hand is still present.
            if distances[min_index] < threshold:  # threshold in pixels, adjust as needed
                selected_index = current_hands[min_index][1]
                tracked_hand_center = current_hands[min_index][0]
            else:
                # Previously tracked hand is lost; switch to the closest hand.
                selected_index = current_hands[min_index][1]
                tracked_hand_center = current_hands[min_index][0]
    else:
        # No previous hand; choose the first detected hand.
        selected_index = current_hands[0][1]
        tracked_hand_center = current_hands[0][0]
        tracked_label = labels[0]
    
    return selected_index, tracked_hand_center, tracked_label

def process_hand_bounding_box(frame, hand_landmarks):
    h, w, _ = frame.shape
    
    # Convert normalized landmarks to pixel coordinates.
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

    # Determine the bounding box.
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Draw the bounding box.
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Calculate the center of the bounding box.
    box_center_x = (x_min + x_max) // 2
    box_center_y = (y_min + y_max) // 2

    # Draw the center point on the box.
    cv2.circle(frame, (box_center_x, box_center_y), 5, (0, 0, 255), -1)

    # Calculate the center of the image.
    image_center_x = w // 2
    image_center_y = h // 2

    # Calculate the offsets.
    offset_x = box_center_x - image_center_x
    offset_y = box_center_y - image_center_y

    # Optionally, draw the offset text on the frame.
    cv2.putText(frame, f"Offset: X={offset_x} Y={offset_y}", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Calculate bounding box size.
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    bbox_area = bbox_width * bbox_height

    return offset_x, offset_y

def is_right_or_left(handedness):
    '''
    Determine if Left or Right Hand
    Output False if score is low
    '''
    label = handedness.classification[0].label  # "Left" or "Right"
    score = handedness.classification[0].score
    if score < 0.8:
        return False
    return label

def is_shaka_gesture(landmarks, orientation, label, upside):
    """
    Detects a 'shaka' gesture (hang loose) where the thumb and pinky are extended,
    and the index, middle, and ring fingers are curled.
    
    This function uses normalized landmark coordinates from MediaPipe Hands.
    It assumes that for an extended finger, the tip is positioned higher (smaller y)
    than the corresponding MCP joint for index, middle, and ring fingers.
    
    For the thumb, because of its orientation, we use a simple heuristic:
    for a right hand (when the hand is roughly facing the camera), the thumb tip should
    lie to the left (smaller x) of the thumb MCP by a small margin.
    
    Adjust the thresholds as needed for your application.
    """
    # Check that the index, middle, and ring fingers are curled.
    # Check TIP below PIP
    index_curled = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > \
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_curled = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > \
                    landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_curled = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > \
                landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    
    # Check that the pinky is extended.
    pinky_extended = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < \
                     landmarks[mp_hands.HandLandmark.PINKY_MCP].y

    

    # Check that the thumb is extended.
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    # For a right hand with the palm roughly facing the camera,
    # an extended thumb will have its tip further to the left than its MCP.

    if orientation == 'forward':
        if label == 'Right':
            thumb_extended = thumb_tip.x < thumb_mcp.x - 0.02  # Threshold may be adjusted
        elif label == 'Left':
            thumb_extended = thumb_tip.x > thumb_mcp.x - 0.02
    elif orientation == 'backward':
        if label == 'Right':
            thumb_extended = thumb_tip.x > thumb_mcp.x - 0.02  # Threshold may be adjusted
        elif label == 'Left':
            thumb_extended = thumb_tip.x < thumb_mcp.x - 0.02
    
    if not upside:
        # If upside down, change boolean to opposite
        index_curled = not index_curled
        middle_curled = not middle_curled
        ring_curled = not ring_curled 
        pinky_extended = not pinky_extended
        thumb_extended = not thumb_extended

    shaka_detected = thumb_extended and pinky_extended and index_curled and middle_curled and ring_curled
    
    logging.debug(f'SHAKA_DEBUG | {shaka_detected}')
    logging.debug(f'SHAKA_DEBUG | Thumb_ext: {thumb_extended}, pinky_ext: {pinky_extended}, index_curles: {index_curled}, middle_curles: {middle_curled}, ring_curled: {ring_curled}')

    return shaka_detected

def get_palm_orientation(landmarks, label):
    """
    Compute the approximate orientation of the palm.
    Returns 'forward' if the palm is likely facing the camera,
    and 'backward' if it's facing away.
    
    This is done by computing the palm normal vector using the
    wrist, index finger MCP, and pinky MCP landmarks.
    """
    # Extract the 3D coordinates from the landmarks.
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x,
                      landmarks[mp_hands.HandLandmark.WRIST].y,
                      landmarks[mp_hands.HandLandmark.WRIST].z])
    
    index_mcp = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                          landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                          landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].z])
    
    pinky_mcp = np.array([landmarks[mp_hands.HandLandmark.PINKY_MCP].x,
                          landmarks[mp_hands.HandLandmark.PINKY_MCP].y,
                          landmarks[mp_hands.HandLandmark.PINKY_MCP].z])
    
    # Form two vectors on the palm.
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    
    # Compute the normal vector via cross product.
    palm_normal = np.cross(v1, v2)
    
    # In MediaPipe, the coordinate system is such that z increases away from the camera.
    # One heuristic is to assume that if the z component of the normal is positive,
    # the palm is facing toward the camera ("forward"); if negative, it's facing away.
    if label == 'Right':
        if palm_normal[2] > 0:
            return "forward"
        else:
            return "backward"
    elif label == 'Left':
        if palm_normal[2] < 0:
            return "forward"
        else:
            return "backward"
    
def is_hand_upside(landmarks):
    """
    Determines if the hand is upside down based on the relative positions
    of the wrist and the MCP joints of the index, middle, and ring fingers.
    
    Returns True if the hand is upside (wrist is below the average MCP),
    and False otherwise.
    """
    # Get normalized y-coordinates.
    wrist_y = landmarks[mp_hands.HandLandmark.WRIST].y
    index_mcp_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_mcp_y = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_mcp_y = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
    
    # Compute the average y-coordinate of these MCP joints.
    avg_mcp_y = (index_mcp_y + middle_mcp_y + ring_mcp_y) / 3
    
    # If the wrist is higher (i.e., has a lower y value) than the average MCP,
    # the hand is likely upside down.
    return wrist_y > avg_mcp_y

def circle_from_three_points(p1, p2, p3):
    """
    Compute the circle (center and radius) that passes through three points.
    Points are given as (x, y) tuples.
    """
    # Convert points to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Calculate the midpoints of p1p2 and p2p3
    mid1 = (p1 + p2) / 2.0
    mid2 = (p2 + p3) / 2.0
    
    # Calculate the slopes of p1p2 and p2p3
    delta1 = p2 - p1
    delta2 = p3 - p2
    
    # Check if the lines are vertical to avoid division by zero.
    if delta1[0] == 0:  # p1p2 vertical
        m1_inv = 0  # slope of perpendicular is 0 (horizontal)
    else:
        m1 = delta1[1] / delta1[0]
        m1_inv = -1 / m1 if m1 != 0 else 0
        
    if delta2[0] == 0:  # p2p3 vertical
        m2_inv = 0
    else:
        m2 = delta2[1] / delta2[0]
        m2_inv = -1 / m2 if m2 != 0 else 0

    # Now solve for intersection of the two perpendicular bisectors.
    # Equation of line through mid1 with slope m1_inv: y - mid1_y = m1_inv * (x - mid1_x)
    # Equation of line through mid2 with slope m2_inv: y - mid2_y = m2_inv * (x - mid2_x)
    # Solve for x:
    # m1_inv * (x - mid1_x) + mid1_y = m2_inv * (x - mid2_x) + mid2_y
    # (m1_inv - m2_inv) * x = m1_inv * mid1_x - m2_inv * mid2_x + mid2_y - mid1_y
    if m1_inv == m2_inv:
        # Points are collinear or nearly collinear; no unique circle
        return None, None
    x_center = (m1_inv * mid1[0] - m2_inv * mid2[0] + mid2[1] - mid1[1]) / (m1_inv - m2_inv)
    y_center = m1_inv * (x_center - mid1[0]) + mid1[1]
    center = (x_center, y_center)
    radius = np.linalg.norm(np.array(center) - p1)
    return center, radius

def is_index_finger_extended(landmarks, frame_shape, extension_factor=1.2, tolerance=5):
    """
    Determines if the index finger is extended, robust to hand orientation.
    
    The method:
      - Computes a circle from the wrist, index PIP, and pinky PIP.
      - Computes distances from the circle center to each fingertip.
      - Returns True if the index finger tip & DIP lies outside the circle by a factor (extension_factor),
        while the other finger tips are within or close to the circle.
    
    extension_factor: the index finger tip must be at least this multiple of the radius away from the center.
    tolerance: additional tolerance in pixels for the other fingertips.
    """
    h, w, _ = frame_shape

    # Convert normalized coordinates to pixel coordinates.
    def to_pixel(lm):
        return (int(lm.x * w), int(lm.y * h))
    
    wrist = to_pixel(landmarks[mp_hands.HandLandmark.WRIST])
    index_pip = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP])
    pinky_pip = to_pixel(landmarks[mp_hands.HandLandmark.PINKY_PIP])
    
    # Compute circle from wrist, index MCP, and pinky MCP.
    center, radius = circle_from_three_points(wrist, index_pip, pinky_pip)
    if center is None:
        return False  # cannot compute circle reliably

    # Compute distances from circle center to each fingertip.
    def distance(pt):
        return math.hypot(pt[0] - center[0], pt[1] - center[1])
    
    index_tip = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    index_dip = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP])
    middle_tip = to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    ring_tip = to_pixel(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP])
    pinky_tip = to_pixel(landmarks[mp_hands.HandLandmark.PINKY_TIP])
    thumb_tip = to_pixel(landmarks[mp_hands.HandLandmark.THUMB_TIP])
    
    index_dist = distance(index_tip)
    index_dist_dip = distance(index_dip)
    middle_dist = distance(middle_tip)
    ring_dist = distance(ring_tip)
    pinky_dist = distance(pinky_tip)
    thumb_dist = distance(thumb_tip)

    # For a folded finger, the tip should lie within the circle (allowing a small tolerance).
    others_folded = (middle_dist <= radius + tolerance and
                     ring_dist <= radius + tolerance and
                     pinky_dist <= radius + tolerance and
                     thumb_dist <= radius + tolerance)
    
    
    # For the index finger to be considered extended, its tip & dip should lie outside the circle
    # by a significant factor.
    index_extended =    (index_dist >= extension_factor * radius and
                        index_dist_dip >= (extension_factor/2) * radius)
    
    #print(f'Index extended: {index_extended}, others folded: {others_folded}')
    return index_extended and others_folded

def is_two_fingers_extended(landmarks, frame_shape, extension_factor=1.2, tolerance=5, angle_threshold=0.3):
    """
    Determines if a 'V' sign (index and middle fingers extended) is present.
    
    Parameters:
      - landmarks: list of MediaPipe hand landmarks.
      - frame_shape: tuple (height, width, channels) of the frame.
      - extension_factor: the index and middle fingertips must be at least this multiple
                          of the computed circle radius away from its center.
      - tolerance: allowable error (in pixels) for folded fingers.
      - angle_threshold: the minimum angle (in radians) between the index and middle finger
                         vectors (from the circle center) to confirm a V shape.
    
    Returns:
      True if the V sign is detected, False otherwise.
    """
    h, w, _ = frame_shape

    # Helper to convert normalized landmarks to pixel coordinates.
    def to_pixel(lm):
        return (int(lm.x * w), int(lm.y * h))
    
    # Define three points to create our reference circle:
    wrist = to_pixel(landmarks[mp_hands.HandLandmark.WRIST])
    index_mcp = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP])
    middle_pip = to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])
    pinky_mcp = to_pixel(landmarks[mp_hands.HandLandmark.PINKY_MCP])
    index_pip = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP])
    
    # CircleA wrist-index_mcp-pinky_mcp used to check if fingers are folded
    centerA, radiusA = circle_from_three_points(wrist, index_mcp, pinky_mcp)
    # CircleB wrist-index_pip-pinky_pip used to check if fingers are extended
    centerB, radiusB = circle_from_three_points(wrist, index_pip, middle_pip)
    if centerA is None or centerB is None:
        return False  # Could not compute a reliable circle.
    
    # Get fingertip positions.
    index_tip = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    middle_tip = to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    ring_tip = to_pixel(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP])
    pinky_tip = to_pixel(landmarks[mp_hands.HandLandmark.PINKY_TIP])
    thumb_tip = to_pixel(landmarks[mp_hands.HandLandmark.THUMB_TIP])

    def distance(pt, center):
        return math.hypot(pt[0] - center[0], pt[1] - center[1])
    
    index_dist = distance(index_tip, centerB)
    middle_dist = distance(middle_tip, centerB)
    ring_dist = distance(ring_tip, centerA)
    pinky_dist = distance(pinky_tip, centerA)
    thumb_dist = distance(thumb_tip, centerA)

    # Check extension conditions:
    # Index and middle fingertips should be significantly outside the circle.
    index_extended = index_dist >= extension_factor * radiusB
    middle_extended = middle_dist >= extension_factor * radiusB
    
    # The other fingertips (ring, pinky, thumb) should be folded; i.e., inside the circle (with some tolerance).
    others_folded = (ring_dist <= radiusA + tolerance and 
                     pinky_dist <= radiusA + tolerance and
                     thumb_dist <= radiusA + tolerance)
    
    try:
        # Optionally, compute the angle between the index and middle finger vectors (from the circle center).
        v_index = (index_tip[0] - centerA[0], index_tip[1] - centerA[1])
        v_middle = (middle_tip[0] - centerA[0], middle_tip[1] - centerA[1])
        
        # Compute the dot product and magnitudes.
        dot = v_index[0]*v_middle[0] + v_index[1]*v_middle[1]
        mag_index = math.hypot(v_index[0], v_index[1])
        mag_middle = math.hypot(v_middle[0], v_middle[1])
        
        if mag_index == 0 or mag_middle == 0:
            return False
        
        angle = math.acos(dot / (mag_index * mag_middle))
    except:
        return False
    
    # Check if the angle between the index and middle vectors is large enough to indicate a V sign.
    angle_ok = angle >= angle_threshold

    return index_extended and middle_extended and others_folded and angle_ok

def is_three_fingers_sign(landmarks, frame_shape, extension_factor=1.2, folded_tolerance=10):
    """
    Determines if the hand is showing a "3 sign" by having the middle, ring,
    and pinky fingers extended, while the index finger and thumb are folded.
    
    The algorithm:
      1. Computes a reference circle using three stable palm landmarks:
         the wrist, index finger PIP, and pinky PIP.
      2. For each of the extended fingers (middle, ring, pinky), it checks if the
         fingertip lies at least extension_factor * radius away from the circle center.
      3. For the folded fingers (thumb and index), it checks if their fingertips lie 
         within the circle (plus a small folded_tolerance in pixels).
    
    This approach is robust to rotation, so it works whether the hand is facing up,
    down, left, or right.
    
    Parameters:
      - landmarks: List of MediaPipe hand landmarks.
      - frame_shape: Tuple (height, width, channels) used for converting normalized
                     coordinates to pixel coordinates.
      - extension_factor: Factor by which the extended finger tips must exceed the circle radius.
      - folded_tolerance: Tolerance in pixels for folded fingers to be considered inside.
    
    Returns:
      True if the "3 sign" is detected, False otherwise.
    """
    h, w, _ = frame_shape

    def to_pixel(lm):
        return (int(lm.x * w), int(lm.y * h))
    
    # Compute reference circle from stable palm landmarks.
    wrist = to_pixel(landmarks[mp_hands.HandLandmark.WRIST])
    index_pip = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP])
    pinky_pip = to_pixel(landmarks[mp_hands.HandLandmark.PINKY_PIP])
    
    center, radius = circle_from_three_points(wrist, index_pip, pinky_pip)
    if center is None:
        return False  # Unable to compute a reliable circle.
    
    # Retrieve fingertip positions.
    middle_tip = to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    ring_tip   = to_pixel(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP])
    pinky_tip  = to_pixel(landmarks[mp_hands.HandLandmark.PINKY_TIP])
    index_tip  = to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    thumb_tip  = to_pixel(landmarks[mp_hands.HandLandmark.THUMB_TIP])
    
    def distance(pt):
        return math.hypot(pt[0] - center[0], pt[1] - center[1])
    
    # Compute distances for extended fingers.
    middle_dist = distance(middle_tip)
    ring_dist   = distance(ring_tip)
    pinky_dist  = distance(pinky_tip)
    
    extended_condition = (middle_dist >= extension_factor * radius and
                          ring_dist   >= extension_factor * radius and
                          pinky_dist  >= extension_factor * radius)
    
    # Compute distances for folded fingers.
    index_dist = distance(index_tip)
    thumb_dist = distance(thumb_tip)
    
    folded_condition = (index_dist <= radius + folded_tolerance and
                        thumb_dist <= radius + folded_tolerance)
    
    return extended_condition and folded_condition

def main():
    # Start capturing from the webcam.
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flippedFrame, offset_x, offset_y, a, b, c, d = detect_hand(frame)
        
        cv2.imshow("Gesture Sequence Detection", flippedFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
