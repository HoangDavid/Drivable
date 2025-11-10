import cv2
import numpy as np
from ultralytics import YOLO
from pynput.keyboard import Controller
import time

# ---------------------------
# Configurable parameters
# ---------------------------
STEER_ANGLE_THRESHOLD_DEG = 15  # degrees for left/right steering
THUMB_UP_ANGLE_THRESHOLD = 40   # degrees from vertical for thumb up/down
KEY_PRESS_REPEAT = 0.05          # seconds between key updates

# Key mapping
KEY_FORWARD = 'w'
KEY_BRAKE = 's'
KEY_LEFT = 'a'
KEY_RIGHT = 'd'

keyboard = Controller()
pressed_keys = set()

def press_key(key):
    if key not in pressed_keys:
        keyboard.press(key)
        pressed_keys.add(key)

def release_key(key):
    if key in pressed_keys:
        keyboard.release(key)
        pressed_keys.remove(key)

def release_all_keys():
    for k in list(pressed_keys):
        try:
            keyboard.release(k)
        except:
            pass
    pressed_keys.clear()

# ---------------------------
# Gesture detection helpers
# ---------------------------
def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = np.degrees(np.arctan2(-dy, dx))  # negative dy because y grows downward
    return ang

def detect_thumb_up(hand_kpts):
    """
    Rough check for thumb up/down based on wrist and thumb tip
    hand_kpts: numpy array of shape (N,2) for keypoints
    Returns: 'up', 'down', or None
    """
    if hand_kpts.shape[0] < 2:
        return None
    wrist = hand_kpts[0]
    thumb_tip = hand_kpts[1]  # thumb tip index may vary; adjust if needed
    v = thumb_tip - wrist
    angle = np.degrees(np.arctan2(-v[1], v[0]))
    angle_from_vertical = ((angle - 90 + 180) % 360) - 180
    if abs(angle_from_vertical) < THUMB_UP_ANGLE_THRESHOLD:
        return 'up'
    elif abs(abs(angle_from_vertical) - 180) < THUMB_UP_ANGLE_THRESHOLD:
        return 'down'
    return None

# ---------------------------
# Main function
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n-pose.pt")  # small pose model

    last_key_assert_time = 0

    print("Starting YOLOv8 hand control. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_h, image_w = frame.shape[:2]

            # Run YOLOv8 pose detection
            results = model(frame)[0]

            steering_angle = None
            thumb_action = None

            hand_centers = []

            if results.keypoints is not None:
                for kpts in results.keypoints:
                    # Convert Keypoints object to NumPy array (x,y)
                    kpts_array = kpts.xy  # shape (N,2)
                    if kpts_array.shape[0] < 2:
                        continue
                    wrist = kpts_array[0]
                    hand_centers.append(wrist)

                    # Detect thumb gesture
                    gesture = detect_thumb_up(kpts_array)
                    if gesture:
                        thumb_action = gesture
                        cv2.putText(frame, f"Thumb: {gesture}",
                                    (int(wrist[0]), int(wrist[1]-30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Steering: use first two hands
            if len(hand_centers) >= 2:
                p1 = np.array(hand_centers[0])
                p2 = np.array(hand_centers[1])
                steering_angle = angle_between_points(p1, p2)
                cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (0,255,0), 2)
                cv2.putText(frame, f"Steer angle: {steering_angle:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # Determine actions
            steer_left = steer_right = forward = brake = False
            if thumb_action == 'up':
                forward = True
            elif thumb_action == 'down':
                brake = True

            if steering_angle is not None:
                if steering_angle > STEER_ANGLE_THRESHOLD_DEG:
                    steer_right = True
                elif steering_angle < -STEER_ANGLE_THRESHOLD_DEG:
                    steer_left = True

            # Send keyboard events periodically
            current_time = time.time()
            if current_time - last_key_assert_time > KEY_PRESS_REPEAT:
                # Steering
                if steer_left:
                    press_key(KEY_LEFT)
                    release_key(KEY_RIGHT)
                elif steer_right:
                    press_key(KEY_RIGHT)
                    release_key(KEY_LEFT)
                else:
                    release_key(KEY_LEFT)
                    release_key(KEY_RIGHT)

                # Forward / brake
                if forward:
                    press_key(KEY_FORWARD)
                    release_key(KEY_BRAKE)
                elif brake:
                    press_key(KEY_BRAKE)
                    release_key(KEY_FORWARD)
                else:
                    release_key(KEY_FORWARD)
                    release_key(KEY_BRAKE)

                last_key_assert_time = current_time

            # Overlay current actions
            acts = []
            if forward: acts.append("FORWARD")
            if brake: acts.append("BRAKE")
            if steer_left: acts.append("LEFT")
            if steer_right: acts.append("RIGHT")
            if not acts: acts = ["IDLE"]
            status_text = "Action: " + ",".join(acts)
            cv2.putText(frame, status_text, (10, image_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)

            cv2.imshow("YOLOv8 Hand Drive", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        release_all_keys()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
