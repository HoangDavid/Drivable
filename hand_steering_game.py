"""
Hand Steering Controller for Roblox
A Python application that uses MediaPipe Hands to control Roblox games via keyboard input.
Steering is controlled by hand height comparison, and braking is triggered by open palm gesture.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from typing import Tuple, Optional, List
from pynput.keyboard import Controller, Key


# ============================================================================
# HAND DETECTOR MODULE
# ============================================================================

class HandDetector:
    """
    Handles MediaPipe Hands detection, line drawing, angle calculation,
    and gesture recognition (thumbs up for braking).
    """
    
    def __init__(self):
        """Initialize MediaPipe Hands detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.25
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_hands(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect hands in the image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (list of hand landmarks, list of hand side info)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else []
        hand_sides = results.multi_handedness if results.multi_handedness else []
        return landmarks, hand_sides
    
    def is_fist_closed(self, landmarks) -> bool:
        """
        Check if a hand is making a closed fist (all fingers except thumb are closed).
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            True if fist is closed, False otherwise
        """
        # MediaPipe landmark indices
        INDEX_TIP = 8
        INDEX_PIP = 6
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        RING_TIP = 16
        RING_PIP = 14
        PINKY_TIP = 20
        PINKY_PIP = 18
        
        try:
            # Check if all fingers (except thumb) are closed
            # Fingers are closed when tip is below PIP joint
            index_closed = landmarks.landmark[INDEX_TIP].y > landmarks.landmark[INDEX_PIP].y
            middle_closed = landmarks.landmark[MIDDLE_TIP].y > landmarks.landmark[MIDDLE_PIP].y
            ring_closed = landmarks.landmark[RING_TIP].y > landmarks.landmark[RING_PIP].y
            pinky_closed = landmarks.landmark[PINKY_TIP].y > landmarks.landmark[PINKY_PIP].y
            
            # All fingers must be closed for a fist
            return index_closed and middle_closed and ring_closed and pinky_closed
        except:
            return False
    
    def get_hand_center(self, landmarks, image_width: int, image_height: int) -> Optional[Tuple[int, int]]:
        """
        Get the center point of any hand for steering.
        Returns position for any detected hand (not just specific gestures).
        
        Args:
            landmarks: MediaPipe hand landmarks
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            (x, y) coordinates of hand center, or None if not detected
        """
        # MediaPipe landmark indices
        WRIST = 0
        INDEX_MCP = 5  # Index finger knuckle
        
        try:
            # Use index finger knuckle (MCP) as hand center for any hand
            index_mcp = landmarks.landmark[INDEX_MCP]
            index_x = int(index_mcp.x * image_width)
            index_y = int(index_mcp.y * image_height)
            
            return (index_x, index_y)
        except:
            return None
    
    def is_thumbs_up(self, landmarks, hand_side: str) -> bool:
        """
        Detect thumbs up gesture (thumb extended, other fingers closed in fist).
        
        Args:
            landmarks: MediaPipe hand landmarks
            hand_side: 'Left' or 'Right' (from MediaPipe's perspective)
            
        Returns:
            True if thumbs up detected, False otherwise
        """
        # MediaPipe landmark indices
        THUMB_TIP = 4
        THUMB_MCP = 2
        
        try:
            thumb_tip = landmarks.landmark[THUMB_TIP]
            thumb_mcp = landmarks.landmark[THUMB_MCP]
            
            # Thumb should be extended upward (thumb tip above thumb MCP)
            thumb_extended = thumb_tip.y < thumb_mcp.y
            
            # Other fingers should be closed (fist)
            fingers_closed = self.is_fist_closed(landmarks)
            
            # Thumbs up = thumb extended + other fingers closed
            return thumb_extended and fingers_closed
        except:
            return False
    
    def is_open_palm(self, landmarks) -> bool:
        """
        Detect open palm gesture (all fingers extended).
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            True if open palm detected, False otherwise
        """
        # MediaPipe landmark indices
        THUMB_TIP = 4
        THUMB_IP = 3
        INDEX_TIP = 8
        INDEX_PIP = 6
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        RING_TIP = 16
        RING_PIP = 14
        PINKY_TIP = 20
        PINKY_PIP = 18
        
        try:
            # Check if all fingers are extended
            # Fingers are extended when tip is above PIP joint
            thumb_extended = landmarks.landmark[THUMB_TIP].x > landmarks.landmark[THUMB_IP].x  # Thumb uses x-axis
            index_extended = landmarks.landmark[INDEX_TIP].y < landmarks.landmark[INDEX_PIP].y
            middle_extended = landmarks.landmark[MIDDLE_TIP].y < landmarks.landmark[MIDDLE_PIP].y
            ring_extended = landmarks.landmark[RING_TIP].y < landmarks.landmark[RING_PIP].y
            pinky_extended = landmarks.landmark[PINKY_TIP].y < landmarks.landmark[PINKY_PIP].y
            
            # All fingers must be extended for open palm
            return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended
        except:
            return False
    
    def calculate_line_angle(self, point1: Tuple[int, int], 
                            point2: Tuple[int, int]) -> float:
        """
        Calculate the angle of a line connecting two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Angle in degrees (-90 to 90, where 0 is horizontal)
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        # Calculate angle in radians, then convert to degrees
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to -90 to 90 range for steering
        # Horizontal line (0 degrees) = straight
        # Negative angle = tilt left (steer left)
        # Positive angle = tilt right (steer right)
        return angle_deg
    
    def draw_detection(self, image: np.ndarray, hand_landmarks_list: List,
                      hand_centers: List[Tuple[int, int]], 
                      angle: Optional[float], is_braking: bool,
                      gesture_classifications: List[str] = None,
                      is_detecting: bool = False) -> np.ndarray:
        """
        Draw hand landmarks, connecting line, angle, brake indicator, and gesture classifications.
        
        Args:
            image: Input image to draw on
            hand_landmarks_list: List of detected hand landmarks
            hand_centers: List of hand center points (for steering)
            angle: Steering angle in degrees
            is_braking: Whether brake is active
            gesture_classifications: List of gesture labels for each hand
            
        Returns:
            Image with overlays drawn
        """
        # Draw hand landmarks
        for i, hand_landmarks in enumerate(hand_landmarks_list):
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw gesture classification text near each hand
            if gesture_classifications and i < len(gesture_classifications):
                gesture_text = gesture_classifications[i]
                # Get wrist position for text placement
                wrist = hand_landmarks.landmark[0]
                h, w = image.shape[:2]
                text_x = int(wrist.x * w) - 50
                text_y = int(wrist.y * h) - 30
                
                # Choose color based on gesture
                if "BRAKE" in gesture_text:
                    color = (0, 0, 255)  # Red (brake)
                elif "STRAIGHT" in gesture_text:
                    color = (255, 255, 0)  # Yellow (straight/neutral)
                else:
                    color = (0, 255, 0)  # Green (driving - left/right)
                
                cv2.putText(image, gesture_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw line connecting hands (for steering)
        if len(hand_centers) == 2:
            cv2.line(image, hand_centers[0], hand_centers[1], (0, 255, 0), 3)
            # Draw circles at hand centers
            cv2.circle(image, hand_centers[0], 10, (255, 0, 0), -1)
            cv2.circle(image, hand_centers[1], 10, (255, 0, 0), -1)
        
        # Draw status information in top-left corner
        y_offset = 30
        line_height = 30
        
        # Draw angle text
        if angle is not None:
            angle_text = f"Steering Angle: {angle:.1f}°"
            cv2.putText(image, angle_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += line_height
        else:
            cv2.putText(image, "Steering: Need 2 hands detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += line_height
        
        # Draw brake indicator
        if is_braking:
            cv2.putText(image, "BRAKE: ACTIVE", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += line_height
        else:
            cv2.putText(image, "BRAKE: Inactive", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            y_offset += line_height
        
        # Draw hand count
        hand_count_text = f"Hands detected: {len(hand_landmarks_list)}"
        cv2.putText(image, hand_count_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # Draw detection status
        if is_detecting:
            status_text = "STATUS: STARTED (Press 'x' to STOP)"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "STATUS: STOPPED (Press 's' to START)"
            status_color = (0, 0, 255)  # Red
        
        cv2.putText(image, status_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y_offset += line_height
        
        # Draw instructions
        cv2.putText(image, "Controls: 's'=START, 'x'=STOP, 'q'=QUIT", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image


# ============================================================================
# KEYBOARD CONTROLLER MODULE
# ============================================================================

class KeyboardController:
    """
    Handles keyboard input simulation for Roblox games.
    Maps hand gestures to WASD keys.
    """
    
    def __init__(self):
        """Initialize keyboard controller."""
        self.keyboard = Controller()
        self.current_steering = None  # Track current steering direction
        self.is_brake_pressed = False  # Track brake key state
        self.is_accelerating = False  # Track acceleration (W key) state
        
    def press_key(self, key: str):
        """
        Press and hold a key.
        
        Args:
            key: Key to press (e.g., 'a', 'd', 's')
        """
        try:
            self.keyboard.press(key)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
    
    def release_key(self, key: str):
        """
        Release a key.
        
        Args:
            key: Key to release (e.g., 'a', 'd', 's')
        """
        try:
            self.keyboard.release(key)
        except Exception as e:
            print(f"Error releasing key {key}: {e}")
    
    def release_all(self):
        """Release all steering, brake, and acceleration keys."""
        if self.current_steering == "LEFT":
            self.release_key('a')
        elif self.current_steering == "RIGHT":
            self.release_key('d')
        if self.is_brake_pressed:
            self.release_key('s')
        if self.is_accelerating:
            self.release_key('w')
        
        self.current_steering = None
        self.is_brake_pressed = False
        self.is_accelerating = False
    
    def update_steering(self, direction: Optional[str]):
        """
        Update steering based on direction.
        
        Args:
            direction: "LEFT", "RIGHT", "STRAIGHT", or None
        """
        # Release previous steering keys if direction changed
        if self.current_steering != direction:
            if self.current_steering == "LEFT":
                self.release_key('a')
            elif self.current_steering == "RIGHT":
                self.release_key('d')
            
            # Press new steering key
            if direction == "LEFT":
                self.press_key('a')
            elif direction == "RIGHT":
                self.press_key('d')
            # STRAIGHT or None: keys already released above
            
            self.current_steering = direction
    
    def update_brake(self, is_braking: bool):
        """
        Update brake key based on brake status.
        
        Args:
            is_braking: True to press brake (S key), False to release
        """
        if is_braking and not self.is_brake_pressed:
            self.press_key('s')
            self.is_brake_pressed = True
        elif not is_braking and self.is_brake_pressed:
            self.release_key('s')
            self.is_brake_pressed = False
    
    def update_acceleration(self, should_accelerate: bool):
        """
        Update acceleration key based on fist detection.
        Auto-accelerate (W key) when fists are detected.
        
        Args:
            should_accelerate: True to press acceleration (W key), False to release
        """
        if should_accelerate and not self.is_accelerating:
            self.press_key('w')
            self.is_accelerating = True
        elif not should_accelerate and self.is_accelerating:
            self.release_key('w')
            self.is_accelerating = False


# ============================================================================
# MAIN MODULE
# ============================================================================

class HandSteeringApp:
    """
    Main application that connects hand detection with keyboard input for Roblox.
    Displays camera feed with hand tracking overlay.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.hand_detector = HandDetector()
        self.keyboard_controller = KeyboardController()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Smoothing for steering angle (moving average)
        self.angle_history = deque(maxlen=10)
        self.current_angle = 0.0
        self.is_braking = False
        
        # Detection state
        self.is_detecting = False  # Start/Stop state
        
        # Window name
        self.camera_window_name = "Hand Tracking - Roblox Controller"
        cv2.namedWindow(self.camera_window_name)
        
    def smooth_angle(self, new_angle: float) -> float:
        """
        Apply moving average filter to smooth steering angle.
        
        Args:
            new_angle: New angle reading
            
        Returns:
            Smoothed angle
        """
        self.angle_history.append(new_angle)
        if len(self.angle_history) > 0:
            return sum(self.angle_history) / len(self.angle_history)
        return new_angle
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[float], bool]:
        """
        Process a camera frame to extract steering angle and brake status.
        
        Args:
            frame: Input camera frame
            
        Returns:
            (steering_angle, is_braking) tuple
        """
        # Get image dimensions for hand detector
        h, w = frame.shape[:2]
        
        # Detect hands
        hand_landmarks_list, hand_sides_list = self.hand_detector.detect_hands(frame)
        
        # Get hand centers and classify gestures
        hand_centers = []
        is_braking = False
        is_accelerating = False  # Track if fists are detected for acceleration
        gesture_classifications = []
        
        # Check for open palm (brake) and fists (acceleration)
        fist_count = 0
        for hand_landmarks in hand_landmarks_list:
            is_open_palm_gesture = self.hand_detector.is_open_palm(hand_landmarks)
            is_fist = self.hand_detector.is_fist_closed(hand_landmarks)
            
            if is_open_palm_gesture:
                is_braking = True  # Open palm triggers brake
            if is_fist:
                fist_count += 1
        
        # Auto-accelerate when both hands are fists (and not braking)
        if fist_count == 2 and not is_braking:
            is_accelerating = True
        
        # Get hand centers for steering
        for hand_landmarks in hand_landmarks_list:
            hand_center = self.hand_detector.get_hand_center(hand_landmarks, w, h)
            if hand_center:
                hand_centers.append(hand_center)
        
        # Calculate steering based on hand height comparison
        angle = None
        steering_direction = None  # Initialize to None
        
        if len(hand_centers) == 2:
            # Identify left and right hands based on x-position
            if hand_centers[0][0] < hand_centers[1][0]:
                # hand_centers[0] is left, hand_centers[1] is right
                left_hand = hand_centers[0]
                right_hand = hand_centers[1]
            else:
                # hand_centers[1] is left, hand_centers[0] is right
                left_hand = hand_centers[1]
                right_hand = hand_centers[0]
            
            # Compare heights (y-position: smaller y = higher on screen)
            left_hand_y = left_hand[1]
            right_hand_y = right_hand[1]
            
            # Calculate height difference
            height_diff = abs(right_hand_y - left_hand_y)
            
            # Calculate average height for margin calculation
            avg_height = (left_hand_y + right_hand_y) / 2
            
            # 10% margin of error for straight driving
            margin_threshold = avg_height * 0.10
            
            # Determine steering direction
            # If height difference is within 10% margin, drive straight
            if height_diff <= margin_threshold:
                steering_direction = "STRAIGHT"
                angle = 0.0  # Neutral/straight
            elif left_hand_y < right_hand_y:  # Left hand is higher
                steering_direction = "LEFT"
                # Convert to angle: negative for left steering
                angle = -height_diff * 0.1  # Scale the difference to angle
            else:  # Right hand is higher
                steering_direction = "RIGHT"
                # Convert to angle: positive for right steering
                angle = height_diff * 0.1  # Scale the difference to angle
            
            # Smooth the angle
            angle = self.smooth_angle(angle)
        
        # Set classification labels for each hand
        for hand_landmarks in hand_landmarks_list:
            is_open_palm_gesture = self.hand_detector.is_open_palm(hand_landmarks)
            
            # Determine label: BRAKE if open palm, otherwise steering direction
            if is_open_palm_gesture:
                gesture_label = "BRAKE"
            elif steering_direction:
                gesture_label = steering_direction
            else:
                gesture_label = "LEFT"  # Default when no angle calculated
            
            gesture_classifications.append(gesture_label)
        
        # Draw detection overlay with gesture classifications
        frame = self.hand_detector.draw_detection(
            frame, hand_landmarks_list, hand_centers, angle, is_braking, 
            gesture_classifications, self.is_detecting
        )
        
        # Send keyboard inputs only when detecting
        if self.is_detecting:
            self.keyboard_controller.update_steering(steering_direction)
            self.keyboard_controller.update_brake(is_braking)
            self.keyboard_controller.update_acceleration(is_accelerating)
        else:
            # Release all keys when stopped
            self.keyboard_controller.release_all()
        
        return angle, is_braking
    
    def run(self):
        """Main application loop."""
        print("Hand Steering Controller for Roblox Started!")
        print("Controls:")
        print("  - Press 's' to START detection and keyboard input")
        print("  - Press 'x' to STOP detection and release all keys")
        print("  - Press 'q' to QUIT")
        print("  - Two hands detected: Steering (LEFT/RIGHT/STRAIGHT)")
        print("  - Open palm (either hand): Brake")
        print("-" * 50)
        
        running = True
        
        while running:
            # Read camera frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read camera frame")
                break
            
            # Process frame
            angle, is_braking = self.process_frame(frame)
            
            # Update current angle (use 0 if no hands detected)
            if angle is not None:
                self.current_angle = angle
            else:
                # Gradually return to center when no hands detected
                self.current_angle *= 0.95
            
            # Update brake status
            self.is_braking = is_braking
            
            # Display camera feed
            cv2.imshow(self.camera_window_name, frame)
            
            # Handle keyboard input for Start/Stop/Quit
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                running = False
            elif key == ord('s'):
                if not self.is_detecting:
                    self.is_detecting = True
                    print("\n[STARTED] Detection and keyboard input enabled")
            elif key == ord('x'):
                if self.is_detecting:
                    self.is_detecting = False
                    self.keyboard_controller.release_all()
                    print("\n[STOPPED] Detection and keyboard input disabled")
            
            # Print status to terminal
            status = "STARTED" if self.is_detecting else "STOPPED"
            brake_status = "BRAKING" if self.is_braking else "No brake"
            print(f"\rStatus: {status:7s} | Steering: {self.current_angle:6.1f}° | {brake_status:12s}", 
                  end="", flush=True)
        
        # Cleanup
        self.cleanup()
        print("\nApplication closed.")
    
    def cleanup(self):
        """Clean up resources."""
        # Release all keys before closing
        self.keyboard_controller.release_all()
        self.cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        app = HandSteeringApp()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

