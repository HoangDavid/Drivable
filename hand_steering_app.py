"""
Hand Steering Application Module
Main application that coordinates hand detection with keyboard input for Roblox.
"""

import cv2
import numpy as np
import math
from collections import deque
from typing import Tuple, Optional

from hand_detector import HandDetector
from keyboard_controller import KeyboardController


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
        is_accelerating = False  # Track if acceleration should be active
        gesture_classifications = []
        
        # Check for open palm (brake)
        for hand_landmarks in hand_landmarks_list:
            is_open_palm_gesture = self.hand_detector.is_open_palm(hand_landmarks)
            if is_open_palm_gesture:
                is_braking = True  # Open palm triggers brake
                break
        
        # Get hand centers for steering
        for hand_landmarks in hand_landmarks_list:
            hand_center = self.hand_detector.get_hand_center(hand_landmarks, w, h)
            if hand_center:
                hand_centers.append(hand_center)
        
        # Auto-accelerate when 2 hands are detected for steering (and not braking)
        # Always accelerate except when braking
        if len(hand_centers) == 2 and not is_braking:
            is_accelerating = True
        
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
            
            # Calculate horizontal distance between hands for angle calculation
            horizontal_dist = abs(right_hand[0] - left_hand[0])
            
            # Calculate angle in degrees from height difference and horizontal distance
            # Use atan2 to get angle: atan2(height_diff, horizontal_dist) gives angle in radians
            if horizontal_dist > 0:
                angle_rad = math.atan2(height_diff, horizontal_dist)
                angle_deg = math.degrees(angle_rad)
            else:
                angle_deg = 0.0
            
            # Fixed angle threshold for straight driving (12.5 degrees average of 10-15)
            straight_threshold_deg = 12.5
            
            # Determine steering direction based on angle
            if angle_deg <= straight_threshold_deg:
                steering_direction = "STRAIGHT"
                angle = 0.0  # Neutral/straight
            elif left_hand_y < right_hand_y:  # Left hand is higher
                steering_direction = "LEFT"
                # Calculate angle magnitude based on how much it exceeds the threshold
                # Steeper angles (larger angle) = larger angle magnitude
                excess_angle = angle_deg - straight_threshold_deg
                # Scale to create proportional steering (negative for left)
                angle = -excess_angle
            else:  # Right hand is higher
                steering_direction = "RIGHT"
                # Calculate angle magnitude based on how much it exceeds the threshold
                excess_angle = angle_deg - straight_threshold_deg
                # Scale to create proportional steering (positive for right)
                angle = excess_angle
            
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
            # Calculate angle magnitude for proportional steering
            angle_magnitude = abs(angle) if angle is not None else 0.0
            # Normalize to 0-1 range for steering intensity (assuming max angle ~90)
            normalized_magnitude = min(angle_magnitude / 90.0, 1.0) if angle_magnitude > 0 else 0.0
            
            self.keyboard_controller.update_steering(steering_direction, normalized_magnitude)
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

