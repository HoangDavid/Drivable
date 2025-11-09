"""
Hand Steering Game
A Python application that uses MediaPipe Hands to control a Pygame driving game.
Steering is controlled by the angle of a line connecting two fists,
and braking is triggered by thumbs up gestures.
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
from collections import deque
from typing import Tuple, Optional, List


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
                      gesture_classifications: List[str] = None) -> np.ndarray:
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
        
        return image


# ============================================================================
# GAME MODULE
# ============================================================================

class DrivingGame:
    """
    Pygame-based driving game with a car sprite that moves forward automatically.
    Steering adjusts horizontal position, and braking slows forward speed.
    """
    
    def __init__(self, width: int = 400, height: int = 600):
        """
        Initialize the driving game.
        
        Args:
            width: Game window width
            height: Game window height
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Hand Steering Game")
        
        # Game state
        self.car_x = width // 2
        self.car_y = height - 100
        self.base_speed = 3.0
        self.current_speed = self.base_speed
        self.steering_sensitivity = 2.0
        
        # Road boundaries
        self.road_left = width // 4
        self.road_right = 3 * width // 4
        
        # Create simple car sprite (rectangle)
        self.car_width = 40
        self.car_height = 60
        self.car_color = (255, 0, 0)  # Red car
        
    def update(self, steering_angle: float, is_braking: bool):
        """
        Update game state based on steering angle and brake status.
        
        Args:
            steering_angle: Steering angle in degrees (-90 to 90)
            is_braking: Whether brake is active
        """
        # Normalize steering angle to -1 to 1 range
        # Map -90 to 90 degrees to -1 to 1
        normalized_angle = steering_angle / 90.0
        normalized_angle = max(-1.0, min(1.0, normalized_angle))
        
        # Apply steering (adjust car x position)
        # Negate to fix inverse control: tilt left = steer left, tilt right = steer right
        self.car_x -= normalized_angle * self.steering_sensitivity
        
        # Keep car within road boundaries
        self.car_x = max(self.road_left + self.car_width // 2, 
                        min(self.road_right - self.car_width // 2, self.car_x))
        
        # Apply braking (reduce speed)
        if is_braking:
            self.current_speed = max(0.5, self.current_speed * 0.95)
        else:
            # Gradually return to base speed
            self.current_speed = min(self.base_speed, 
                                    self.current_speed * 1.02)
        
        # Move car forward
        self.car_y -= self.current_speed
        
        # Reset car position when it goes off screen (top)
        if self.car_y < -self.car_height:
            self.car_y = self.height
    
    def draw(self):
        """Draw the game scene."""
        # Clear screen (dark gray background)
        self.screen.fill((40, 40, 40))
        
        # Draw road
        pygame.draw.rect(self.screen, (60, 60, 60), 
                        (self.road_left, 0, 
                         self.road_right - self.road_left, self.height))
        
        # Draw road center line (dashed)
        center_x = self.width // 2
        for y in range(0, self.height, 40):
            pygame.draw.line(self.screen, (255, 255, 0), 
                           (center_x, y), (center_x, y + 20), 2)
        
        # Draw road boundaries
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (self.road_left, 0), (self.road_left, self.height), 3)
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (self.road_right, 0), (self.road_right, self.height), 3)
        
        # Draw car
        car_rect = pygame.Rect(
            self.car_x - self.car_width // 2,
            self.car_y - self.car_height // 2,
            self.car_width,
            self.car_height
        )
        pygame.draw.rect(self.screen, self.car_color, car_rect)
        
        # Draw car details (simple windows)
        window_rect = pygame.Rect(
            self.car_x - self.car_width // 4,
            self.car_y - self.car_height // 3,
            self.car_width // 2,
            self.car_height // 3
        )
        pygame.draw.rect(self.screen, (100, 150, 255), window_rect)
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self):
        """Handle Pygame events (for window closing)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True


# ============================================================================
# MAIN MODULE
# ============================================================================

class HandSteeringApp:
    """
    Main application that connects hand detection with the driving game.
    Displays camera feed and game window side-by-side.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.hand_detector = HandDetector()
        self.game = DrivingGame()
        
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
        
        # Window positions for side-by-side display
        self.camera_window_name = "Hand Tracking"
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
        
        # Calculate steering based on hand height comparison
        angle = None
        steering_direction = None
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
            frame, hand_landmarks_list, hand_centers, angle, is_braking, gesture_classifications
        )
        
        return angle, is_braking
    
    def run(self):
        """Main application loop."""
        print("Hand Steering Game Started!")
        print("Controls:")
        print("  - Two hands detected: Steering (angle of line between hands)")
        print("  - Open palm (either hand): Brake")
        print("  - Press 'q' in camera window to quit")
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
            
            # Update game
            self.game.update(self.current_angle, self.is_braking)
            self.game.draw()
            
            # Check if game window should close
            if not self.game.handle_events():
                running = False
            
            # Display camera feed
            cv2.imshow(self.camera_window_name, frame)
            
            # Print status to terminal
            brake_status = "BRAKING" if self.is_braking else "No brake"
            print(f"\rSteering Angle: {self.current_angle:6.1f}° | {brake_status:12s}", 
                  end="", flush=True)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
        
        # Cleanup
        self.cleanup()
        print("\nApplication closed.")
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


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

