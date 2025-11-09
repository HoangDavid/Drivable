"""
Hand Detector Module
Handles MediaPipe Hands detection, gesture recognition, and visualization.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Tuple, Optional, List


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
            is_detecting: Whether detection is active
            
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

