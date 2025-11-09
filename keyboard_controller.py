"""
Keyboard Controller Module
Handles keyboard input simulation for Roblox games using pynput.
"""

from pynput.keyboard import Controller
from typing import Optional


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
    
    def update_steering(self, direction: Optional[str], angle_magnitude: float = 0.0):
        """
        Update steering based on direction and angle magnitude.
        Steeper angles result in more aggressive steering.
        
        Args:
            direction: "LEFT", "RIGHT", "STRAIGHT", or None
            angle_magnitude: Magnitude of the steering angle (0.0 to 1.0+)
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
        
        # For steeper angles, we can simulate more aggressive steering
        # by using rapid key press/release cycles (pulsing)
        # This simulates "harder" steering input
        if direction in ["LEFT", "RIGHT"] and angle_magnitude > 0.3:
            # For steeper angles, we could pulse the key more frequently
            # But for now, we'll keep it simple and just ensure the key is pressed
            # The magnitude is already accounted for in the angle calculation
            pass
    
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

