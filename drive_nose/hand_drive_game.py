import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import math

# ---------------------------
# YOLOv8 pose model
# ---------------------------
model = YOLO("yolov8n-pose.pt")  # Full-body pose model

# ---------------------------
# Game settings
# ---------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAR_WIDTH = 50
CAR_HEIGHT = 30
MAX_SPEED = 10
ACCELERATION = 0.5
FRICTION = 0.2
STEER_ANGLE_THRESHOLD = 15  # degrees

# ---------------------------
# Pygame init
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hand Drive Game")
clock = pygame.time.Clock()

# ---------------------------
# Car class
# ---------------------------
class Car:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT - 100
        self.angle = 0
        self.speed = 5  # auto-forward

    def update(self, forward, brake, steer_left, steer_right):
        if forward:
            self.speed += ACCELERATION
        elif brake:
            self.speed -= ACCELERATION
        else:
            self.speed *= (1 - FRICTION)

        self.speed = max(0, min(MAX_SPEED, self.speed))

        if steer_left:
            self.angle += 3
        if steer_right:
            self.angle -= 3

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

        self.x = max(50, min(SCREEN_WIDTH-50, self.x))
        self.y = max(50, min(SCREEN_HEIGHT-50, self.y))

    def draw(self, surface):
        rect = pygame.Rect(0, 0, CAR_WIDTH, CAR_HEIGHT)
        rect.center = (self.x, self.y)
        car_surf = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        car_surf.fill((0, 255, 0))
        rotated = pygame.transform.rotate(car_surf, self.angle)
        surface.blit(rotated, rotated.get_rect(center=rect.center))

# ---------------------------
# Helper functions
# ---------------------------
def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(-dy, dx))

def detect_thumb_up(hand_kpts):
    if hand_kpts.shape[0] < 2:
        return None
    wrist = hand_kpts[0]
    thumb_tip = hand_kpts[-1]  # use last visible point as thumb
    v = thumb_tip - wrist
    angle = np.degrees(np.arctan2(-v[1], v[0]))
    angle_from_vertical = ((angle - 90 + 180) % 360) - 180
    if abs(angle_from_vertical) < 40:
        return 'up'
    elif abs(abs(angle_from_vertical) - 180) < 40:
        return 'down'
    return None

# ---------------------------
# Main loop
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)
    car = Car()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return

        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]

        hand_centers = []
        thumb_action = None

        if results.keypoints is not None:
            for kpts in results.keypoints:
                kpts_array = kpts.xy
                if kpts_array.shape[0] < 2:
                    continue

                # Take first two points as hand centers (wrist/finger)
                hand_kpts = kpts_array[:2]
                hand_centers.extend(hand_kpts)
                gesture = detect_thumb_up(hand_kpts)
                if gesture:
                    thumb_action = gesture

                # Draw keypoints
                for point in hand_kpts:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Steering logic
        steer_left = steer_right = forward = brake = False

        if len(hand_centers) >= 2:
            p1 = np.array(hand_centers[0])
            p2 = np.array(hand_centers[1])
            steering_angle = angle_between_points(p1, p2)
            if steering_angle > STEER_ANGLE_THRESHOLD:
                steer_right = True
            elif steering_angle < -STEER_ANGLE_THRESHOLD:
                steer_left = True

        if thumb_action == 'up':
            forward = True
        elif thumb_action == 'down':
            brake = True

        print("Steer Left:", steer_left, "Steer Right:", steer_right,
              "Forward:", forward, "Brake:", brake)

        # Update car
        car.update(forward, brake, steer_left, steer_right)

        # Draw game
        screen.fill((30, 30, 30))
        # Draw lanes
        pygame.draw.line(screen, (255, 255, 255), (SCREEN_WIDTH//3, 0), (SCREEN_WIDTH//3, SCREEN_HEIGHT), 2)
        pygame.draw.line(screen, (255, 255, 255), (2*SCREEN_WIDTH//3, 0), (2*SCREEN_WIDTH//3, SCREEN_HEIGHT), 2)
        car.draw(screen)
        pygame.display.flip()
        clock.tick(30)

        # Show camera feed
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
