import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import math

# ---------------------------
# YOLOv8 pose model
# ---------------------------
model = YOLO("yolov8n-pose.pt")  # Make sure you have this model

# ---------------------------
# Game settings
# ---------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAR_WIDTH = 50
CAR_HEIGHT = 30
LANE_WIDTH = 200

MAX_SPEED = 10
ACCELERATION = 0.5
FRICTION = 0.05
STEER_SENSITIVITY = 50  # px difference for steering
BRAKE_SENSITIVITY = 30  # px difference for braking

# ---------------------------
# Pygame init
# ---------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hand Drive Lane Game")
clock = pygame.time.Clock()

# ---------------------------
# Car class
# ---------------------------
class Car:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT - 100
        self.angle = 0
        self.speed = 5  # auto forward

    def update(self, brake, steer_left, steer_right):
        # Auto forward
        self.speed = max(0, self.speed)  # can't go negative
        if brake:
            self.speed -= ACCELERATION
        else:
            self.speed += FRICTION  # slight acceleration if not braking
        self.speed = max(0, min(MAX_SPEED, self.speed))

        # Steering
        if steer_left:
            self.angle += 3
        if steer_right:
            self.angle -= 3

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

        # Keep car on screen
        self.x = max(CAR_WIDTH//2, min(SCREEN_WIDTH - CAR_WIDTH//2, self.x))
        self.y = max(CAR_HEIGHT//2, min(SCREEN_HEIGHT - CAR_HEIGHT//2, self.y))

    def draw(self, surface):
        car_surf = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        car_surf.fill((0, 255, 0))
        rotated = pygame.transform.rotate(car_surf, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rect)

# ---------------------------
# Helper functions
# ---------------------------
def get_wrist_position(kpts_array):
    if kpts_array.shape[0] > 0:
        wrist = kpts_array[0]
        wrist = np.array(wrist).reshape(-1)
        return wrist
    return None

# Draw road
def draw_road(surface):
    surface.fill((50, 50, 50))  # asphalt
    lane_center = SCREEN_WIDTH // 2
    left_lane = lane_center - LANE_WIDTH // 2
    right_lane = lane_center + LANE_WIDTH // 2
    pygame.draw.rect(surface, (100, 100, 100), (left_lane, 0, LANE_WIDTH, SCREEN_HEIGHT))
    # lane divider lines
    for y in range(0, SCREEN_HEIGHT, 40):
        pygame.draw.line(surface, (255, 255, 255), (lane_center, y), (lane_center, y+20), 5)

# ---------------------------
# Main loop
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)
    car = Car()
    base_wrist = None

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

        wrist_pos = None
        if results.keypoints is not None and len(results.keypoints) > 0:
            kpts_array = results.keypoints[0].xy
            wrist = get_wrist_position(kpts_array)
            if wrist is not None:
                wrist_pos = wrist
                x, y = int(wrist_pos[0]), int(wrist_pos[1])
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        if wrist_pos is not None and base_wrist is None:
            base_wrist = wrist_pos

        steer_left = steer_right = brake = False
        if wrist_pos is not None and base_wrist is not None:
            dx = wrist_pos[0] - base_wrist[0]
            dy = wrist_pos[1] - base_wrist[1]

            if dx > STEER_SENSITIVITY:
                steer_right = True
            elif dx < -STEER_SENSITIVITY:
                steer_left = True

            if dy > BRAKE_SENSITIVITY:
                brake = True

        # Debug
        print("Steer Left:", steer_left, "Steer Right:", steer_right, "Brake:", brake)

        # Update car
        car.update(brake, steer_left, steer_right)

        # Draw game
        draw_road(screen)
        car.draw(screen)
        pygame.display.flip()
        clock.tick(30)

        # Optional camera debug
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
