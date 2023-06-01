import cv2
import numpy as np
import math
import random

class MagneticCoil:
    def __init__(self, position):
        self.current = 2.0  # Amperes
        self.wire_radius = 0.0003  # meters
        self.wire_length = 9.0  # meters
        self.num_wraps = 214
        self.coil_radius = 0.05  # meters
        self.max_force = 20.0  # Newtons
        self.position = position

    def get_magnetic_force(self, position, velocity):
        # Calculate the magnetic force between the coil and the player
        distance = np.linalg.norm(position - self.position)
        direction = (self.position - position) / distance
        magnetic_force = direction * distance  # Adjust this calculation based on your requirements
        return magnetic_force


class Player:
    def __init__(self, max_speed=4.5):
        self.max_speed = max_speed
        self.drag_coefficient = 1.0
        self.hydro_radius = 0.3314
        self.fluid_viscosity = 0.00089
        self.rb = None
        self.magnetic_force = None
        self.coils = None
        self.is_attached = False
        self.attachment_coil = None

    def start(self, rb, coils):
        self.rb = rb
        self.coils = coils

    def move(self, direction):
        if self.rb is not None:
            desired_velocity = direction * self.max_speed
            velocity_change = desired_velocity - self.rb.velocity
            self.rb.add_force(velocity_change, force_mode="VelocityChange")

    def attach(self, coil):
        self.attachment_coil = coil
        self.is_attached = True

    def update(self, position, velocity):
        if self.is_attached and self.attachment_coil is not None:
            self.magnetic_force = self.attachment_coil.get_magnetic_force(position, velocity)
        else:
            self.magnetic_force = np.zeros(3)
            for coil in self.coils:
                self.magnetic_force += coil.get_magnetic_force(position, velocity)

        speed = np.linalg.norm(velocity)
        drag_force = self.drag_coefficient * self.fluid_viscosity * np.pi * self.hydro_radius * speed
        drag = -drag_force * velocity / speed if speed > 0 else np.zeros(3)
        self.rb.add_force(drag)

        if speed > self.max_speed:
            self.rb.velocity = self.rb.velocity / np.linalg.norm(self.rb.velocity) * self.max_speed


class ColorTracker:
    def __init__(self, lower_color, upper_color):
        self.lower_color = np.array(lower_color, dtype=np.uint8)
        self.upper_color = np.array(upper_color, dtype=np.uint8)
        self.tracked_object = None
        self.tracking_threshold = 0.5

    def track_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            detection = np.array([[x / frame.shape[1], y / frame.shape[0], (x + w) / frame.shape[1], (y + h) / frame.shape[0]]])
            confidence = cv2.contourArea(largest_contour) / (frame.shape[0] * frame.shape[1])
        else:
            detection = np.empty((0, 4))
            confidence = 0.0

        if confidence > self.tracking_threshold:
            self.tracked_object = detection

        return detection


def main():
    # Initialize magnetic coils and player
    coil_positions = [
        [0.1, 0.1, 0.0],  # Specify the positions of the coils in the pool (x, y, z) in camera's perspective
        [0.2, 0.2, 0.0],
        [0.3, 0.3, 0.0],
        [0.4, 0.4, 0.0],
        [0.5, 0.5, 0.0],
        [0.6, 0.6, 0.0],
        [0.7, 0.7, 0.0],
        [0.8, 0.8, 0.0]
    ]
    coils = [MagneticCoil(position) for position in coil_positions]
    player = Player()

    # Set up color tracker
    color_tracker = ColorTracker([0, 0, 0], [50, 50, 50])  # Adjust the lower and upper color values as per your requirement

    # Set up camera
    camera = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, frame = camera.read()

        # Track color in the frame
        detection = color_tracker.track_color(frame)

        # Process detection
        if len(detection) > 0:
            x, y, _, _ = detection[0]
            x = int(x * frame.shape[1])
            y = int(y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Control player's movement based on detected position
            player.move(np.array([(x - frame.shape[1] / 2) / (frame.shape[1] / 2),
                                  (y - frame.shape[0] / 2) / (frame.shape[0] / 2),
                                  0.0]))

            # Attach player to a coil if not already attached
            if not player.is_attached:
                for coil in coils:
                    if np.array_equal(coil.position, [x, y, 0.0]):
                        player.attach(coil)
                        break

        # Show frame with detections
        cv2.imshow("Color Tracking", frame)

        # Check for exit key
        if cv2.waitKey(1) == ord('q'):
            break

    # Release camera and destroy windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()