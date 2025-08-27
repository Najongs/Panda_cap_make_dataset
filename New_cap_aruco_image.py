import pyzed.sl as sl
import cv2
import threading
import time
import os
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

# --- Configuration ---

# Main directory to save the output files
OUTPUT_DIR = "./ArUco_capture_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of cameras to use, defined by their serial number and a view name
CAMERAS_CONFIG = [
    {"serial_number": 41182735, "view_name": "view1"}, # front
    {"serial_number": 49429257, "view_name": "view2"}, # right
    {"serial_number": 44377151, "view_name": "view3"}, # left
    {"serial_number": 49045152, "view_name": "view4"}, # top
]

# ArUco marker configuration
MARKER_SIZE_METERS = 0.05 # The actual size of the marker in meters

# --- ArUco Capture Thread Class ---

class ArUcoCaptureThread(threading.Thread):
    """
    A thread that manages a single ZED camera for ArUco marker detection.
    It initializes the camera, waits for a start signal, captures one frame,
    detects markers, and saves the annotated image and pose data.
    """
    def __init__(self, serial_number, view_name, start_event, timestamp_str):
        super().__init__()
        self.serial_number = serial_number
        self.view_name = view_name
        self.output_dir = os.path.join(OUTPUT_DIR, self.view_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Threading and timing
        self.start_event = start_event
        self.timestamp_str = timestamp_str

        # ZED SDK objects
        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()

        # Camera and ArUco parameters (will be initialized later)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.aruco_detector = None
        self.ready = False

    def init_camera_and_aruco(self):
        """Initializes the ZED camera and sets up ArUco detector."""
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.METER
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Optional, but good practice

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print(f"ERROR: Failed to open camera {self.serial_number}")
            return

        print(f"Camera {self.serial_number} ({self.view_name}) initialized.")
        
        # Get camera calibration parameters for the LEFT camera
        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.camera_matrix = np.array([
            [calibration_params.fx, 0, calibration_params.cx],
            [0, calibration_params.fy, calibration_params.cy],
            [0, 0, 1]
        ])
        # Note: solvePnP requires a 5-element distortion vector (k1, k2, p1, p2, k3).
        self.dist_coeffs = np.array(calibration_params.disto[:5]).reshape(5, 1)

        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        self.ready = True

    def run(self):
        """The main execution method for the thread."""
        if not self.ready:
            print(f"Camera {self.serial_number} not ready. Skipping capture.")
            return

        # Wait for the main thread to signal the start
        self.start_event.wait()
        print(f"--> Capturing from {self.view_name} (S/N: {self.serial_number})")

        image_zed = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image (as it corresponds to the calibration data used)
            self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()[:, :, :3] # Get BGR channels

            # Detect markers
            corners, ids, _ = self.aruco_detector.detectMarkers(frame)

            marker_data = {}
            if ids is not None:
                # Refine corners for better accuracy
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                for corner in corners:
                    cv2.cornerSubPix(gray, corner, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

                # Estimate pose for each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_METERS, self.camera_matrix, self.dist_coeffs)

                # Draw decorations and collect data
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i]
                    tvec = tvecs[i]

                    # Draw axis on the marker
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, MARKER_SIZE_METERS / 2)

                    # Prepare data for JSON file
                    position = tvec.flatten().tolist()
                    quat = R.from_rotvec(rvec.flatten()).as_quat().tolist() # [x, y, z, w]

                    marker_data[int(marker_id)] = {
                        "position_m": {
                            "x": round(position[0], 6),
                            "y": round(position[1], 6),
                            "z": round(position[2], 6)
                        },
                        "rotation_quat": {
                            "x": round(quat[0], 6),
                            "y": round(quat[1], 6),
                            "z": round(quat[2], 6),
                            "w": round(quat[3], 6)
                        },
                        "corners_pixel": corners[i].reshape((4, 2)).tolist()
                    }

            # Save the results
            self.save_results(frame, marker_data)
        else:
            print(f"ERROR: Failed to grab frame from camera {self.serial_number}")

        # Clean up
        self.zed.close()
        print(f"Camera {self.serial_number} ({self.view_name}) closed.")

    def save_results(self, image, data):
        """Saves the annotated image and the JSON data file."""
        base_filename = f"{self.view_name}_{self.serial_number}_{self.timestamp_str}"
        
        # Save annotated image
        image_path = os.path.join(self.output_dir, f"{base_filename}.png")
        cv2.imwrite(image_path, image)

        # Save marker data as JSON
        json_path = os.path.join(self.output_dir, f"{base_filename}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"SUCCESS: Saved results for {self.view_name} to {self.output_dir}")

# --- Main Execution ---

def main():
    start_event = threading.Event()
    
    # Generate a single timestamp for this entire capture session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a list of camera threads from the configuration
    threads = [
        ArUcoCaptureThread(
            serial_number=config["serial_number"],
            view_name=config["view_name"],
            start_event=start_event,
            timestamp_str=timestamp
        ) for config in CAMERAS_CONFIG
    ]

    # Initialize all cameras
    for t in threads:
        t.init_camera_and_aruco()

    # Wait until all cameras are confirmed ready
    while not all(t.ready for t in threads):
        print("Waiting for all cameras to be ready...")
        time.sleep(1)
        if not any(t.is_alive() for t in threads) and not all(t.ready for t in threads):
             print("A camera thread failed to initialize. Exiting.")
             sys.exit(1)

    print("\nAll cameras are ready.")

    # Start all threads (they will wait at start_event.wait())
    for t in threads:
        t.start()

    # Countdown before triggering the capture
    delay = 5
    print(f"Starting capture in {delay} seconds...")
    time.sleep(delay)

    # Trigger all threads to start capturing simultaneously
    print("\n--- Triggering Capture Now! ---")
    start_event.set()

    # Wait for all threads to complete their work
    for t in threads:
        t.join()

    print("\n--- Data collection finished. ---")


if __name__ == "__main__":
    main()