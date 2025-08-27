import pyzed.sl as sl
import cv2
import threading
import time
import os
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from datetime import datetime

# --- Configuration ---
OUTPUT_DIR = "./ArUco_capture_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERAS_CONFIG = [
    {"serial_number": 41182735, "view_name": "view1"}, # front
    {"serial_number": 49429257, "view_name": "view2"}, # right
    {"serial_number": 44377151, "view_name": "view3"}, # left
    {"serial_number": 49045152, "view_name": "view4"}, # top
]
CAM_VIEW_CONFIG = {"leftcam": sl.VIEW.LEFT, "rightcam": sl.VIEW.RIGHT}
MARKER_SIZE_METERS = 0.05

# --- Helper Functions for Smoothing (Included as requested, but not used in single-shot capture) ---
def slerp_ema_stable(prev_quat, curr_quat, alpha):
    if np.dot(prev_quat, curr_quat) < 0.0:
        curr_quat = -curr_quat
    key_rots = R.from_quat([prev_quat, curr_quat])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([alpha])[0]
    return interp_rot.as_quat()

# --- ArUco Capture Thread Class ---
class ArUcoCaptureThread(threading.Thread):
    def __init__(self, serial_number, view_name, start_event, timestamp_str):
        super().__init__()
        self.serial_number = serial_number
        self.view_name = view_name
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.start_event = start_event
        self.timestamp_str = timestamp_str
        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.calib_data = {}
        self.aruco_detector = None
        self.ready = False

    def init_camera_and_aruco(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.METER
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print(f"ERROR: Failed to open camera {self.serial_number}")
            return

        print(f"Camera {self.serial_number} ({self.view_name}) initialized.")
        cam_info = self.zed.get_camera_information().camera_configuration
        
        for view_name, view in CAM_VIEW_CONFIG.items():
            params = cam_info.calibration_parameters.left_cam if view == sl.VIEW.LEFT else cam_info.calibration_parameters.right_cam
            self.calib_data[view] = {
                "matrix": np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]]),
                "dist": np.array(params.disto) # Store full distortion coeffs
            }
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.ready = True

    def run(self):
        if not self.ready:
            return
        self.start_event.wait()
        print(f"--> Capturing from {self.view_name} (S/N: {self.serial_number})")

        image_zed = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            for cam_view_name, cam_view_sl in CAM_VIEW_CONFIG.items():
                self.zed.retrieve_image(image_zed, cam_view_sl)
                frame = image_zed.get_data()[:, :, :3].copy()
                
                cam_matrix = self.calib_data[cam_view_sl]["matrix"]
                dist_coeffs_full = self.calib_data[cam_view_sl]["dist"]
                
                # 1. Undistort the image first
                frame_undistorted = cv2.undistort(frame, cam_matrix, dist_coeffs_full)
                
                # Detect markers on the undistorted image
                corners, ids, _ = self.aruco_detector.detectMarkers(frame_undistorted)
                marker_data = {}

                if ids is not None:
                    # 2. Refine corner detection
                    gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                    for corner in corners:
                        cv2.cornerSubPix(gray, corner, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
                    
                    marker_3d_edges = np.array([[0, 0, 0], [MARKER_SIZE_METERS, 0, 0], 
                                                [MARKER_SIZE_METERS, MARKER_SIZE_METERS, 0], [0, MARKER_SIZE_METERS, 0]], dtype='float32')
                    dist_coeffs_for_pnp = np.zeros((5, 1)) # Use zero distortion as image is already undistorted

                    for i, marker_id_arr in enumerate(ids):
                        marker_id = int(marker_id_arr[0])
                        corner = corners[i]
                        
                        # 3. Use solvePnP for robust pose estimation
                        ret, rvec, tvec = cv2.solvePnP(marker_3d_edges, corner, cam_matrix, dist_coeffs_for_pnp, flags=cv2.SOLVEPNP_ITERATIVE)
                        if ret:
                            rvec, tvec = cv2.solvePnPRefineLM(marker_3d_edges, corner, cam_matrix, dist_coeffs_for_pnp, rvec, tvec)
                            
                            # 4. Draw detailed visualizations
                            cv2.drawFrameAxes(frame_undistorted, cam_matrix, dist_coeffs_for_pnp, rvec, tvec, MARKER_SIZE_METERS / 2)
                            topLeft = (int(corner[0][0][0]), int(corner[0][0][1]))
                            pos_text = f"Pos: ({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f})m"
                            cv2.putText(frame_undistorted, f"ID: {marker_id}", (topLeft[0]-10, topLeft[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame_undistorted, pos_text, (topLeft[0]-10, topLeft[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Store data for JSON
                            position = tvec.flatten().tolist()
                            quat = R.from_rotvec(rvec.flatten()).as_quat().tolist()
                            quat_text = f"Rot (quat): ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})"
                            cv2.putText(frame_undistorted, quat_text, (topLeft[0]-10, topLeft[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                            marker_data[marker_id] = {
                                "position_m": {"x": position[0], "y": position[1], "z": position[2]},
                                "rotation_quat": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
                                "corners_pixel": corner.reshape((4, 2)).tolist()
                            }
                
                self.save_results(frame_undistorted, marker_data, cam_view_name)
        else:
            print(f"ERROR: Failed to grab frame from camera {self.serial_number}")
        
        self.zed.close()
        print(f"Camera {self.serial_number} ({self.view_name}) closed.")

    def save_results(self, image, data, cam_view_name):
        base_filename = f"{self.view_name}_{self.serial_number}_{cam_view_name}_{self.timestamp_str}"
        image_path = os.path.join(self.output_dir, f"{base_filename}.png")
        cv2.imwrite(image_path, image)
        json_path = os.path.join(self.output_dir, f"{base_filename}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"SUCCESS: Saved results for {self.view_name}/{cam_view_name}")

# --- Main Execution ---
def main():
    start_event = threading.Event()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    threads = [ArUcoCaptureThread(c["serial_number"], c["view_name"], start_event, timestamp) for c in CAMERAS_CONFIG]

    for t in threads:
        t.init_camera_and_aruco()

    while not all(t.ready for t in threads):
        print("Waiting for all cameras to be ready...")
        time.sleep(1)
        if not any(t.is_alive() for t in threads) and not all(t.ready for t in threads):
             print("A camera thread failed to initialize. Exiting.")
             sys.exit(1)

    print("\nAll cameras are ready.")
    for t in threads:
        t.start()

    delay = 5
    print(f"Starting capture in {delay} seconds...")
    time.sleep(delay)
    print("\n--- Triggering Capture Now! ---")
    start_event.set()

    for t in threads:
        t.join()
    print("\n--- Data collection finished. ---")

if __name__ == "__main__":
    main()