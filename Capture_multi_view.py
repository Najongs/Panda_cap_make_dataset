import pyzed.sl as sl
import cv2
import threading
import time
import os
import json
import sys

# 저장 폴더
OUTPUT_DIR = "./Panda_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ZedCamera(threading.Thread):
    def __init__(self, serial_number, output_subdir, start_event, duration=30):
        super().__init__()
        self.serial_number = serial_number
        self.output_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.start_event = start_event
        self.duration = duration
        self.ready = False

    def init_camera(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1200
        init_params.camera_fps = 30
        init_params.set_from_serial_number(self.serial_number)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        if self.zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.ready = True
            print(f"Camera {self.serial_number} initialized")
        else:
            print(f"Failed to open camera {self.serial_number}")
            sys.exit(1)

    def run(self):
        if not self.ready:
            print(f"Camera {self.serial_number} not ready. Skipping capture.")
            return

        left_image = sl.Mat()
        right_image = sl.Mat()

        self.start_event.wait()
        print(f"Camera {self.serial_number} started capturing")
        start_time = time.time()

        while time.time() - start_time < self.duration:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                timestamp = time.time()
                timestamp_str = f"{timestamp:.3f}"

                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                left_data = left_image.get_data()
                right_data = right_image.get_data()

                left_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_left_{timestamp_str}.jpg")
                right_path = os.path.join(self.output_dir, f"zed_{self.serial_number}_right_{timestamp_str}.jpg")

                success_left = cv2.imwrite(left_path, left_data[:, :, :3])
                success_right = cv2.imwrite(right_path, right_data[:, :, :3])

                if not (success_left and success_right):
                    print(f"Camera {self.serial_number} - Failed to save images at {timestamp:.3f}")
            
            # 0.1초 대기하여 너무 많은 파일이 생성되지 않도록 조절
            time.sleep(0.1)

        self.zed.close()
        print(f"Camera {self.serial_number} stopped")


def main():
    start_event = threading.Event()
    duration = 30  # 30초 동안 촬영

    cameras = [
        ZedCamera(serial_number=41182735, output_subdir="view1", start_event=start_event, duration=duration),
        ZedCamera(serial_number=49429257, output_subdir="view2", start_event=start_event, duration=duration),
        ZedCamera(serial_number=44377151, output_subdir="view3", start_event=start_event, duration=duration),
        ZedCamera(serial_number=49045152, output_subdir="view4", start_event=start_event, duration=duration),
    ]

    for cam in cameras:
        cam.init_camera()

    while not all(cam.ready for cam in cameras):
        print("Waiting for all cameras to be ready...")
        time.sleep(1)

    for cam in cameras:
        cam.start()

    print("Starting data capture in 3 seconds...")
    time.sleep(3)
    start_event.set()

    for cam in cameras:
        cam.join()

    print("Data collection finished.")


if __name__ == "__main__":
    main()