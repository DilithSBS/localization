import cv2
import cv2.aruco as aruco
import numpy as np
import math

# 1. Load Camera Calibration (Uses the pre-saved calibration in the repo)
calib_file = "board/calibration_data.txt"
cv_file = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
cameraMatrix = cv_file.getNode("K").mat()
distCoeffs = cv_file.getNode("D").mat()
cv_file.release()

# 2. Setup ArUco Detector (Dictionary: 6x6_250 as used in Pera-Swarm)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Modern OpenCV 4.7+ detector vs legacy fallback
if hasattr(aruco, 'ArucoDetector'):
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    detect_fn = lambda img: detector.detectMarkers(img)
else:
    detect_fn = lambda img: aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# 3. Open Webcam (0 is default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam live! Show Marker (DICT_6X6_250, IDs 0-5) to camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detect_fn(frame)

    if ids is not None:
        # Draw bounding boxes around detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate 3D pose (marker physical size: 50mm)
        marker_size_mm = 50.0
        
        for i in range(len(ids)):
            marker_id = ids[i][0]
            
            # Use solvePnP for robust cross-version pose estimation
            obj_points = np.array([
                [-marker_size_mm/2,  marker_size_mm/2, 0],
                [ marker_size_mm/2,  marker_size_mm/2, 0],
                [ marker_size_mm/2, -marker_size_mm/2, 0],
                [-marker_size_mm/2, -marker_size_mm/2, 0]
            ], dtype=np.float32)
            
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], cameraMatrix, distCoeffs)
            
            # Extract coordinates relative to camera (in cm)
            x_cm = tvec[0][0] / 10.0
            y_cm = tvec[1][0] / 10.0
            z_cm = tvec[2][0] / 10.0  # Distance from camera
            
            # Draw 3D coordinate axes on the marker
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 30)

            # Display telemetry on screen
            info_text = f"ID: {marker_id} | X: {x_cm:.1f}cm Y: {y_cm:.1f}cm Dist: {z_cm:.1f}cm"
            cv2.putText(frame, info_text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Detected Robot {info_text}")

    cv2.imshow("Pera-Swarm Webcam Localization Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
