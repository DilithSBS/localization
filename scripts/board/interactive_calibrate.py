import cv2
import numpy as np
import os

# Chessboard inner corners: (cols - 1, rows - 1)
# For the standard 9x6 chessboard in Pera-Swarm/calibration/chess_board.pdf:
CHECKERBOARD = (8, 5) 
SQUARE_SIZE_MM = 25.0  # Measure one square on your screen/paper in mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points (0,0,0), (1,0,0), (2,0,0) ... in real mm
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

objpoints = [] # 3D real world points
imgpoints = [] # 2D image plane points

cap = cv2.VideoCapture(1) # Change to 0 or 1 depending on your webcam ID
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("="*60)
print("WEBCAM CALIBRATION TOOL")
print("1. Point camera at the chessboard.")
print("2. When corners are detected (drawn in color), press [SPACE] to capture a frame.")
print("3. Capture 15-20 frames from different angles/distances.")
print("4. Press [C] to calculate and save calibration data.")
print("5. Press [Q] to quit without saving.")
print("="*60)

captured_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    display_frame = frame.copy()

    if ret_corners:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret_corners)
        status_text = f"Pattern Detected! Captured: {captured_count}/15. Press SPACE to capture."
        color = (0, 255, 0)
    else:
        status_text = f"Searching for {CHECKERBOARD[0]}x{CHECKERBOARD[1]} corners... Captured: {captured_count}"
        color = (0, 0, 255)

    cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Webcam Calibration", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32 and ret_corners:  # SPACEBAR
        objpoints.append(objp)
        imgpoints.append(corners2)
        captured_count += 1
        print(f"Captured frame #{captured_count}")
    elif key == ord('c'):
        if captured_count < 10:
            print("Please capture at least 10 frames before calibrating!")
            continue
        print("Calculating calibration parameters...")
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # Save to board/calibration_data.txt
        output_path = "board/calibration_data.txt"
        cv_file = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("K", K)
        cv_file.write("D", D)
        cv_file.release()
        
        print("\n" + "="*50)
        print("SUCCESSFULLY CALIBRATED!")
        print(f"Reprojection Error: {ret:.4f} pixels")
        print(f"Camera Matrix K:\n{K}")
        print(f"Distortion Coefficients D:\n{D}")
        print(f"Saved to: {output_path}")
        print("="*50)
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
