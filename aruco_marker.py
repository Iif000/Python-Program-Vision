import cv2
import numpy as np
import sys
import time


def cek_opencv():
    print("OpenCV version:", cv2.__version__)

    if not hasattr(cv2, "aruco"):
        print("ERROR: cv2.aruco tidak tersedia.")
        print("Install dengan: pip install opencv-contrib-python")
        sys.exit(1)


def buka_kamera_index_2():
    index = 2
    print(f"Membuka kamera index {index}...")

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Gagal dengan V4L2, mencoba default...")
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("ERROR: Kamera index 2 tidak bisa dibuka.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    time.sleep(1)

    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print("Kamera index 2 berhasil digunakan.")
            return cap
        time.sleep(0.2)

    print("ERROR: Kamera terbuka tapi tidak ada frame.")
    sys.exit(1)


def buat_aruco():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        aruco_params = cv2.aruco.DetectorParameters_create()
    else:
        aruco_params = cv2.aruco.DetectorParameters()

    return aruco_dict, aruco_params


def deteksi_aruco(frame, aruco_dict, aruco_params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=aruco_params
        )

    return corners, ids


def main():
    cek_opencv()

    cap = buka_kamera_index_2()
    aruco_dict, aruco_params = buat_aruco()

    print("Tekan 'q' untuk keluar")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame")
            break

        corners, ids = deteksi_aruco(frame, aruco_dict, aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, corner in enumerate(corners):
                pts = corner[0].astype(int)

                center_x = int(np.mean(pts[:, 0]))
                center_y = int(np.mean(pts[:, 1]))

                marker_id = int(ids[i][0])

                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.putText(frame,
                            f"ID:{marker_id}",
                            (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0), 2)

                print(f"ID {marker_id} | x={center_x}, y={center_y}")

        else:
            cv2.putText(frame,
                        "Tidak ada marker",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255), 2)

        cv2.imshow("Aruco Kamera Index 2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai")


if __name__ == "__main__":
    main()