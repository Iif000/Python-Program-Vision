import cv2
import numpy as np
import sys
import os
import time
import math


# ==================================================
# KONFIGURASI DASAR
# ==================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 25

ARUCO_MARKER_SIZE_CM = 5.0
FOCAL_LENGTH_PX = 700

ARUCO_TARGET_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

FRONT_TOLERANCE_DEGREE = 15

# Grid dibuat agak besar
GRID_SIZE = 13

AUTO_SCAN_INTERVAL_FRAME = 20
COLOR_DETECTION_INTERVAL_FRAME = 1


# ==================================================
# CEK OPENCV
# ==================================================

def cek_opencv():
    print("OpenCV version:", cv2.__version__)

    if not hasattr(cv2, "aruco"):
        print("ERROR: cv2.aruco tidak tersedia.")
        print("Jalankan:")
        print("pip uninstall opencv-python opencv-contrib-python -y")
        print("pip install opencv-contrib-python numpy")
        sys.exit(1)


# ==================================================
# LABEL SWARM
# ==================================================

def ambil_label_swarm(marker_id):
    if marker_id == 1:
        return "Swarm Leader"
    elif marker_id in [2, 3, 4]:
        return "Swarm Follower"
    else:
        return "Unknown"


def ambil_warna_label(marker_id):
    if marker_id == 1:
        return (255, 0, 0)   # biru
    elif marker_id in [2, 3, 4]:
        return (0, 255, 0)     # hijau
    else:
        return (255, 255, 255)


# ==================================================
# ARUCO PARAMETER
# ==================================================

def buat_parameter_aruco():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7

    params.minMarkerPerimeterRate = 0.025
    params.maxMarkerPerimeterRate = 4.0

    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 3

    params.errorCorrectionRate = 0.6

    return params


def daftar_dictionary_aruco():
    return [
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
        ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
        ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),

        ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
        ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
        ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),

        ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
        ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
        ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ]


def buat_detector(aruco_dict, aruco_params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return None


def deteksi_aruco(gray, aruco_dict, aruco_params, detector=None):
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=aruco_params
        )

    return corners, ids


def cari_dictionary_id(dictionary_list, dict_name_dicari):
    for dict_name, dict_id in dictionary_list:
        if dict_name == dict_name_dicari:
            return dict_id

    return cv2.aruco.DICT_4X4_50


# ==================================================
# FIX FLIP UNTUK KOORDINAT ARUCO
# ==================================================

def balik_corner_horizontal(corner, width):
    """
    ArUco dideteksi dari frame asli.
    Kalau tampilan di-flip, titik corner dibalik horizontal
    dan urutan corner disusun ulang agar tetap benar:
    0 = top-left
    1 = top-right
    2 = bottom-right
    3 = bottom-left
    pada frame tampilan.
    """
    corner_baru = corner.copy()

    # Balik koordinat X
    corner_baru[0][:, 0] = width - 1 - corner_baru[0][:, 0]

    # Setelah mirror horizontal, urutan visual harus dibetulkan:
    # TL, TR, BR, BL = old TR, old TL, old BL, old BR
    corner_baru[0] = corner_baru[0][[1, 0, 3, 2]]

    return corner_baru


# ==================================================
# KAMERA
# ==================================================

def buka_kamera(index):
    print(f"Mencoba membuka kamera index {index}...")

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Gagal dengan CAP_V4L2, mencoba backend default...")
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"ERROR: Kamera index {index} tidak bisa dibuka.")
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    time.sleep(1)

    for percobaan in range(10):
        ret, frame = cap.read()

        if ret and frame is not None:
            h, w = frame.shape[:2]
            fps_actual = cap.get(cv2.CAP_PROP_FPS)

            print("Kamera berhasil dibuka.")
            print(f"Resolusi aktual: {w} x {h}")
            print(f"FPS kamera terbaca: {fps_actual}")

            return cap

        print(f"Percobaan baca frame {percobaan + 1}/10 gagal...")
        time.sleep(0.2)

    print("ERROR: Kamera terbuka, tetapi frame tidak terbaca.")
    cap.release()
    return None


def cari_kamera_otomatis():
    print("Mencari kamera otomatis...")

    for index in range(10):
        cap = buka_kamera(index)

        if cap is not None:
            print(f"Menggunakan kamera index {index}")
            return cap, index

    return None, None


# ==================================================
# TRACKBAR HSV TERPISAH
# ==================================================

def kosong(x):
    pass


def buat_trackbar_warna():
    cv2.namedWindow("HSV Kuning")
    cv2.resizeWindow("HSV Kuning", 420, 300)
    cv2.createTrackbar("H Min", "HSV Kuning", 15, 179, kosong)
    cv2.createTrackbar("H Max", "HSV Kuning", 47, 179, kosong)
    cv2.createTrackbar("S Min", "HSV Kuning", 5, 255, kosong)
    cv2.createTrackbar("S Max", "HSV Kuning", 255, 255, kosong)
    cv2.createTrackbar("V Min", "HSV Kuning", 32, 255, kosong)
    cv2.createTrackbar("V Max", "HSV Kuning", 255, 255, kosong)

    cv2.namedWindow("HSV Biru")
    cv2.resizeWindow("HSV Biru", 420, 300)
    cv2.createTrackbar("H Min", "HSV Biru", 90, 179, kosong)
    cv2.createTrackbar("H Max", "HSV Biru", 130, 179, kosong)
    cv2.createTrackbar("S Min", "HSV Biru", 80, 255, kosong)
    cv2.createTrackbar("S Max", "HSV Biru", 255, 255, kosong)
    cv2.createTrackbar("V Min", "HSV Biru", 50, 255, kosong)
    cv2.createTrackbar("V Max", "HSV Biru", 255, 255, kosong)

    cv2.namedWindow("HSV Hijau")
    cv2.resizeWindow("HSV Hijau", 420, 300)
    cv2.createTrackbar("H Min", "HSV Hijau", 35, 179, kosong)
    cv2.createTrackbar("H Max", "HSV Hijau", 85, 179, kosong)
    cv2.createTrackbar("S Min", "HSV Hijau", 70, 255, kosong)
    cv2.createTrackbar("S Max", "HSV Hijau", 255, 255, kosong)
    cv2.createTrackbar("V Min", "HSV Hijau", 50, 255, kosong)
    cv2.createTrackbar("V Max", "HSV Hijau", 255, 255, kosong)


def ambil_hsv_dari_window(nama_window):
    h_min = cv2.getTrackbarPos("H Min", nama_window)
    h_max = cv2.getTrackbarPos("H Max", nama_window)

    s_min = cv2.getTrackbarPos("S Min", nama_window)
    s_max = cv2.getTrackbarPos("S Max", nama_window)

    v_min = cv2.getTrackbarPos("V Min", nama_window)
    v_max = cv2.getTrackbarPos("V Max", nama_window)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    return lower, upper


def buat_trackbar_brightness_contrast():
    """
    Trackbar untuk mengatur brightness dan contrast frame kamera.
    Brightness:
        nilai trackbar 0-200, nilai asli -100 sampai +100
    Contrast:
        nilai trackbar 0-300, nilai asli 0.0 sampai 3.0
        default 100 = contrast normal 1.0
    """
    cv2.namedWindow("Setting Kamera")
    cv2.resizeWindow("Setting Kamera", 420, 140)

    cv2.createTrackbar("Brightness", "Setting Kamera", 100, 200, kosong)
    cv2.createTrackbar("Contrast", "Setting Kamera", 100, 300, kosong)


def ambil_brightness_contrast():
    brightness_trackbar = cv2.getTrackbarPos("Brightness", "Setting Kamera")
    contrast_trackbar = cv2.getTrackbarPos("Contrast", "Setting Kamera")

    brightness = brightness_trackbar - 100
    contrast = contrast_trackbar / 100.0

    return brightness, contrast


def terapkan_brightness_contrast(frame, brightness, contrast):
    """
    Mengatur brightness dan contrast frame.

    Rumus:
    output = frame * contrast + brightness
    """
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)



# ==================================================
# GRID OBJECT DETECTION
# ==================================================

def bbox_intersect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    return not (
        ax + aw < bx or
        bx + bw < ax or
        ay + ah < by or
        by + bh < ay
    )


def gambar_grid(frame, grid_size=60, object_bboxes=None):
    if object_bboxes is None:
        object_bboxes = []

    h, w = frame.shape[:2]

    overlay = frame.copy()

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            grid_w = min(grid_size, w - x)
            grid_h = min(grid_size, h - y)
            grid_bbox = (x, y, grid_w, grid_h)

            terkena_objek = False

            for obj_bbox in object_bboxes:
                if bbox_intersect(grid_bbox, obj_bbox):
                    terkena_objek = True
                    break

            if terkena_objek:
                cv2.rectangle(
                    overlay,
                    (x, y),
                    (x + grid_w, y + grid_h),
                    (0, 255, 0),
                    -1
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + grid_w, y + grid_h),
                    (0, 255, 0),
                    2
                )
            else:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + grid_w, y + grid_h),
                    (60, 60, 60),
                    1
                )

    alpha = 0.25
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            grid_w = min(grid_size, w - x)
            grid_h = min(grid_size, h - y)
            grid_bbox = (x, y, grid_w, grid_h)

            terkena_objek = False

            for obj_bbox in object_bboxes:
                if bbox_intersect(grid_bbox, obj_bbox):
                    terkena_objek = True
                    break

            if terkena_objek:
                warna = (0, 255, 0)
                tebal = 2
            else:
                warna = (60, 60, 60)
                tebal = 1

            cv2.rectangle(
                frame,
                (x, y),
                (x + grid_w, y + grid_h),
                warna,
                tebal
            )

    center_x = w // 2
    center_y = h // 2

    cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 2)
    cv2.line(frame, (0, center_y), (w, center_y), (255, 255, 255), 2)

    cv2.putText(
        frame,
        "CENTER",
        (center_x + 5, center_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


def tampilkan_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


# ==================================================
# DETEKSI WARNA
# ==================================================

def proses_mask_warna(
    frame_deteksi,
    frame_tampil,
    hsv,
    nama_warna,
    nama_window_hsv,
    warna_bgr,
    object_bboxes
):
    lower, upper = ambil_hsv_dari_window(nama_window_hsv)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 900:
            x, y, w, h = cv2.boundingRect(cnt)

            object_bboxes.append((x, y, w, h))

            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(frame_tampil, (x, y), (x + w, y + h), warna_bgr, 2)
            cv2.circle(frame_tampil, (cx, cy), 5, warna_bgr, -1)

            cv2.putText(
                frame_tampil,
                f"Objek {nama_warna}",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                warna_bgr,
                2
            )

    return mask


def deteksi_semua_warna(
    frame_deteksi,
    frame_tampil,
    mode_kuning,
    mode_biru,
    mode_hijau,
    mode_mask,
    object_bboxes
):
    hsv = cv2.cvtColor(frame_deteksi, cv2.COLOR_BGR2HSV)

    mask_kuning = np.zeros(frame_deteksi.shape[:2], dtype=np.uint8)
    mask_biru = np.zeros(frame_deteksi.shape[:2], dtype=np.uint8)
    mask_hijau = np.zeros(frame_deteksi.shape[:2], dtype=np.uint8)

    if mode_kuning:
        mask_kuning = proses_mask_warna(
            frame_deteksi,
            frame_tampil,
            hsv,
            "KUNING",
            "HSV Kuning",
            (0, 255, 255),
            object_bboxes
        )

    if mode_biru:
        mask_biru = proses_mask_warna(
            frame_deteksi,
            frame_tampil,
            hsv, 
            "BIRU",
            "HSV Biru",
            (255, 0, 0),
            object_bboxes
        )

    if mode_hijau:
        mask_hijau = proses_mask_warna(
            frame_deteksi,
            frame_tampil,
            hsv,
            "HIJAU",
            "HSV Hijau",
            (0, 255, 0),
            object_bboxes
        )

    if mode_mask:
        cv2.imshow("Mask Kuning", mask_kuning)
        cv2.imshow("Mask Biru", mask_biru)
        cv2.imshow("Mask Hijau", mask_hijau)


# ==================================================
# PERHITUNGAN ARUCO
# ==================================================

def hitung_jarak_cm(marker_width_px):
    """
    Estimasi jarak marker ke kamera.
    """
    if marker_width_px <= 0:
        return 0

    distance_cm = (ARUCO_MARKER_SIZE_CM * FOCAL_LENGTH_PX) / marker_width_px
    return distance_cm


def hitung_skala_cm_per_px(pts):
    """
    Menghitung skala cm/pixel berdasarkan ukuran marker ArUco.
    Dipakai untuk estimasi jarak antar marker ID 1-2, 2-3, 3-4.
    """
    sisi_atas = np.linalg.norm(pts[0] - pts[1])
    sisi_bawah = np.linalg.norm(pts[3] - pts[2])
    sisi_kiri = np.linalg.norm(pts[0] - pts[3])
    sisi_kanan = np.linalg.norm(pts[1] - pts[2])

    marker_size_px = (sisi_atas + sisi_bawah + sisi_kiri + sisi_kanan) / 4.0

    if marker_size_px <= 0:
        return 0

    return ARUCO_MARKER_SIZE_CM / marker_size_px


def hitung_sudut_marker(pts):
    """
    Menghitung arah depan marker ArUco.

    Konvensi corner:
    0 = top-left
    1 = top-right
    2 = bottom-right
    3 = bottom-left

    FIX:
    Pada robot ini, arah depan fisik robot ternyata berlawanan
    dengan sisi atas marker. Jadi arah DEPAN dihitung dari center
    marker menuju titik tengah sisi bawah marker.
    """
    center = np.mean(pts, axis=0)

    bottom_right = pts[2]
    bottom_left = pts[3]

    bottom_mid = (bottom_right + bottom_left) / 2.0

    dx = bottom_mid[0] - center[0]
    dy = bottom_mid[1] - center[1]

    return math.degrees(math.atan2(dy, dx))


def gambar_panah_depan(frame, center_x, center_y, angle_degree, panjang=70):
    angle_rad = math.radians(angle_degree)

    end_x = int(center_x + panjang * math.cos(angle_rad))
    end_y = int(center_y + panjang * math.sin(angle_rad))

    cv2.arrowedLine(
        frame,
        (center_x, center_y),
        (end_x, end_y),
        (0, 255, 255),
        4,
        tipLength=0.35
    )

    # Label teks "DEPAN" tidak ditampilkan di frame kamera.
    # Informasi arah depan dipindahkan ke Panel Tombol.


def hitung_jarak_antar_marker_berurutan(marker_data):
    """
    Menghitung jarak antar marker berurutan:
    ID 1 - ID 2
    ID 2 - ID 3
    ID 3 - ID 4

    Return:
    [
        {
            "id_a": 1,
            "id_b": 2,
            "jarak_cm": nilai
        },
        ...
    ]
    """
    pasangan_marker = [
        (1, 2),
        (2, 3),
        (3, 4)
    ]

    hasil = []

    for id_a, id_b in pasangan_marker:
        if id_a not in marker_data or id_b not in marker_data:
            continue

        center_a = marker_data[id_a]["center"]
        center_b = marker_data[id_b]["center"]

        scale_a = marker_data[id_a]["scale_cm_per_px"]
        scale_b = marker_data[id_b]["scale_cm_per_px"]

        if scale_a <= 0 or scale_b <= 0:
            continue

        # Pakai rata-rata skala dari dua marker.
        scale_rata = (scale_a + scale_b) / 2.0

        dx = center_b[0] - center_a[0]
        dy = center_b[1] - center_a[1]

        jarak_px = math.sqrt(dx * dx + dy * dy)
        jarak_cm = jarak_px * scale_rata

        hasil.append({
            "id_a": id_a,
            "id_b": id_b,
            "jarak_cm": jarak_cm
        })

    return hasil


def gambar_jarak_antar_marker(frame, marker_data):
    """
    Menggambar garis antar marker berurutan saja:
    ID 1 - ID 2
    ID 2 - ID 3
    ID 3 - ID 4

    Info jarak tidak ditulis di frame kamera.
    Info jarak dipindahkan ke Panel Tombol.
    """
    pasangan_marker = [
        (1, 2),
        (2, 3),
        (3, 4)
    ]

    for id_a, id_b in pasangan_marker:
        if id_a not in marker_data or id_b not in marker_data:
            continue

        center_a = marker_data[id_a]["center"]
        center_b = marker_data[id_b]["center"]

        cv2.line(
            frame,
            center_a,
            center_b,
            (255, 0, 255),
            2
        )
        # ==========================================
        # TULIS JARAK ANTAR ROBOT
        # ==========================================

        scale_rata = (
            marker_data[id_a]["scale_cm_per_px"] +
            marker_data[id_b]["scale_cm_per_px"]
        ) / 2.0

        dx = center_b[0] - center_a[0]
        dy = center_b[1] - center_a[1]

        jarak_px = math.sqrt(dx * dx + dy * dy)
        jarak_cm = jarak_px * scale_rata

        mid_x = int((center_a[0] + center_b[0]) / 2)
        mid_y = int((center_a[1] + center_b[1]) / 2)

        cv2.putText(
            frame,
            f"{jarak_cm:.1f} cm",
            (mid_x, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 255),
            2
        )


def gambar_marker_aruco(
    frame_tampil,
    marker_id,
    dict_name,
    corner,
    object_bboxes,
    marker_data,
    mode_directional
):
    pts = corner[0].astype(int)

    label_swarm = ambil_label_swarm(marker_id)
    warna_label = ambil_warna_label(marker_id)

    x, y, w, h = cv2.boundingRect(pts)

    object_bboxes.append((x, y, w, h))

    center_x = int(np.mean(pts[:, 0]))
    center_y = int(np.mean(pts[:, 1]))

    marker_width_px = np.linalg.norm(pts[0] - pts[1])
    jarak_cm = hitung_jarak_cm(marker_width_px)

    angle_degree = hitung_sudut_marker(pts)

    scale_cm_per_px = hitung_skala_cm_per_px(pts)

    marker_data[marker_id] = {
        "center": (center_x, center_y),
        "scale_cm_per_px": scale_cm_per_px,
        "jarak_kamera_cm": jarak_cm,
        "angle_degree": angle_degree
    }

    cv2.polylines(frame_tampil, [pts], True, warna_label, 2)
    cv2.rectangle(frame_tampil, (x, y), (x + w, y + h), warna_label, 2)
    cv2.circle(frame_tampil, (center_x, center_y), 6, (0, 0, 255), -1)
    # ==========================================
    # IDENTITAS ROBOT DI FRAME KAMERA
    # ==========================================

    teks_robot = (
        f"ID {marker_id} | "
        f"{label_swarm}"
    )

    cv2.putText(
        frame_tampil,
        teks_robot,
        (x, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        warna_label,
        2
    )

    # Di frame kamera hanya gambar visual marker dan panah arah.
    # Semua info teks deteksi dipindahkan ke Panel Tombol.
    if mode_directional:
        gambar_panah_depan(
        frame_tampil,
        center_x,
        center_y,
        angle_degree
    )

    error_depan = abs(angle_degree - 90)

    if error_depan <= FRONT_TOLERANCE_DEGREE:
        status_robot = "DEPAN"
    else:
        status_robot = f"MIRING {angle_degree:.1f} deg"

    marker_data[marker_id]["status_robot"] = status_robot
    marker_data[marker_id]["dict_name"] = dict_name
    marker_data[marker_id]["label_swarm"] = label_swarm


# ==================================================
# PROSES ARUCO
# ==================================================

def proses_aruco_optimal(
    frame_deteksi_aruco,
    frame_tampil,
    aruco_params,
    dictionary_list,
    preferred_dict_name,
    detector_cache,
    frame_count,
    mode_flip,
    object_bboxes,
    mode_directional,
    mode_jarak_marker
):
    """
    Penting:
    - frame_deteksi_aruco harus frame asli, tidak di-flip.
    - frame_tampil boleh sudah di-flip.
    - Kalau mode_flip ON, koordinat corner dibalik agar gambar cocok dengan tampilan.
    - Jarak antar marker yang ditampilkan hanya 1-2, 2-3, dan 3-4.
    """
    gray = cv2.cvtColor(frame_deteksi_aruco, cv2.COLOR_BGR2GRAY)

    tinggi, lebar = frame_deteksi_aruco.shape[:2]

    ada_marker_target = False
    detected_dict_name = preferred_dict_name
    marker_data = {}

    # 1. Coba dictionary aktif dulu
    dict_id = cari_dictionary_id(dictionary_list, preferred_dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    if preferred_dict_name not in detector_cache:
        detector_cache[preferred_dict_name] = buat_detector(aruco_dict, aruco_params)

    detector = detector_cache[preferred_dict_name]

    corners, ids = deteksi_aruco(gray, aruco_dict, aruco_params, detector)

    if ids is not None and len(ids) > 0:
        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])

            if marker_id not in ARUCO_TARGET_IDS:
                continue

            ada_marker_target = True

            corner_tampil = corners[i]
            if mode_flip:
                corner_tampil = balik_corner_horizontal(corners[i], lebar)

            gambar_marker_aruco(
                frame_tampil,
                marker_id,
                preferred_dict_name,
                corner_tampil,
                object_bboxes,
                marker_data,
                mode_directional
            )

    # 2. Kalau belum ketemu, scan dictionary lain tiap beberapa frame
    if not ada_marker_target and frame_count % AUTO_SCAN_INTERVAL_FRAME == 0:
        for dict_name, dict_id in dictionary_list:
            if dict_name == preferred_dict_name:
                continue

            aruco_dict_scan = cv2.aruco.getPredefinedDictionary(dict_id)

            if dict_name not in detector_cache:
                detector_cache[dict_name] = buat_detector(aruco_dict_scan, aruco_params)

            detector_scan = detector_cache[dict_name]

            corners_scan, ids_scan = deteksi_aruco(
                gray,
                aruco_dict_scan,
                aruco_params,
                detector_scan
            )

            if ids_scan is not None and len(ids_scan) > 0:
                for i, marker_id_arr in enumerate(ids_scan):
                    marker_id = int(marker_id_arr[0])

                    if marker_id not in ARUCO_TARGET_IDS:
                        continue

                    ada_marker_target = True
                    detected_dict_name = dict_name

                    corner_tampil = corners_scan[i]
                    if mode_flip:
                        corner_tampil = balik_corner_horizontal(corners_scan[i], lebar)

                    gambar_marker_aruco(
                        frame_tampil,
                        marker_id,
                        dict_name,
                        corner_tampil,
                        object_bboxes,
                        marker_data,
                        mode_directional
                    )

                    print(f"Dictionary cocok ditemukan: {dict_name} | ID {marker_id}")

            if ada_marker_target:
                break

    if mode_jarak_marker:
        if len(marker_data) >= 2:
            gambar_jarak_antar_marker(frame_tampil, marker_data)

    # Status "marker tidak terdeteksi" tidak ditulis di frame kamera.
    # Status tersebut ditampilkan di Panel Tombol.

    return detected_dict_name, ada_marker_target, marker_data


# ==================================================
# PANEL TOMBOL
# ==================================================

def buat_panel_tombol(
    mode_aruco,
    mode_kuning,
    mode_biru,
    mode_hijau,
    mode_grid,
    mode_mask,
    mode_flip,
    mode_directional,
    mode_jarak_marker,
    fps,
    preferred_dict_name,
    ada_marker,
    marker_data,
    brightness,
    contrast
):
    
    """
    Panel tetap lengkap seperti versi sebelumnya,
    tetapi ukuran frame panel tidak terlalu besar.
    Tulisan diperkecil agar semua info tetap terlihat.

    Ukuran panel: 640 x 430.
    """
    panel = np.zeros((430, 640, 3), dtype=np.uint8)

    # =========================
    # HEADER
    # =========================
    cv2.putText(
        panel,
        "PANEL TOMBOL / INFO DETEKSI",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    cv2.line(panel, (15, 38), (625, 38), (255, 255, 255), 1)

    # =========================
    # INFO MODE DAN TOMBOL
    # =========================
    info = [
        f"FPS realtime              : {fps:.1f}",
        f"Dictionary aktif          : {preferred_dict_name}",
        f"Brightness / Contrast     : {brightness:+d} / {contrast:.2f}",
        f"[a] ArUco ID 1-4          : {'ON' if mode_aruco else 'OFF'}",
        f"[r] Warna KUNING          : {'ON' if mode_kuning else 'OFF'}",
        f"[b] Warna BIRU            : {'ON' if mode_biru else 'OFF'}",
        f"[j] Warna HIJAU           : {'ON' if mode_hijau else 'OFF'}",
        f"[g] Grid                  : {'ON' if mode_grid else 'OFF'}",
        f"[m] Mask warna            : {'ON' if mode_mask else 'OFF'}",
        f"[f] Mirror / flip         : {'ON' if mode_flip else 'OFF'}",
        f"[d] Directional marker    : {'ON' if mode_directional else 'OFF'}",
        f"[k] Jarak leader-follower : {'ON' if mode_jarak_marker else 'OFF'}",
        
    ]

    y = 56
    for text in info:
        cv2.putText(
            panel,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )
        y += 17

    # =========================
    # INFO DETEKSI ARUCO
    # =========================
    cv2.line(panel, (15, 245), (625, 245), (255, 255, 255), 1)

    cv2.putText(
        panel,
        "INFO DETEKSI ARUCO:",
        (18, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1
    )

    y = 285

    if not mode_aruco:
        cv2.putText(
            panel,
            "Deteksi ArUco OFF",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1
        )
        y += 18

    elif not ada_marker or len(marker_data) == 0:
        cv2.putText(
            panel,
            "Marker ArUco ID 1-4 tidak terdeteksi",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1
        )
        y += 18

    else:
        ids_terdeteksi = sorted(marker_data.keys())

        cv2.putText(
            panel,
            f"Marker terdeteksi: {ids_terdeteksi}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0),
            1
        )
        y += 18

        for marker_id in ids_terdeteksi:
            data = marker_data[marker_id]

            label_swarm = data.get("label_swarm", ambil_label_swarm(marker_id))
            dict_name = data.get("dict_name", preferred_dict_name)
            jarak_kamera_cm = data.get("jarak_kamera_cm", 0)
            angle_degree = data.get("angle_degree", 0)
            status_robot = data.get("status_robot", "-")

            if mode_directional:
                arah_text = f"Arah {angle_degree:.1f} deg | {status_robot}"
            else:
                arah_text = "Directional OFF"

            text = (
                f"ID {marker_id} | {label_swarm} | {dict_name} | "
                f"Jarak {jarak_kamera_cm:.1f} cm | "
                f"{arah_text}"
            )

            cv2.putText(
                panel,
                text,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1
            )
            y += 17

    # =========================
    # JARAK ANTAR MARKER
    # =========================
    cv2.line(panel, (15, 365), (625, 365), (255, 255, 255), 1)

    cv2.putText(
        panel,
        "JARAK ANTAR MARKER:",
        (18, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 0, 255),
        1
    )

    if mode_jarak_marker:
        jarak_antar_marker = hitung_jarak_antar_marker_berurutan(marker_data)
    else:
        jarak_antar_marker = []

    if not mode_jarak_marker:
        jarak_text = "Pembacaan jarak OFF"

    elif len(jarak_antar_marker) == 0:
        jarak_text = "Belum cukup marker untuk jarak 1-2, 2-3, atau 3-4"
    else:
        bagian = []
        for item in jarak_antar_marker:
            bagian.append(
                f"{item['id_a']}-{item['id_b']}: {item['jarak_cm']:.1f} cm"
            )
        jarak_text = " | ".join(bagian)

    cv2.putText(
        panel,
        jarak_text,
        (18, 407),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 0, 255),
        1
    )

    return panel


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():
    cek_opencv()

    print("Daftar device video di /dev:")
    os.system("ls -l /dev/video* 2>/dev/null")

    kamera_index = 0
    cap = buka_kamera(kamera_index)

    if cap is None:
        print("Kamera tidak berhasil dibuka.")
        sys.exit(1)

    aruco_params = buat_parameter_aruco()
    dictionary_list = daftar_dictionary_aruco()
    detector_cache = {}

    buat_trackbar_warna()
    buat_trackbar_brightness_contrast()

    mode_aruco = True

    # Deteksi semua warna default ON
    mode_kuning = True
    mode_biru = True
    mode_hijau = True

    mode_grid = True
    mode_mask = False

    # Toggle arah directional
    mode_directional = True

    # Toggle jarak antar marker
    mode_jarak_marker = True

    # Tekan f untuk ON/OFF mirror
    mode_flip = False

    preferred_dict_name = "DICT_4X4_50"

    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    ada_marker = False
    marker_data = {}
    brightness = 0
    contrast = 1.0

    print("\nProgram berjalan.")
    print("Fix utama:")
    print("1. ArUco dideteksi dari frame asli, bukan frame flip.")
    print("2. Jika flip ON, hanya tampilan yang di-flip.")
    print("3. Koordinat ArUco dibalik manual dan urutan corner dikoreksi saat flip ON.")
    print("4. Arah depan robot dihitung dari center ke titik tengah sisi bawah marker.")
    print("5. Info teks deteksi marker dipindahkan dari frame kamera ke Panel Tombol.")
    print("6. Jarak antar marker yang ditampilkan hanya 1-2, 2-3, dan 3-4.")
    print("7. Grid digambar terakhir.")
    print("8. Grid berubah hijau jika terkena bounding box objek.")
    print("9. Brightness dan contrast kamera dapat diatur lewat trackbar Setting Kamera.")
    print("\nTombol:")
    print("a = ON/OFF ArUco")
    print("r = ON/OFF kuning")
    print("b = ON/OFF biru")
    print("j = ON/OFF hijau")
    print("g = ON/OFF grid")
    print("m = ON/OFF mask")
    print("f = ON/OFF mirror kamera")
    print("d = ON/OFF directional")
    print("k = ON/OFF jarak marker")
    print("s = screenshot")
    print("q = keluar")
    print("\nTrackbar Setting Kamera:")
    print("Brightness = kiri lebih gelap, kanan lebih terang")
    print("Contrast   = kiri lebih rendah, kanan lebih tinggi")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Gagal membaca frame.")
                break

            frame_count += 1

            brightness, contrast = ambil_brightness_contrast()
            frame = terapkan_brightness_contrast(frame, brightness, contrast)

            # ==================================================
            # URUTAN PENTING
            # ==================================================

            # 1. Frame asli untuk deteksi ArUco.
            # Jangan di-flip, karena marker ArUco tidak valid kalau dicermin.
            frame_deteksi_aruco = frame.copy()

            # 2. Frame tampilan boleh di-flip.
            if mode_flip:
                frame_tampil = cv2.flip(frame, 1)
            else:
                frame_tampil = frame.copy()

            # 3. Frame deteksi warna mengikuti tampilan.
            frame_deteksi_warna = frame_tampil.copy()

            # 4. List semua bounding box objek pada frame tampilan.
            # Format: (x, y, w, h)
            object_bboxes = []
            ada_marker = False
            marker_data = {}

            current_time = time.time()
            delta_time = current_time - prev_time

            if delta_time > 0:
                fps = 1.0 / delta_time

            prev_time = current_time

            # 5. Deteksi ArUco dari frame asli.
            # Hasil gambar masuk ke frame_tampil.
            # Bounding box ArUco masuk ke object_bboxes.
            if mode_aruco:
                preferred_dict_name, ada_marker, marker_data = proses_aruco_optimal(
            frame_deteksi_aruco,
            frame_tampil,
            aruco_params,
            dictionary_list,
            preferred_dict_name,
            detector_cache,
            frame_count,
            mode_flip,
            object_bboxes,
            mode_directional,
            mode_jarak_marker
            )
                

            # 6. Deteksi warna dari frame tampilan.
            # Bounding box warna masuk ke object_bboxes.
            if frame_count % COLOR_DETECTION_INTERVAL_FRAME == 0:
                if mode_kuning or mode_biru or mode_hijau:
                    deteksi_semua_warna(
                        frame_deteksi_warna,
                        frame_tampil,
                        mode_kuning,
                        mode_biru,
                        mode_hijau,
                        mode_mask,
                        object_bboxes
                    )

            # 7. Grid digambar terakhir.
            # Kotak grid yang terkena object_bboxes berubah hijau.
            if mode_grid:
                gambar_grid(frame_tampil, GRID_SIZE, object_bboxes)

            # 8. FPS digambar paling akhir.
            tampilkan_fps(frame_tampil, fps)

            panel = buat_panel_tombol(
            mode_aruco,
            mode_kuning,
            mode_biru,
            mode_hijau,
            mode_grid,
            mode_mask,
            mode_flip,
            mode_directional,
            mode_jarak_marker,
            fps,
            preferred_dict_name,
            ada_marker,
            marker_data,
            brightness,
            contrast
            )

            cv2.imshow("Frame Kamera Robot", frame_tampil)
            cv2.imshow("Panel Tombol", panel)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("a"):
                mode_aruco = not mode_aruco
                print("Mode ArUco:", "ON" if mode_aruco else "OFF")

            elif key == ord("r"):
                mode_kuning = not mode_kuning
                print("Mode kuning:", "ON" if mode_kuning else "OFF")

            elif key == ord("b"):
                mode_biru = not mode_biru
                print("Mode Biru:", "ON" if mode_biru else "OFF")

            elif key == ord("j"):
                mode_hijau = not mode_hijau
                print("Mode Hijau:", "ON" if mode_hijau else "OFF")

            elif key == ord("g"):
                mode_grid = not mode_grid
                print("Mode Grid:", "ON" if mode_grid else "OFF")

            elif key == ord("m"):
                mode_mask = not mode_mask
                print("Mode Mask:", "ON" if mode_mask else "OFF")

                if not mode_mask:
                    try:
                        cv2.destroyWindow("Mask Kuning")
                        cv2.destroyWindow("Mask Biru")
                        cv2.destroyWindow("Mask Hijau")
                    except Exception:
                        pass
            elif key == ord("d"):
                mode_directional = not mode_directional
                print("Mode Directional:", "ON" if mode_directional else "OFF")

            elif key == ord("k"):
                mode_jarak_marker = not mode_jarak_marker
                print("Mode Jarak Marker:", "ON" if mode_jarak_marker else "OFF")

            elif key == ord("f"):
                mode_flip = not mode_flip
                print("Mode Flip:", "ON" if mode_flip else "OFF")

            elif key == ord("s"):
                nama_file = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(nama_file, frame_tampil)
                print(f"Screenshot disimpan: {nama_file}")

    except KeyboardInterrupt:
        print("\nProgram dihentikan dengan Ctrl+C.")

    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")


if __name__ == "__main__":
    main()
