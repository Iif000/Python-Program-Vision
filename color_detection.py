import cv2
import numpy as np

# ===============================
# Inisialisasi Kamera
# ===============================
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Tidak dapat membuka kamera!")
    exit()

# Set resolusi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Resolusi:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# ===============================
# HSV Range Warna Kuning (STABIL)
# ===============================
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Kernel untuk filtering
kernel = np.ones((5, 5), np.uint8)

# ===============================
# Loop Kamera
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    # Flip opsional (kalau kamera terbalik)
    # frame = cv2.flip(frame, 1)

    # Convert ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask warna kuning
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # ===============================
    # Noise Reduction (WAJIB)
    # ===============================
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Ambil objek hasil masking
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # ===============================
    # Deteksi Kontur
    # ===============================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 500:  # filter objek kecil
            x, y, w, h = cv2.boundingRect(contour)

            # Gambar bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ===============================
            # Hitung centroid (titik tengah)
            # ===============================
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Gambar titik tengah
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                # Tampilkan koordinat
                cv2.putText(frame, f"({cx},{cy})",
                            (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 2)

    # ===============================
    # Display
    # ===============================
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Exit dengan tombol Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# Release Resource
# ===============================
cap.release()
cv2.destroyAllWindows()