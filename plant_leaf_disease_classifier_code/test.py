import numpy as np
import serial
import cv2
import time

ser = serial.Serial()
ser.port = '/dev/cu.usbmodem1101'
ser.baudrate = 115200
ser.timeout = 1
ser.open()
ser.reset_input_buffer()

cam = cv2.VideoCapture(1)
if not cam.isOpened():
    raise RuntimeError("Failed to open camera")
time.sleep(0.3)

def serial_readline(obj):
    return obj.readline().decode("utf-8", errors="ignore").strip()

def center_square_crop(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0+side, x0:x0+side]

try:
    while True:
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        line = serial_readline(ser)

        if line == "<cam-read>":
            ret, frame = cam.read()
            if not ret:
                continue

            cropped = center_square_crop(frame)
            resized = cv2.resize(cropped, (96, 96), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            ser.write(gray.astype(np.uint8).tobytes())

            best_label = "Unknown"
            while True:
                out = serial_readline(ser)
                if out.startswith("BEST:"):
                    best_label = out.split()[1]
                    break

            display = cropped.copy()
            cv2.putText(
                display,
                f"Class: {best_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow("Classification", display)

finally:
    cam.release()
    cv2.destroyAllWindows()
    try:
        ser.close()
    except:
        pass
