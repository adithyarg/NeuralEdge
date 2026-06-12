"""
thermal_detect.py — Send thermal images to FPGA, display human detection result.

Works with any grayscale image source (thermal camera, saved .jpg/.png files).
Sends 1024 bytes (32x32 patch) to FPGA, receives 1 byte:
    0 = no human detected
    1 = human detected

Dependencies:
    pip install opencv-python pyserial numpy

Usage:
    # Live webcam (simulated thermal — for testing without a real thermal camera)
    python host/thermal_detect.py --port COM15

    # Single image file
    python host/thermal_detect.py --port COM15 --image path/to/thermal.jpg

    # Folder of images (batch test)
    python host/thermal_detect.py --port COM15 --folder path/to/images/

Controls (live mode):
    SPACE  — run inference on current frame
    Q      — quit
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import serial

PATCH_SIZE = 32   # must match accelerator L1_IN = 1024 = 32*32
LABELS     = {0: "No Human", 1: "HUMAN DETECTED"}
COLORS     = {0: (0, 200, 0), 1: (0, 0, 255)}


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Crop central square, convert to grayscale, resize to 32x32.
    Returns uint8 array shape (1024,).
    For real thermal input the frame is already grayscale — colour conversion
    is a no-op on single-channel images.
    """
    h, w  = frame.shape[:2]
    side  = int(min(h, w) * 0.6)
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    crop  = frame[y0:y0+side, x0:x0+side]
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    resized = cv2.resize(gray, (PATCH_SIZE, PATCH_SIZE),
                         interpolation=cv2.INTER_AREA)
    return resized.flatten().astype(np.uint8)


def infer_fpga(ser: serial.Serial, pixels: np.ndarray) -> tuple:
    """
    Send 1024 bytes to FPGA, receive 1 byte result.
    Returns (label_int, total_ms, fpga_ms).
    """
    n_bytes    = len(pixels)                              # 1024
    uart_tx_ms = (n_bytes * 10 / ser.baudrate) * 1000    # ~89 ms @ 115200
    uart_rx_ms = (1       * 10 / ser.baudrate) * 1000

    ser.reset_input_buffer()
    t0   = time.perf_counter()
    ser.write(pixels.tobytes())
    resp = ser.read(1)
    total_ms = (time.perf_counter() - t0) * 1000

    if not resp:
        raise TimeoutError("No response — check FPGA is programmed and port is correct")

    fpga_ms = max(0.0, total_ms - uart_tx_ms - uart_rx_ms)
    return resp[0], total_ms, fpga_ms


def run_live(ser, cam_idx):
    """Live webcam mode."""
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_idx}")
        ser.close()
        sys.exit(1)

    print("Hold a thermal image / person in front of the camera.")
    print("SPACE = infer   Q = quit\n")
    overlay = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pixels = preprocess(frame)

        # Draw crop guide
        h, w  = frame.shape[:2]
        side  = int(min(h, w) * 0.6)
        y0    = (h - side) // 2
        x0    = (w - side) // 2
        cv2.rectangle(frame, (x0, y0), (x0+side, y0+side), (0, 200, 0), 2)
        cv2.putText(frame, "SPACE=infer  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        if overlay:
            color = COLORS[last_label] if 'last_label' in dir() else (0, 200, 0)
            cv2.putText(frame, overlay,
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 32x32 preview
        preview = cv2.resize(pixels.reshape(PATCH_SIZE, PATCH_SIZE), (256, 256),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Camera", frame)
        cv2.imshow(f"{PATCH_SIZE}x{PATCH_SIZE} sent to FPGA",
                   cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            try:
                label, total_ms, fpga_ms = infer_fpga(ser, pixels)
                last_label = label
                overlay = (f"{LABELS.get(label, f'class {label}')}  "
                           f"(FPGA: {fpga_ms:.1f}ms  total: {total_ms:.0f}ms)")
                print(overlay)
            except TimeoutError as e:
                overlay = "Timeout!"
                print(e)

    cap.release()
    cv2.destroyAllWindows()


def run_image(ser, image_path):
    """Single image inference."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot read {image_path}")
        return

    pixels = preprocess(frame)
    label, total_ms, fpga_ms = infer_fpga(ser, pixels)
    result_str = LABELS.get(label, f"class {label}")
    print(f"{os.path.basename(image_path):30s}  →  {result_str}  "
          f"(FPGA: {fpga_ms:.1f}ms  total: {total_ms:.0f}ms)")

    # Show result
    h, w = frame.shape[:2]
    color = COLORS.get(label, (255, 255, 255))
    cv2.putText(frame, result_str, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder(ser, folder_path):
    """Batch inference on all images in a folder."""
    exts    = {".jpg", ".jpeg", ".png", ".bmp"}
    files   = sorted([f for f in os.listdir(folder_path)
                      if os.path.splitext(f)[1].lower() in exts])
    if not files:
        print(f"No images found in {folder_path}")
        return

    correct = total = 0
    print(f"Running batch inference on {len(files)} images...\n")

    for fname in files:
        path   = os.path.join(folder_path, fname)
        frame  = cv2.imread(path)
        if frame is None:
            continue
        pixels = preprocess(frame)
        try:
            label, total_ms, fpga_ms = infer_fpga(ser, pixels)
            result_str = LABELS.get(label, f"class {label}")
            print(f"  {fname:30s}  →  {result_str}  ({total_ms:.0f}ms)")
            total += 1
        except TimeoutError as e:
            print(f"  {fname:30s}  →  TIMEOUT: {e}")

    print(f"\nProcessed {total}/{len(files)} images.")


def main():
    ap = argparse.ArgumentParser(
        description="Thermal human detection via FPGA accelerator")
    ap.add_argument("--port",    default="/dev/ttyUSB0",
                    help="Serial port (e.g. COM15 or /dev/ttyUSB0)")
    ap.add_argument("--baud",    type=int, default=115200)
    ap.add_argument("--cam",     type=int, default=0,
                    help="Camera index for live mode")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--image",   default=None,
                    help="Path to a single image file")
    ap.add_argument("--folder",  default=None,
                    help="Path to folder of images for batch inference")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud,
                            timeout=args.timeout,
                            write_timeout=args.timeout)
        time.sleep(0.5)
        print(f"Opened {args.port} at {args.baud} baud\n")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)

    try:
        if args.image:
            run_image(ser, args.image)
        elif args.folder:
            run_folder(ser, args.folder)
        else:
            run_live(ser, args.cam)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
