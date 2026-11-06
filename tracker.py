import os
import sys
import time
import argparse

import cv2
import serial
from serial.tools import list_ports


# Serial device path; leave as None to auto-detect a connected Arduino.
SERIAL_PORT = "COM10"
# USB serial baud rate; must match the Arduino sketch.
BAUD_RATE = 115200
# When True, open serial and pan the servo to center the face. False: preview only.
USE_SERVO = True
# Camera device index.
CAMERA_INDEX = 0
# Requested capture width in pixels.
CAMERA_WIDTH = 320
# Requested capture height in pixels.
CAMERA_HEIGHT = 240
# Digital-zoom crop size drawn in the preview window.
CROP_WIDTH = 320
CROP_HEIGHT = 240
# Run Haar detection every N processed frames; servo still updates every frame.
DETECT_EVERY_N_FRAMES = 1
# Proportional pan gain in degrees per pixel of horizontal face error.
KP = 0.10
# Horizontal pixel error within which the servo holds its current position.
DEAD_ZONE_PX = 8
# Largest commanded angle change per processed frame, in degrees.
MAX_STEP_DEG = 8.0
# Exponential-smoothing factor applied to the target angle, from 0 to 1.
SMOOTHING = 0.9
# This mounting direction was confirmed by observing the camera turn away.
INVERT_DIRECTION = True
# Minimum allowed pan angle in degrees.
MIN_ANGLE = 20
# Maximum allowed pan angle in degrees.
MAX_ANGLE = 160
# When True, print servo debug values about once every 10 frames.
DEBUG_PRINT = False
CAMERA_READ_FAILURE_LIMIT = 10


def available_port_names(ports):
    if not ports:
        return "none detected"
    return ", ".join(
        f"{port.device} ({port.description})" for port in ports
    )


def find_serial_port(serial_port=SERIAL_PORT):
    if serial_port:
        return serial_port

    ports = list(list_ports.comports())
    available = available_port_names(ports)
    if not ports:
        raise RuntimeError(
            "No serial ports were detected. Connect the Arduino and check its "
            f"USB cable/driver. Available ports: {available}."
        )

    arduino_keywords = (
        "arduino",
        "ch340",
        "usb serial",
        "usb-serial",
        "cp210",
        "ftdi",
    )
    matches = [
        port.device
        for port in ports
        if any(
            keyword in " ".join(
                (port.description or "", port.manufacturer or "", port.hwid or "")
            ).lower()
            for keyword in arduino_keywords
        )
    ]

    if len(matches) >= 1:
        return matches[0]
    return ports[0].device


def open_camera(camera_index):
    if os.name == "nt":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera index {camera_index}."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return capture


def detect_largest_face(gray_frame, face_cascade):
    """Detect faces on the full frame and return the largest box, or None."""
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.15,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: int(face[2]) * int(face[3]))


def crop_origin(center_x, center_y, frame_width, frame_height):
    crop_width = min(CROP_WIDTH, frame_width)
    crop_height = min(CROP_HEIGHT, frame_height)
    origin_x = int(center_x - crop_width // 2)
    origin_y = int(center_y - crop_height // 2)
    origin_x = max(0, min(origin_x, frame_width - crop_width))
    origin_y = max(0, min(origin_y, frame_height - crop_height))
    return origin_x, origin_y, crop_width, crop_height


def send_servo_angle(serial_connection, angle, write_error_printed):
    try:
        serial_connection.write("{0:d}\n".format(int(angle)).encode("ascii"))
        serial_connection.flush()
        return False
    except (serial.SerialException, OSError) as error:
        if not write_error_printed:
            print(
                "Serial write failed ({0}); continuing without crashing.".format(
                    error
                ),
                file=sys.stderr,
            )
        return True


def update_pan_angle(current_angle, error_px, invert_direction=INVERT_DIRECTION):
    if abs(error_px) <= DEAD_ZONE_PX:
        return current_angle

    excess = abs(error_px) - DEAD_ZONE_PX
    taper = 0.5 + 0.5 * min(1.0, excess / 80.0)
    delta = KP * error_px * taper
    if invert_direction:
        delta = -delta
    delta = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, delta))
    if abs(delta) < 1.0:
        delta = 1.0 if delta >= 0.0 else -1.0

    updated_angle = current_angle + SMOOTHING * delta
    updated_angle = max(MIN_ANGLE, min(MAX_ANGLE, updated_angle))
    if round(updated_angle) == round(current_angle):
        updated_angle = max(
            MIN_ANGLE,
            min(MAX_ANGLE, current_angle + (1.0 if delta > 0 else -1.0)),
        )
    return updated_angle


def run_tracker(
    camera_index=CAMERA_INDEX,
    serial_port=SERIAL_PORT,
    invert_direction=INVERT_DIRECTION,
    debug_print=DEBUG_PRINT,
):
    serial_connection = None
    camera = None
    write_error_printed = False

    try:
        if USE_SERVO:
            port = find_serial_port(serial_port)
            try:
                serial_connection = serial.Serial(
                    port,
                    BAUD_RATE,
                    timeout=1,
                    write_timeout=1,
                    dsrdtr=False,
                )
            except serial.SerialException as error:
                raise RuntimeError(
                    "Could not open serial port {0}. Check that the port is "
                    "correct and that the Arduino Serial Monitor is closed.".format(
                        port
                    )
                ) from error
            serial_connection.dtr = True
            time.sleep(2)
            serial_connection.reset_input_buffer()
            serial_connection.reset_output_buffer()
            send_servo_angle(serial_connection, 90, False)

        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise RuntimeError(
                "Could not load the Haar cascade: {0}".format(cascade_path)
            )

        camera = open_camera(camera_index)
        current_angle = 90.0
        last_sent_angle = 90 if USE_SERVO else None
        last_face = None
        frame_count = 0
        consecutive_read_failures = 0
        face_center_x = None
        error_px = None

        cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Tracker", 640, 480)

        while True:
            success, frame = camera.read()
            if not success:
                consecutive_read_failures += 1
                if consecutive_read_failures >= CAMERA_READ_FAILURE_LIMIT:
                    print(
                        "Camera stopped delivering frames. Close other camera "
                        "apps and check the webcam connection, then restart.",
                        file=sys.stderr,
                    )
                    break
                time.sleep(0.05)
                continue
            consecutive_read_failures = 0

            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2
            frame_count += 1

            if (frame_count - 1) % DETECT_EVERY_N_FRAMES == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.equalizeHist(gray_frame)
                last_face = detect_largest_face(gray_frame, face_cascade)

            if last_face is not None:
                x, y, width, height = (int(value) for value in last_face)
                face_center_x = x + width // 2
                face_center_y = y + height // 2
                error_px = face_center_x - frame_center_x
                current_angle = update_pan_angle(
                    current_angle,
                    error_px,
                    invert_direction,
                )
            else:
                face_center_x = None
                error_px = None

            if USE_SERVO:
                crop_x, crop_y, crop_w, crop_h = crop_origin(
                    frame_center_x,
                    frame_center_y,
                    frame_width,
                    frame_height,
                )
            elif last_face is not None:
                crop_x, crop_y, crop_w, crop_h = crop_origin(
                    face_center_x,
                    face_center_y,
                    frame_width,
                    frame_height,
                )
            else:
                crop_x, crop_y, crop_w, crop_h = crop_origin(
                    frame_center_x,
                    frame_center_y,
                    frame_width,
                    frame_height,
                )

            preview = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()
            preview = cv2.resize(preview, (640, 480), interpolation=cv2.INTER_LINEAR)
            scale_x = 640.0 / max(1, crop_w)
            scale_y = 480.0 / max(1, crop_h)

            if last_face is not None:
                x, y, width, height = (int(value) for value in last_face)
                box_x1 = int((x - crop_x) * scale_x)
                box_y1 = int((y - crop_y) * scale_y)
                box_x2 = int((x + width - crop_x) * scale_x)
                box_y2 = int((y + height - crop_y) * scale_y)
                cv2.rectangle(
                    preview,
                    (box_x1, box_y1),
                    (box_x2, box_y2),
                    (0, 255, 0),
                    2,
                )

            center_x = preview.shape[1] // 2
            center_y = preview.shape[0] // 2
            cv2.line(
                preview,
                (center_x, center_y - 20),
                (center_x, center_y + 20),
                (255, 255, 255),
                1,
            )
            cv2.line(
                preview,
                (center_x - 20, center_y),
                (center_x + 20, center_y),
                (255, 255, 255),
                1,
            )
            cv2.circle(preview, (center_x, center_y), 4, (255, 255, 255), 1)

            commanded_angle = int(round(current_angle))
            if USE_SERVO and serial_connection is not None:
                if commanded_angle != last_sent_angle:
                    if send_servo_angle(
                        serial_connection,
                        commanded_angle,
                        write_error_printed,
                    ):
                        write_error_printed = True
                    else:
                        last_sent_angle = commanded_angle

            if debug_print and frame_count % 10 == 0:
                print(
                    face_center_x,
                    error_px,
                    current_angle,
                    last_sent_angle,
                )

            cv2.putText(
                preview,
                "Angle: {0}".format(commanded_angle),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                "TRACKING" if last_face is not None else "NO FACE",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if last_face is not None else (0, 0, 255),
                2,
            )
            offset_text = (
                "X offset: --"
                if error_px is None
                else "X offset: {0}px".format(error_px)
            )
            cv2.putText(
                preview,
                offset_text,
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            direction_label = "REVERSED" if invert_direction else "NORMAL"
            cv2.putText(
                preview,
                "Pan: {0} | R switch | S save | Esc quit".format(direction_label),
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Face Tracker", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                invert_direction = not invert_direction
                print(
                    "Pan direction: {0}".format(
                        "reversed" if invert_direction else "normal"
                    )
                )
            if key == ord("s") and last_face is not None:
                cv2.imwrite("only_face{0}.png".format(frame_count), preview)
    finally:
        if camera is not None:
            camera.release()
        if serial_connection is not None and serial_connection.is_open:
            serial_connection.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Track a face with a pan servo.")
    parser.add_argument("--port", default=SERIAL_PORT, help="Arduino serial port.")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera device index (default: {0}).".format(CAMERA_INDEX),
    )
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument(
        "--invert-direction",
        dest="invert_direction",
        action="store_true",
        help="Reverse the default pan direction.",
    )
    direction.add_argument(
        "--normal-direction",
        dest="invert_direction",
        action="store_false",
        help="Use the original pan direction if the reversed direction moves away.",
    )
    parser.set_defaults(invert_direction=INVERT_DIRECTION)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=DEBUG_PRINT,
        help="Print face error and servo angle while tracking.",
    )
    args = parser.parse_args()
    try:
        run_tracker(
            camera_index=args.camera,
            serial_port=args.port,
            invert_direction=args.invert_direction,
            debug_print=args.debug,
        )
    except (RuntimeError, serial.SerialException, OSError) as error:
        print("Error: {0}".format(error), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
