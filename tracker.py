import argparse
import math
import os
import sys
import time

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
# Reject frames that are too dark or flat to contain reliable face detail.
MIN_FRAME_BRIGHTNESS = 18.0
MIN_FRAME_CONTRAST = 10.0
MIN_FACE_CONTRAST = 12.0
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (20, 20)
# Rotate frontal-cascade search images to handle mild head roll.
FRONTAL_SEARCH_ANGLES = (0, -15, 15)
# Eye validation must find a plausible pair for frontal detections.
EYE_SEARCH_ANGLES = (0, -15, 15)
EYE_MIN_NEIGHBORS = 4
EYE_MIN_SIZE = (6, 6)
# Require repeatable face detections to reject one-frame cascade false positives.
FACE_CONFIRMATION_FRAMES = 2
# Proportional pan gain in degrees per pixel of horizontal face error.
KP = 0.08
# Horizontal pixel error within which the servo holds its current position.
DEAD_ZONE_PX = 8
# Largest commanded angle change per processed frame, in degrees.
MAX_STEP_DEG = 3.0
# Filter factor for measured face movement; lower values smooth more.
ERROR_SMOOTHING = 0.65
# Fraction of each pan correction applied per frame.
SMOOTHING = 0.65
# Face motion at or above this many pixels per frame gets the full speed boost.
VELOCITY_SCALE_PX_PER_FRAME = 6.0
# Moving-away gain boost is limited to avoid overshooting the face.
MAX_VELOCITY_BOOST = 0.5
# Filter factor for face speed; a high value also releases stale speed quickly.
VELOCITY_SMOOTHING = 0.7
VELOCITY_DEAD_ZONE_PX_PER_FRAME = 1.5
# Start searching after this long without a detected face.
NO_FACE_SCAN_DELAY_SECONDS = 10.0
# Pause this long after an unsuccessful scan before searching again.
SCAN_RETRY_DELAY_SECONDS = 5 * 60
# Servo target speed while searching, in degrees per second.
SCAN_SPEED_DEG_PER_SECOND = 45.0
SCAN_CYCLES = 2
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


def detect_largest_face(gray_frame, face_cascades, eye_cascades=None):
    """Detect a well-lit frontal or profile face, including mild head roll."""
    if eye_cascades is None:
        eye_cascades = load_eye_cascades()

    brightness, contrast = cv2.meanStdDev(gray_frame)
    if (
        float(brightness[0][0]) < MIN_FRAME_BRIGHTNESS
        or float(contrast[0][0]) < MIN_FRAME_CONTRAST
    ):
        return None

    candidates = []
    frame_width = gray_frame.shape[1]
    frame_height = gray_frame.shape[0]

    for search_phase in ("frontal", "tilted", "profile"):
        phase_candidate_start = len(candidates)
        for cascade, flipped, is_frontal in face_cascades:
            if search_phase == "profile" and is_frontal:
                continue
            if search_phase != "profile" and not is_frontal:
                continue
            if search_phase == "frontal":
                angles = (0,)
            elif search_phase == "tilted":
                angles = FRONTAL_SEARCH_ANGLES[1:]
            else:
                angles = (0,)

            base_frame = cv2.flip(gray_frame, 1) if flipped else gray_frame
            for angle in angles:
                if angle:
                    matrix = cv2.getRotationMatrix2D(
                        (frame_width / 2.0, frame_height / 2.0),
                        angle,
                        1.0,
                    )
                    search_frame = cv2.warpAffine(
                        base_frame,
                        matrix,
                        (frame_width, frame_height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    inverse_matrix = cv2.invertAffineTransform(matrix)
                else:
                    search_frame = base_frame
                    inverse_matrix = None

                equalized_frame = cv2.equalizeHist(search_frame)
                faces = cascade.detectMultiScale(
                    equalized_frame,
                    scaleFactor=1.1,
                    minNeighbors=FACE_MIN_NEIGHBORS,
                    minSize=FACE_MIN_SIZE,
                )
                for x, y, width, height in faces:
                    if inverse_matrix is not None:
                        corners = (
                            (x, y),
                            (x + width, y),
                            (x, y + height),
                            (x + width, y + height),
                        )
                        mapped_corners = [
                            (
                                inverse_matrix[0, 0] * corner_x
                                + inverse_matrix[0, 1] * corner_y
                                + inverse_matrix[0, 2],
                                inverse_matrix[1, 0] * corner_x
                                + inverse_matrix[1, 1] * corner_y
                                + inverse_matrix[1, 2],
                            )
                            for corner_x, corner_y in corners
                        ]
                        left = max(
                            0,
                            int(
                                math.floor(
                                    min(point[0] for point in mapped_corners)
                                )
                            ),
                        )
                        top = max(
                            0,
                            int(
                                math.floor(
                                    min(point[1] for point in mapped_corners)
                                )
                            ),
                        )
                        right = min(
                            frame_width,
                            int(
                                math.ceil(
                                    max(point[0] for point in mapped_corners)
                                )
                            ),
                        )
                        bottom = min(
                            frame_height,
                            int(
                                math.ceil(
                                    max(point[1] for point in mapped_corners)
                                )
                            ),
                        )
                        x, y = left, top
                        width, height = right - left, bottom - top

                    if flipped:
                        x = frame_width - x - width
                    x = max(0, int(x))
                    y = max(0, int(y))
                    width = min(int(width), frame_width - x)
                    height = min(int(height), frame_height - y)
                    if width <= 0 or height <= 0:
                        continue

                    face_region = gray_frame[y:y + height, x:x + width]
                    face_brightness, face_contrast = cv2.meanStdDev(
                        face_region
                    )
                    if (
                        float(face_brightness[0][0]) < MIN_FRAME_BRIGHTNESS
                        or float(face_contrast[0][0]) < MIN_FACE_CONTRAST
                    ):
                        continue
                    candidates.append((x, y, width, height, is_frontal))

        phase_candidates = sorted(
            candidates[phase_candidate_start:],
            key=lambda face: face[2] * face[3],
            reverse=True,
        )
        for x, y, width, height, is_frontal in phase_candidates:
            face_region = gray_frame[y:y + height, x:x + width]
            if has_face_eye_structure(face_region, eye_cascades, is_frontal):
                return x, y, width, height
    return None


def has_face_eye_structure(face_region, eye_cascades, require_eye_pair):
    """Reject Haar face boxes without eyes in a plausible face layout."""
    region_height, region_width = face_region.shape[:2]
    eye_region = face_region[: max(1, int(region_height * 0.72)), :]
    eye_detections = []

    for angle in EYE_SEARCH_ANGLES:
        if angle:
            matrix = cv2.getRotationMatrix2D(
                (region_width / 2.0, region_height * 0.36),
                angle,
                1.0,
            )
            search_region = cv2.warpAffine(
                eye_region,
                matrix,
                (region_width, eye_region.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            inverse_matrix = cv2.invertAffineTransform(matrix)
        else:
            search_region = eye_region
            inverse_matrix = None

        equalized_region = cv2.equalizeHist(search_region)
        for cascade in eye_cascades:
            for eye_x, eye_y, eye_width, eye_height in cascade.detectMultiScale(
                equalized_region,
                scaleFactor=1.1,
                minNeighbors=EYE_MIN_NEIGHBORS,
                minSize=EYE_MIN_SIZE,
            ):
                center_x = eye_x + eye_width / 2.0
                center_y = eye_y + eye_height / 2.0
                if inverse_matrix is not None:
                    original_center_x = (
                        inverse_matrix[0, 0] * center_x
                        + inverse_matrix[0, 1] * center_y
                        + inverse_matrix[0, 2]
                    )
                    original_center_y = (
                        inverse_matrix[1, 0] * center_x
                        + inverse_matrix[1, 1] * center_y
                        + inverse_matrix[1, 2]
                    )
                    center_x = original_center_x
                    center_y = original_center_y
                if (
                    0.12 * region_width <= center_x <= 0.88 * region_width
                    and center_y <= 0.62 * region_height
                ):
                    is_duplicate = any(
                        math.hypot(center_x - existing_x, center_y - existing_y)
                        < 0.12 * region_width
                        for existing_x, existing_y in eye_detections
                    )
                    if not is_duplicate:
                        eye_detections.append((center_x, center_y))

    return has_plausible_eye_layout(
        eye_detections,
        region_width,
        region_height,
        require_eye_pair,
    )


def has_plausible_eye_layout(
    eye_detections,
    region_width,
    region_height,
    require_eye_pair,
):
    """Require one profile eye or a symmetric pair for a frontal face."""
    if not eye_detections:
        return False
    if not require_eye_pair:
        return True
    for first_index, first_eye in enumerate(eye_detections):
        for second_eye in eye_detections[first_index + 1:]:
            horizontal_distance = abs(first_eye[0] - second_eye[0])
            vertical_distance = abs(first_eye[1] - second_eye[1])
            if (
                0.18 * region_width <= horizontal_distance <= 0.75 * region_width
                and vertical_distance <= 0.28 * region_height
            ):
                return True
    return False


def is_same_face_detection(previous_face, current_face):
    """Check that consecutive candidate boxes plausibly show the same face."""
    previous_x, previous_y, previous_width, previous_height = previous_face
    current_x, current_y, current_width, current_height = current_face
    previous_center = (
        previous_x + previous_width / 2.0,
        previous_y + previous_height / 2.0,
    )
    current_center = (
        current_x + current_width / 2.0,
        current_y + current_height / 2.0,
    )
    maximum_size = max(
        previous_width,
        previous_height,
        current_width,
        current_height,
    )
    center_distance = math.hypot(
        previous_center[0] - current_center[0],
        previous_center[1] - current_center[1],
    )
    size_ratio = max(
        previous_width / max(1.0, current_width),
        current_width / max(1.0, previous_width),
        previous_height / max(1.0, current_height),
        current_height / max(1.0, previous_height),
    )
    return center_distance <= maximum_size * 1.25 and size_ratio <= 2.0


def load_face_cascades():
    """Load frontal and mirrored profile cascades provided by OpenCV."""
    cascade_directory = cv2.data.haarcascades
    cascade_files = (
        ("haarcascade_frontalface_default.xml", False, True),
        ("haarcascade_frontalface_alt2.xml", False, True),
        ("haarcascade_profileface.xml", False, False),
        ("haarcascade_profileface.xml", True, False),
    )
    face_cascades = []
    for filename, flipped, is_frontal in cascade_files:
        path = os.path.join(cascade_directory, filename)
        cascade = cv2.CascadeClassifier(path)
        if not cascade.empty():
            face_cascades.append((cascade, flipped, is_frontal))

    if not face_cascades:
        raise RuntimeError(
            "Could not load any frontal or profile face cascades from "
            "{0}.".format(cascade_directory)
        )
    return face_cascades


def load_eye_cascades():
    """Load eye classifiers used to verify Haar face candidates."""
    cascade_directory = cv2.data.haarcascades
    eye_files = (
        "haarcascade_eye.xml",
        "haarcascade_eye_tree_eyeglasses.xml",
    )
    eye_cascades = []
    for filename in eye_files:
        cascade = cv2.CascadeClassifier(
            os.path.join(cascade_directory, filename)
        )
        if not cascade.empty():
            eye_cascades.append(cascade)

    if not eye_cascades:
        raise RuntimeError(
            "Could not load an OpenCV eye cascade from {0}.".format(
                cascade_directory
            )
        )
    return eye_cascades


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


def update_pan_angle(
    current_angle,
    error_px,
    invert_direction=INVERT_DIRECTION,
    face_velocity_px_per_frame=0.0,
):
    if abs(error_px) <= DEAD_ZONE_PX:
        return current_angle

    velocity = face_velocity_px_per_frame
    if abs(velocity) < VELOCITY_DEAD_ZONE_PX_PER_FRAME:
        velocity = 0.0
    velocity_ratio = min(abs(velocity) / VELOCITY_SCALE_PX_PER_FRAME, 1.0)
    if error_px * velocity > 0:
        response_scale = 1.0 + MAX_VELOCITY_BOOST * velocity_ratio
    elif error_px * velocity < 0:
        response_scale = 1.0 - 0.5 * velocity_ratio
    else:
        response_scale = 1.0
    excess = abs(error_px) - DEAD_ZONE_PX
    taper = 0.5 + 0.5 * min(1.0, excess / 80.0)
    delta = KP * error_px * taper * response_scale
    if invert_direction:
        delta = -delta
    max_step = MAX_STEP_DEG * response_scale
    delta = max(-max_step, min(max_step, delta))

    updated_angle = current_angle + SMOOTHING * delta
    updated_angle = max(MIN_ANGLE, min(MAX_ANGLE, updated_angle))
    return updated_angle


def smooth_face_error(previous_error, measured_error):
    """Filter detection jitter while retaining gradual face movement."""
    if previous_error is None:
        return float(measured_error)
    return previous_error + ERROR_SMOOTHING * (measured_error - previous_error)


def advance_scan(
    current_angle,
    direction,
    completed_cycles,
    reached_start,
    elapsed_seconds,
):
    remaining = max(0.0, elapsed_seconds) * SCAN_SPEED_DEG_PER_SECOND
    while remaining > 0 and completed_cycles < SCAN_CYCLES:
        endpoint = MAX_ANGLE if direction > 0 else MIN_ANGLE
        distance = abs(endpoint - current_angle)
        if remaining < distance:
            current_angle += direction * remaining
            remaining = 0.0
            continue

        current_angle = float(endpoint)
        remaining -= distance
        if direction < 0:
            if reached_start:
                completed_cycles += 1
            else:
                reached_start = True
        direction *= -1

    return current_angle, direction, completed_cycles, reached_start


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

        face_cascades = load_face_cascades()

        camera = open_camera(camera_index)
        current_angle = 90.0
        last_sent_angle = 90 if USE_SERVO else None
        last_face = None
        pending_face = None
        pending_face_count = 0
        frame_count = 0
        consecutive_read_failures = 0
        face_center_x = None
        error_px = None
        filtered_error_px = None
        face_velocity_px_per_frame = 0.0
        last_face_seen_at = time.monotonic()
        scan_active = False
        scan_wait_until = None
        scan_direction = -1
        scan_completed_cycles = 0
        scan_reached_start = False
        last_scan_update = last_face_seen_at

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
            now = time.monotonic()

            if (frame_count - 1) % DETECT_EVERY_N_FRAMES == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidate_face = detect_largest_face(gray_frame, face_cascades)
                if candidate_face is None:
                    pending_face = None
                    pending_face_count = 0
                    last_face = None
                elif (
                    pending_face is not None
                    and is_same_face_detection(pending_face, candidate_face)
                ):
                    pending_face = candidate_face
                    pending_face_count += 1
                    if pending_face_count >= FACE_CONFIRMATION_FRAMES:
                        last_face = candidate_face
                else:
                    pending_face = candidate_face
                    pending_face_count = 1
                    last_face = None

            if last_face is not None:
                last_face_seen_at = now
                scan_active = False
                scan_wait_until = None
                scan_completed_cycles = 0
                scan_reached_start = False
                x, y, width, height = (int(value) for value in last_face)
                face_center_x = x + width // 2
                face_center_y = y + height // 2
                measured_error_px = face_center_x - frame_center_x
                previous_filtered_error_px = filtered_error_px
                filtered_error_px = smooth_face_error(
                    filtered_error_px,
                    measured_error_px,
                )
                if previous_filtered_error_px is not None:
                    measured_velocity = (
                        filtered_error_px - previous_filtered_error_px
                    )
                    face_velocity_px_per_frame += VELOCITY_SMOOTHING * (
                        measured_velocity - face_velocity_px_per_frame
                    )
                else:
                    face_velocity_px_per_frame = 0.0
                error_px = filtered_error_px
                current_angle = update_pan_angle(
                    current_angle,
                    filtered_error_px,
                    invert_direction,
                    face_velocity_px_per_frame,
                )
            else:
                face_center_x = None
                error_px = None
                filtered_error_px = None
                face_velocity_px_per_frame = 0.0
                if USE_SERVO:
                    if scan_active:
                        elapsed = min(max(0.0, now - last_scan_update), 0.1)
                        last_scan_update = now
                        (
                            current_angle,
                            scan_direction,
                            scan_completed_cycles,
                            scan_reached_start,
                        ) = advance_scan(
                            current_angle,
                            scan_direction,
                            scan_completed_cycles,
                            scan_reached_start,
                            elapsed,
                        )
                        if scan_completed_cycles >= SCAN_CYCLES:
                            scan_active = False
                            scan_wait_until = now + SCAN_RETRY_DELAY_SECONDS
                    elif scan_wait_until is not None:
                        if now >= scan_wait_until:
                            scan_wait_until = None
                            scan_active = True
                            scan_direction = -1
                            scan_completed_cycles = 0
                            scan_reached_start = False
                            last_scan_update = now
                    elif now - last_face_seen_at >= NO_FACE_SCAN_DELAY_SECONDS:
                        scan_active = True
                        scan_direction = -1
                        scan_completed_cycles = 0
                        scan_reached_start = False
                        last_scan_update = now

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
            if last_face is not None:
                tracking_status = "TRACKING"
            elif scan_active:
                tracking_status = "SCANNING {0}/{1}".format(
                    scan_completed_cycles + 1,
                    SCAN_CYCLES,
                )
            elif scan_wait_until is not None:
                tracking_status = "RESCAN IN {0}s".format(
                    max(0, math.ceil(scan_wait_until - time.monotonic()))
                )
            else:
                no_face_seconds = int(time.monotonic() - last_face_seen_at)
                tracking_status = "NO FACE ({0}s)".format(no_face_seconds)

            cv2.putText(
                preview,
                tracking_status,
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
