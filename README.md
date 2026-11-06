# Face Tracking with Arduino and OpenCV

This project uses a webcam and OpenCV to detect a face, then turns a pan servo
through an Arduino so the camera follows the face horizontally. A preview shows
the camera image, detected face, image center, and tracking status.

The current tracker uses one pan servo. It cannot move the camera vertically.
The original two-servo Arduino sketch is kept separately as
[`src/servo.ino`](src/servo.ino); it uses a different serial protocol and is
not compatible with the current tracker.

## How tracking works

1. OpenCV reads frames from the selected webcam and detects faces with its
   bundled Haar cascade.
2. The largest detected face is treated as the subject.
3. The tracker compares the face's horizontal center with the frame center.
4. It converts the horizontal error into a bounded pan angle and sends that
   angle to the Arduino over serial.
5. The Arduino moves the servo connected to pin 6.

When the face is near the center, the servo holds its position. The tracker
shows `TRACKING` or `NO FACE`, the horizontal pixel offset, the commanded angle,
and the active pan direction in its preview.

## Hardware

- Arduino board connected to the computer over USB
- One pan servo and a mount for the camera
- Webcam

Connect the servo's signal wire to Arduino pin 6. Power the servo using a
supply appropriate for the servo; if using a separate supply, connect its
ground to Arduino GND. Do not power a servo from the Arduino's 5 V output if
the servo current exceeds what the board can safely supply.

## Controls

- **R**: switch pan direction while tracking.
- **S**: save the current preview image when a face is detected. The image is
  written to the current working directory as `only_face<frame>.png`.
- **Esc** or **Q**: exit and release the camera and serial port.

If the servo moves away from the face, press **R** to reverse direction.

## Troubleshooting

- **The camera does not open:** try another `--camera` index and close other
  programs that might be using the webcam.
- **The preview says `NO FACE`:** improve lighting, face the camera, and make
  sure your face is visible and large enough in the frame.
- **The servo does not move:** verify the servo signal is connected to pin 6,
  upload `arduino/FaceTracker/FaceTracker.ino`, check the selected `--port`,
  close Arduino Serial Monitor, and confirm the sketch and tracker use
  115200 baud.
- **The servo moves in the wrong direction:** press **R** in the preview.
- **The camera or Arduino connection gets stuck:** exit the tracker, close
  other applications using the camera or serial port, check the USB and servo
  power connections, then start the tracker again.

Servo motion depends on the servo's power supply and mechanical range. The
tracker limits commanded angles to reduce the chance of driving the servo into
its mechanical stops; adjust the angle limits in `tracker.py` only to match
your hardware.
