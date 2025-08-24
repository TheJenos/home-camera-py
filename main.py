import os
from tabnanny import verbose
from onvif import WsDiscoveryClient, OnvifClient
import cv2
from ultralytics import YOLO
import time

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def main():
    # Load camera username and password from environment variables
    camera_username = os.environ.get('CAMERA_USERNAME')
    camera_password = os.environ.get('CAMERA_PASSWORD')

    # yolo = YOLO('yolov8s.pt')
    # yolo = YOLO('yolo11n-pose.pt')
    yolo = YOLO('yolo11n.pt')

    wsd_client = WsDiscoveryClient()
    nvts = wsd_client.search()
    for nvt in nvts:
        if nvt.ip_address != '192.168.1.5':
            continue

        onvif_client = OnvifClient(nvt.ip_address, nvt.port, camera_username, camera_password)
        profile_tokens = onvif_client.get_profile_tokens()
        if not profile_tokens:
            print(f"No profiles found for {nvt.ip_address}:{nvt.port}")
            continue
        profile_token = profile_tokens[1]

        stream_uri = onvif_client.get_streaming_uri(profile_token)

        print(f"Streaming from: {stream_uri}")
        cap = cv2.VideoCapture(stream_uri)
        if not cap.isOpened():
            print("Failed to open stream.")
            continue

        fps_smooth = None
        last_time = time.time()
        window_title = f"Stream {profile_token} (q to quit)"

        # Assume cap reads a frame where top half is static camera, bottom half is PTZ camera
        # We'll run YOLO separately on each half and show in two windows

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            # Split the frame into top and bottom halves
            h, w, _ = frame.shape
            mid = h // 2
            static_frame = frame[:mid, :, :]
            ptz_frame = frame[mid:, :, :]

            # Run YOLO on static camera (top half)
            static_results = yolo.track(static_frame, stream=True, verbose=False)
            static_rects = 0
            for result in static_results:
                classes_names = result.names
                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        [x1, y1, x2, y2] = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls = int(box.cls[0])
                        class_name = classes_names[cls]
                        colour = getColours(cls)
                        cv2.rectangle(static_frame, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(static_frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                        static_rects += 1

            # Run YOLO on PTZ camera (bottom half)
            ptz_results = yolo.track(ptz_frame, stream=True, verbose=False)
            ptz_rects = 0
            for result in ptz_results:
                classes_names = result.names
                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        [x1, y1, x2, y2] = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls = int(box.cls[0])
                        class_name = classes_names[cls]
                        colour = getColours(cls)
                        cv2.rectangle(ptz_frame, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(ptz_frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                        ptz_rects += 1

            # FPS calculation (shared for both streams)
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 1.0 / dt
                fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)
            else:
                fps_smooth = fps_smooth or 0

            # Overlay info for static camera
            overlay_static = f"People: {static_rects}  FPS: {fps_smooth:.1f}" if fps_smooth else f"People: {static_rects}"
            cv2.putText(static_frame, overlay_static, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(static_frame, overlay_static, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            # Overlay info for PTZ camera
            overlay_ptz = f"People: {ptz_rects}  FPS: {fps_smooth:.1f}" if fps_smooth else f"People: {ptz_rects}"
            cv2.putText(ptz_frame, overlay_ptz, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(ptz_frame, overlay_ptz, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

            # Show both streams in separate windows
            cv2.imshow(f"{window_title} - Static", static_frame)
            cv2.imshow(f"{window_title} - PTZ", ptz_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    wsd_client.dispose()


if __name__ == "__main__":
    main()
