from dotenv import load_dotenv
import os
import threading
from queue import Queue
from onvif import WsDiscoveryClient, OnvifClient
import cv2
from ultralytics import YOLO
import time

load_dotenv()

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def process_camera(nvt, camera_username, camera_password, yolo, frame_queue, stop_event):
    onvif_client = OnvifClient(nvt.ip_address, nvt.port, camera_username, camera_password)
    profile_tokens = onvif_client.get_profile_tokens()
    if not profile_tokens:
        print(f"No profiles found for {nvt.ip_address}:{nvt.port}")
        return
    profile_token = profile_tokens[1]

    stream_uri = onvif_client.get_streaming_uri(profile_token)

    print(f"Streaming from: {stream_uri}")

    cap = None
    fps_smooth = None
    last_time = time.time()
    window_title = f"Stream {profile_token} ({nvt.ip_address}) (q to quit)"

    while not stop_event.is_set():
        # (Re)open the stream if needed
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(stream_uri)
            if not cap.isOpened():
                print("Failed to open stream. Retrying in 2 seconds...")
                time.sleep(2)
                continue

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Restarting stream...")
            cap.release()
            cap = None
            time.sleep(1)
            continue

        # Convert the frame to grayscale (black and white)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels for further processing if needed
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Split the frame into top and bottom halves
        h, w, _ = frame.shape
        mid = h // 2
        static_frame = frame[:mid, :, :].copy()
        ptz_frame = frame[mid:, :, :].copy()

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

        # Put the processed frames and window titles into the queue for display
        frame_queue.put((f"{window_title} - Static", static_frame))
        frame_queue.put((f"{window_title} - PTZ", ptz_frame))

    if cap is not None:
        cap.release()

def imshow_worker(frame_queues, stop_event):
    # frame_queues: list of (queue, stop_event) tuples
    window_names = set()
    while not stop_event.is_set():
        any_frame = False
        for frame_queue, cam_stop_event in frame_queues:
            try:
                # Non-blocking get
                window_title, frame = frame_queue.get_nowait()
                cv2.imshow(window_title, frame)
                window_names.add(window_title)
                any_frame = True
            except Exception:
                continue
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            for _, cam_stop_event in frame_queues:
                cam_stop_event.set()
            break
        # If no frames at all, sleep a bit to avoid busy loop
        if not any_frame:
            time.sleep(0.01)
    # Cleanup
    for name in window_names:
        cv2.destroyWindow(name)
    cv2.destroyAllWindows()

def main():
    # Load camera username and password from environment variables
    camera_username = os.getenv('CAMERA_USERNAME')
    camera_password = os.getenv('CAMERA_PASSWORD')

    # yolo = YOLO('yolov8s.pt')
    # yolo = YOLO('yolo11n-pose.pt')
    # yolo = YOLO('yolo11n.pt')
    yolo = YOLO('yolo11s-person.pt')

    wsd_client = WsDiscoveryClient()
    nvts = wsd_client.search()

    threads = []
    frame_queues = []
    global_stop_event = threading.Event()

    for nvt in nvts:
        frame_queue = Queue(maxsize=2)
        cam_stop_event = threading.Event()
        t = threading.Thread(target=process_camera, args=(nvt, camera_username, camera_password, yolo, frame_queue, cam_stop_event))
        t.start()
        threads.append((t, cam_stop_event))
        frame_queues.append((frame_queue, cam_stop_event))

    # Start the imshow worker in the main thread (cv2.imshow must run in main thread)
    try:
        imshow_worker(frame_queues, global_stop_event)
    except KeyboardInterrupt:
        global_stop_event.set()
        for _, cam_stop_event in frame_queues:
            cam_stop_event.set()

    # Wait for all threads to finish
    for t, cam_stop_event in threads:
        t.join()

    wsd_client.dispose()


if __name__ == "__main__":
    main()
