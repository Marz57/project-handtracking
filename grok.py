import os
import time
import threading
import cv2
import numpy as np
from rtmlib import Hand, PoseTracker
from filterpy.kalman import KalmanFilter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

HAND_CONF = 0.35
KEYPOINT_CONF = 0.30
SEG_CONF = 0.30

HAND_MEMORY_FRAMES = 20

GESTURE_RATIO = 0.40
GESTURE_MIN_DISTANCE = 30
GESTURE_COOLDOWN = 0.45

PORTAL_SMOOTHING = 0.60

FILTERS = [
    "MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA",
    "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY"
]

current_filter_index = 0
current_filter = FILTERS[current_filter_index]
gesture_triggered = False
last_gesture_time = 0.0

last_hands = []
hand_memory = []
last_hand_points = []
smoothed_portal_points = None
hand_lost_frames = 0

last_segmentation = None
seg_model = None
frame_count = 0

fps = 0.0
fps_counter = 0
fps_timer = time.time()

detector_running = True
detector_lock = threading.Lock()
latest_frame = None
latest_detected_hands = []
latest_detection_id = 0

# ==================== KALMAN FILTER ====================
class HandKalman:
    def __init__(self, num_keypoints=21):
        self.filters = []
        for _ in range(num_keypoints):
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.F = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.float32)
            kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]], dtype=np.float32)
            kf.R *= 4.5
            kf.Q *= 0.08
            kf.P *= 30
            self.filters.append(kf)
        self.initialized = False

    def update(self, keypoints, scores=None):
        result = np.zeros((21, 2), dtype=np.float32)
        for i in range(21):
            if scores is not None and scores[i] < KEYPOINT_CONF:
                if self.initialized:
                    self.filters[i].predict()
                    result[i] = self.filters[i].x[:2].flatten()
                else:
                    result[i] = keypoints[i]
                continue

            z = np.array(keypoints[i], dtype=np.float32)
            if not self.initialized:
                self.filters[i].x[:2] = z.reshape(2, 1)
                result[i] = z
            else:
                self.filters[i].predict()
                self.filters[i].update(z)
                result[i] = self.filters[i].x[:2].flatten()
        self.initialized = True
        return result

    def predict_only(self):
        result = np.zeros((21, 2), dtype=np.float32)
        for i in range(21):
            self.filters[i].predict()
            result[i] = self.filters[i].x[:2].flatten()
        return result

hand_kalmans = [HandKalman(), HandKalman()]

# ==================== FILTER EFFECTS ====================
def apply_filter(frame, filter_name):
    if filter_name == "MONO":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        result = np.zeros_like(frame)
        result[:, :, 0] = normalized
        result[:, :, 1] = np.clip(normalized * 0.55, 0, 255)
        result[:, :, 2] = np.clip(255 - normalized * 0.35, 0, 255)
        return result.astype(np.uint8)
    if filter_name == "PIXELATE":
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 18), max(1, h // 18)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if filter_name == "INVERT":
        return cv2.bitwise_not(frame)
    if filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)
    if filter_name == "BLUR":
        return cv2.GaussianBlur(frame, (21, 21), 0)
    if filter_name == "THERMAL":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    if filter_name == "SKETCH":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    if filter_name == "GLITCH":
        result = frame.copy()
        h, w = frame.shape[:2]
        shift = max(2, w // 70)
        b, g, r = cv2.split(frame)
        result[:, :, 0] = np.roll(b, -shift, axis=1)
        result[:, :, 2] = np.roll(r, shift, axis=1)
        for _ in range(6):
            y = np.random.randint(0, h)
            hh = np.random.randint(2, max(3, h // 28))
            offset = np.random.randint(-shift * 2, shift * 2 + 1)
            y2 = min(h, y + hh)
            if offset > 0:
                result[y:y2, offset:] = frame[y:y2, :-offset]
            elif offset < 0:
                result[y:y2, :offset] = frame[y:y2, -offset:]
        return result
    if filter_name == "NEON":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        glow = cv2.GaussianBlur(edges, (0, 0), 6)
        neon = np.zeros_like(frame)
        neon[:, :, 0] = glow
        neon[:, :, 1] = edges
        neon[:, :, 2] = np.clip(edges * 2.2, 0, 255)
        return cv2.addWeighted(frame, 0.32, neon, 1.45, 0)
    if filter_name == "GALAXY":
        h, w = frame.shape[:2]
        y, x = np.indices((h, w))
        wave1 = np.sin(x * 0.022 + y * 0.011)
        wave2 = np.sin(x * 0.011 - y * 0.022)
        wave3 = np.sin((x + y) * 0.016)
        galaxy = np.zeros((h, w, 3), dtype=np.float32)
        galaxy[:, :, 0] = (wave1 + 1.0) * 68 + 48
        galaxy[:, :, 1] = (wave2 + 1.0) * 42 + 28
        galaxy[:, :, 2] = (wave3 + 1.0) * 68 + 48
        galaxy = np.clip(galaxy, 0, 255).astype(np.uint8)
        stars = (np.random.random((h, w)) > 0.9965).astype(np.uint8) * 255
        stars = cv2.GaussianBlur(stars, (0, 0), 1.2)
        galaxy = np.clip(galaxy.astype(np.int16) + stars[:, :, None].astype(np.int16), 0, 255).astype(np.uint8)
        galaxy = cv2.GaussianBlur(galaxy, (5, 5), 0)
        return cv2.addWeighted(frame, 0.22, galaxy, 0.98, 0)
    return frame.copy()

def apply_galaxy_segmentation(frame, segmentation_result):
    galaxy = apply_filter(frame, "GALAXY")
    if segmentation_result is None or not hasattr(segmentation_result, 'masks') or segmentation_result.masks is None:
        return galaxy
    masks = segmentation_result.masks.data
    if masks is None or len(masks) == 0:
        return galaxy
    combined = np.zeros(frame.shape[:2], dtype=np.uint8)
    for mask in masks:
        m = mask.cpu().numpy()
        m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = (m > SEG_CONF).astype(np.uint8) * 255
        combined = cv2.max(combined, m)
    kernel = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.GaussianBlur(combined, (9, 9), 0)
    alpha = (combined.astype(np.float32) / 255.0)[:, :, None]
    result = frame.astype(np.float32) * (1 - alpha) + galaxy.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)

# ==================== HAND PROCESSING ====================
def get_hand_data_from_rtm(keypoints, scores):
    hands = []
    if keypoints is None or len(keypoints) == 0:
        return hands
    for i, kpts in enumerate(keypoints):
        if kpts is None or len(kpts) < 21:
            continue
        sc = scores[i] if scores is not None and i < len(scores) else np.ones(21)
        if np.mean(sc[[0, 4, 8, 12, 16, 20]]) < HAND_CONF:
            continue
        kpts = np.array(kpts, dtype=np.float32)
        wrist = kpts[0]
        thumb = kpts[4]
        index = kpts[8]
        pinky = kpts[20]
        x_min, y_min = np.min(kpts, axis=0)
        x_max, y_max = np.max(kpts, axis=0)
        hands.append({
            "keypoints": kpts,
            "scores": sc,
            "box": np.array([x_min, y_min, x_max, y_max], dtype=np.float32),
            "wrist": (float(wrist[0]), float(wrist[1])),
            "thumb": (float(thumb[0]), float(thumb[1])),
            "index": (float(index[0]), float(index[1])),
            "pinky": (float(pinky[0]), float(pinky[1])),
            "confidence": float(np.mean(sc))
        })
    hands.sort(key=lambda h: h["wrist"][0])
    return hands[:2]

def apply_kalman(hands):
    global hand_kalmans
    result = []
    for i, hand in enumerate(hands):
        if i >= 2:
            break
        smoothed = hand_kalmans[i].update(hand["keypoints"], hand.get("scores"))
        new_hand = hand.copy()
        new_hand["keypoints"] = smoothed
        new_hand["wrist"] = (float(smoothed[0][0]), float(smoothed[0][1]))
        new_hand["thumb"] = (float(smoothed[4][0]), float(smoothed[4][1]))
        new_hand["index"] = (float(smoothed[8][0]), float(smoothed[8][1]))
        new_hand["pinky"] = (float(smoothed[20][0]), float(smoothed[20][1]))
        result.append(new_hand)
    return result

def detect_gesture(hands):
    for hand in hands:
        x1, y1, x2, y2 = hand["box"]
        hand_size = max(x2 - x1, y2 - y1, 1.0)
        dist = np.linalg.norm(np.array(hand["thumb"]) - np.array(hand["pinky"]))
        if dist <= GESTURE_MIN_DISTANCE or (dist / hand_size) <= GESTURE_RATIO:
            return True
    return False

def update_filter(hands):
    global current_filter_index, current_filter, gesture_triggered, last_gesture_time
    gesture = detect_gesture(hands)
    now = time.time()
    if gesture and not gesture_triggered and now - last_gesture_time >= GESTURE_COOLDOWN:
        current_filter_index = (current_filter_index + 1) % len(FILTERS)
        current_filter = FILTERS[current_filter_index]
        gesture_triggered = True
        last_gesture_time = now
    if not gesture:
        gesture_triggered = False

def get_portal_points(hands):
    if len(hands) < 2:
        return []
    ordered = sorted(hands, key=lambda h: h["wrist"][0])
    return [
        tuple(np.round(ordered[0]["thumb"]).astype(int)),
        tuple(np.round(ordered[0]["index"]).astype(int)),
        tuple(np.round(ordered[1]["thumb"]).astype(int)),
        tuple(np.round(ordered[1]["index"]).astype(int))
    ]

def smooth_portal_points(new_points):
    global smoothed_portal_points
    if new_points is None or len(new_points) < 4:
        return smoothed_portal_points
    pts = np.array(new_points, dtype=np.float32)
    if smoothed_portal_points is None:
        smoothed_portal_points = pts.copy()
    else:
        smoothed_portal_points = smoothed_portal_points * (1 - PORTAL_SMOOTHING) + pts * PORTAL_SMOOTHING
    return [tuple(np.round(p).astype(int)) for p in smoothed_portal_points]

def create_portal_mask(frame, points):
    if len(points) < 4:
        return None
    h, w = frame.shape[:2]
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts.astype(np.int32))
    if cv2.contourArea(hull) < 1200:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return mask, hull

def apply_portal_filter(frame, filter_name, points):
    data = create_portal_mask(frame, points)
    if data is None:
        return frame.copy(), None
    mask, hull = data
    filtered = apply_filter(frame, filter_name)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    result = frame.astype(np.float32) * (1 - alpha) + filtered.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8), hull

def draw_hand_points(frame, hands):
    result = frame.copy()
    for hand in hands:
        for tip in [hand["thumb"], hand["index"], hand["pinky"]]:
            pt = (int(tip[0]), int(tip[1]))
            cv2.circle(result, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(result, pt, 7, (0, 180, 255), 1, cv2.LINE_AA)
    return result

def draw_portal_effect(frame, hull):
    if hull is None:
        return frame
    result = frame.copy()
    glow = np.zeros_like(result)
    cv2.polylines(glow, [hull], True, (0, 220, 255), 16, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 9)
    result = cv2.addWeighted(result, 1.0, glow, 0.70, 0)
    cv2.polylines(result, [hull], True, (255, 255, 255), 3, cv2.LINE_AA)
    return result

def draw_portal_particles(frame, hull):
    if hull is None:
        return frame
    result = frame.copy()
    contour = hull.reshape(-1, 2)
    if len(contour) < 3:
        return result
    now = time.time()
    for i in range(36):
        idx = i % len(contour)
        p1 = contour[idx]
        p2 = contour[(idx + 1) % len(contour)]
        t = (now * 0.65 + i * 0.068) % 1.0
        x = int(p1[0] * (1 - t) + p2[0] * t) + int(np.sin(now * 3.8 + i) * 4)
        y = int(p1[1] * (1 - t) + p2[1] * t) + int(np.cos(now * 3.8 + i) * 4)
        cv2.circle(result, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)
    return result

def draw_ui(frame, hands, portal_active):
    result = frame.copy()
    h, w = result.shape[:2]
    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (310, 115), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.48, result, 0.52, 0)
    cv2.putText(result, f"PORTAL: {current_filter}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"HANDS: {len(hands)}", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    status = "ACTIVE" if portal_active else "WAITING"
    cv2.putText(result, f"PORTAL: {status}", (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"FPS: {fps:.1f}", (w - 115, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "1 tangan: Jempol + Kelingking = Ganti Filter", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return result

# ==================== DETECTOR THREAD ====================
def detector_worker():
    global latest_frame, latest_detected_hands, latest_detection_id, detector_running
    device = 'cpu'
    backend = 'onnxruntime'
    hand_tracker = PoseTracker(
        Hand,
        det_frequency=5,
        to_openpose=False,
        mode='lightweight',
        backend=backend,
        device=device
    )
    while detector_running:
        with detector_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.004)
            continue
        try:
            keypoints, scores = hand_tracker(frame)
            detected = get_hand_data_from_rtm(keypoints, scores)
            with detector_lock:
                latest_detected_hands = detected
                latest_detection_id += 1
        except Exception:
            time.sleep(0.01)

# ==================== MAIN ====================
detector_thread = threading.Thread(target=detector_worker, daemon=True)
detector_thread.start()

camera = None
for src in [2, 1, 0]:
    test = cv2.VideoCapture(src)
    if test.isOpened():
        ret, _ = test.read()
        if ret:
            camera = test
            break
        test.release()
if camera is None:
    camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    detector_running = False
    raise RuntimeError("Tidak ada kamera yang bisa dibuka.")

last_detection_id = 0

while True:
    success, frame = camera.read()
    if not success or frame is None:
        continue

    frame_count += 1
    frame = cv2.flip(frame, 1)

    with detector_lock:
        latest_frame = frame.copy()
        detected_hands = [h.copy() for h in latest_detected_hands]
        detection_id = latest_detection_id

    if detection_id != last_detection_id:
        last_detection_id = detection_id
        if detected_hands:
            last_hands = apply_kalman(detected_hands)
            hand_lost_frames = 0
        else:
            hand_lost_frames += 1
            if hand_lost_frames <= HAND_MEMORY_FRAMES and last_hands:
                # prediksi dengan Kalman
                predicted = []
                for i, hand in enumerate(last_hands[:2]):
                    pred_kpts = hand_kalmans[i].predict_only()
                    new_hand = hand.copy()
                    new_hand["keypoints"] = pred_kpts
                    new_hand["wrist"] = (float(pred_kpts[0][0]), float(pred_kpts[0][1]))
                    new_hand["thumb"] = (float(pred_kpts[4][0]), float(pred_kpts[4][1]))
                    new_hand["index"] = (float(pred_kpts[8][0]), float(pred_kpts[8][1]))
                    new_hand["pinky"] = (float(pred_kpts[20][0]), float(pred_kpts[20][1]))
                    predicted.append(new_hand)
                last_hands = predicted
    else:
        if last_hands:
            hand_lost_frames = 0
        else:
            hand_lost_frames += 1

    if len(last_hands) > 0:
        update_filter(last_hands)

    if len(last_hands) >= 2:
        hand_memory = [h.copy() for h in last_hands]
        raw_points = get_portal_points(last_hands)
        last_hand_points = smooth_portal_points(raw_points)
    elif len(hand_memory) >= 2 and hand_lost_frames <= HAND_MEMORY_FRAMES:
        raw_points = get_portal_points(hand_memory)
        last_hand_points = smooth_portal_points(raw_points)
    else:
        last_hand_points = []
        smoothed_portal_points = None
        if hand_lost_frames > HAND_MEMORY_FRAMES:
            hand_memory = []
            hand_kalmans = [HandKalman(), HandKalman()]

    # Galaxy segmentation (optional, tetap pakai YOLO kalau ada)
    if current_filter == "GALAXY":
        try:
            from ultralytics import YOLO
            if seg_model is None:
                seg_model = YOLO("segmentation_model.pt")
            if last_segmentation is None or frame_count % 12 == 0:
                res = seg_model.predict(frame, imgsz=256, conf=SEG_CONF, verbose=False, device="cpu")
                if res:
                    last_segmentation = res[0]
            base_frame = apply_galaxy_segmentation(frame, last_segmentation)
        except Exception:
            base_frame = frame.copy()
    else:
        base_frame = frame.copy()
        last_segmentation = None

    portal_active = False
    portal_hull = None

    if len(last_hand_points) >= 4:
        portal_frame, portal_hull = apply_portal_filter(frame, current_filter, last_hand_points)
        if portal_hull is not None:
            portal_active = True
            filtered_frame = draw_portal_effect(portal_frame, portal_hull)
            filtered_frame = draw_portal_particles(filtered_frame, portal_hull)
        else:
            filtered_frame = base_frame
    else:
        filtered_frame = base_frame

    filtered_frame = draw_hand_points(filtered_frame, last_hands)

    fps_counter += 1
    now = time.time()
    if now - fps_timer >= 1.0:
        fps = fps_counter / (now - fps_timer)
        fps_counter = 0
        fps_timer = now

    output = draw_ui(filtered_frame, last_hands, portal_active)
    cv2.imshow("Retrolens - RTMPose + Kalman", output)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

detector_running = False
camera.release()
cv2.destroyAllWindows()