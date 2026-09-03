import os
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO

HAND_MODEL = "hand_pose.pt"
SEG_MODEL = "segmentation_model.pt"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

HAND_IMGSZ = 416
SEG_IMGSZ = 256

HAND_CONF = 0.01
KEYPOINT_CONF = 0.01
SEG_CONF = 0.25

HAND_MEMORY_FRAMES = 18

GESTURE_RATIO = 0.38
GESTURE_MIN_DISTANCE = 30
GESTURE_COOLDOWN = 0.55

SMOOTHING_FACTOR = 0.65
TIP_SMOOTHING_FACTOR = 0.82

FLOW_WIN_SIZE = (31, 31)
FLOW_MAX_LEVEL = 3
FLOW_MIN_POINTS = 12
FLOW_MAX_DISTANCE = 80.0

FILTERS = [
    "MONO",
    "DUAL-TONE",
    "PIXELATE",
    "INVERT",
    "SEPIA",
    "BLUR",
    "THERMAL",
    "SKETCH",
    "GLITCH",
    "NEON",
    "GALAXY"
]

current_filter_index = 0
current_filter = FILTERS[current_filter_index]

gesture_triggered = False
last_gesture_time = 0.0

last_hands = []
hand_memory = []
last_hand_points = []

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


def apply_filter(frame, filter_name):
    if filter_name == "MONO":
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        return cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

    if filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        normalized = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        result = np.zeros_like(frame)

        result[:, :, 0] = normalized

        result[:, :, 1] = np.clip(
            normalized * 0.55,
            0,
            255
        )

        result[:, :, 2] = np.clip(
            255 - normalized * 0.35,
            0,
            255
        )

        return result.astype(np.uint8)

    if filter_name == "PIXELATE":
        height, width = frame.shape[:2]

        small_width = max(
            1,
            width // 20
        )

        small_height = max(
            1,
            height // 20
        )

        small = cv2.resize(
            frame,
            (
                small_width,
                small_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

        return cv2.resize(
            small,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_NEAREST
        )

    if filter_name == "INVERT":
        return cv2.bitwise_not(frame)

    if filter_name == "SEPIA":
        kernel = np.array(
            [
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ]
        )

        result = cv2.transform(
            frame,
            kernel
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

    if filter_name == "BLUR":
        return cv2.GaussianBlur(
            frame,
            (21, 21),
            0
        )

    if filter_name == "THERMAL":
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        return cv2.applyColorMap(
            gray,
            cv2.COLORMAP_JET
        )

    if filter_name == "SKETCH":
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        inverted = cv2.bitwise_not(
            gray
        )

        blurred = cv2.GaussianBlur(
            inverted,
            (21, 21),
            0
        )

        sketch = cv2.divide(
            gray,
            255 - blurred,
            scale=256
        )

        return cv2.cvtColor(
            sketch,
            cv2.COLOR_GRAY2BGR
        )

    if filter_name == "GLITCH":
        result = frame.copy()

        height, width = frame.shape[:2]

        shift = max(
            2,
            width // 80
        )

        b, g, r = cv2.split(frame)

        r_shifted = np.roll(
            r,
            shift,
            axis=1
        )

        b_shifted = np.roll(
            b,
            -shift,
            axis=1
        )

        result[:, :, 0] = b_shifted
        result[:, :, 1] = g
        result[:, :, 2] = r_shifted

        for _ in range(5):
            y = np.random.randint(
                0,
                height
            )

            h = np.random.randint(
                2,
                max(
                    3,
                    height // 30
                )
            )

            offset = np.random.randint(
                -shift * 2,
                shift * 2 + 1
            )

            y2 = min(
                height,
                y + h
            )

            if offset > 0:
                result[
                    y:y2,
                    offset:
                ] = frame[
                    y:y2,
                    :-offset
                ]

            elif offset < 0:
                result[
                    y:y2,
                    :offset
                ] = frame[
                    y:y2,
                    -offset:
                ]

        return result

    if filter_name == "NEON":
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            80,
            160
        )

        glow = cv2.GaussianBlur(
            edges,
            (0, 0),
            5
        )

        neon = np.zeros_like(
            frame
        )

        neon[:, :, 0] = glow
        neon[:, :, 1] = edges
        neon[:, :, 2] = np.clip(
            edges * 2,
            0,
            255
        )

        return cv2.addWeighted(
            frame,
            0.35,
            neon,
            1.4,
            0
        )

    if filter_name == "GALAXY":
        height, width = frame.shape[:2]

        y, x = np.indices(
            (
                height,
                width
            )
        )

        wave1 = np.sin(
            x * 0.025 +
            y * 0.012
        )

        wave2 = np.sin(
            x * 0.012 -
            y * 0.025
        )

        wave3 = np.sin(
            (x + y) * 0.018
        )

        galaxy = np.zeros(
            (
                height,
                width,
                3
            ),
            dtype=np.float32
        )

        galaxy[:, :, 0] = (
            wave1 + 1.0
        ) * 70 + 50

        galaxy[:, :, 1] = (
            wave2 + 1.0
        ) * 45 + 30

        galaxy[:, :, 2] = (
            wave3 + 1.0
        ) * 70 + 50

        galaxy = np.clip(
            galaxy,
            0,
            255
        ).astype(np.uint8)

        stars = np.random.random(
            (
                height,
                width
            )
        )

        stars = (
            stars > 0.997
        ).astype(
            np.uint8
        ) * 255

        stars = cv2.GaussianBlur(
            stars,
            (0, 0),
            1
        )

        galaxy[:, :, 0] = np.clip(
            galaxy[:, :, 0].astype(np.int16)
            + stars.astype(np.int16),
            0,
            255
        )

        galaxy[:, :, 1] = np.clip(
            galaxy[:, :, 1].astype(np.int16)
            + stars.astype(np.int16),
            0,
            255
        )

        galaxy[:, :, 2] = np.clip(
            galaxy[:, :, 2].astype(np.int16)
            + stars.astype(np.int16),
            0,
            255
        )

        galaxy = cv2.GaussianBlur(
            galaxy,
            (5, 5),
            0
        )

        return cv2.addWeighted(
            frame,
            0.25,
            galaxy,
            0.95,
            0
        )

    return frame.copy()


def apply_galaxy_segmentation(
    frame,
    segmentation_result
):
    galaxy = apply_filter(
        frame,
        "GALAXY"
    )

    if segmentation_result is None:
        return galaxy

    if segmentation_result.masks is None:
        return galaxy

    masks = segmentation_result.masks.data

    if masks is None or len(masks) == 0:
        return galaxy

    combined_mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    for mask in masks:
        current_mask = (
            mask.cpu().numpy()
        )

        current_mask = cv2.resize(
            current_mask,
            (
                frame.shape[1],
                frame.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        current_mask = (
            current_mask > SEG_CONF
        ).astype(
            np.uint8
        ) * 255

        combined_mask = cv2.max(
            combined_mask,
            current_mask
        )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    combined_mask = cv2.GaussianBlur(
        combined_mask,
        (7, 7),
        0
    )

    alpha = (
        combined_mask.astype(
            np.float32
        ) / 255.0
    )

    alpha = alpha[:, :, None]

    result = (
        frame.astype(np.float32)
        * (1.0 - alpha)
        + galaxy.astype(np.float32)
        * alpha
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def get_hand_data(result):
    if result is None:
        return []

    if result.keypoints is None:
        return []

    if result.boxes is None:
        return []

    keypoints = result.keypoints.xy

    if keypoints is None:
        return []

    keypoints = (
        keypoints
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    box_conf = result.boxes.conf

    if box_conf is not None:
        box_conf = (
            box_conf
            .cpu()
            .numpy()
        )
    else:
        box_conf = np.ones(
            len(keypoints)
        )

    hands = []

    for i, hand in enumerate(
        keypoints
    ):
        if i >= len(box_conf):
            continue

        if box_conf[i] < HAND_CONF:
            continue

        if len(hand) < 21:
            continue

        if not np.isfinite(
            hand
        ).all():
            continue

        x1, y1, x2, y2 = (
            result.boxes.xyxy[i]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        wrist = hand[0]
        thumb = hand[4]
        index_finger = hand[8]
        pinky = hand[20]

        hands.append(
            {
                "keypoints": hand.copy(),
                "box": np.array(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    dtype=np.float32
                ),
                "wrist": (
                    float(wrist[0]),
                    float(wrist[1])
                ),
                "thumb": (
                    float(thumb[0]),
                    float(thumb[1])
                ),
                "index": (
                    float(index_finger[0]),
                    float(index_finger[1])
                ),
                "pinky": (
                    float(pinky[0]),
                    float(pinky[1])
                ),
                "confidence": float(
                    box_conf[i]
                )
            }
        )

    hands.sort(
        key=lambda hand: hand["wrist"][0]
    )

    return hands[:2]


def copy_hand(hand):
    result = dict(hand)

    result["keypoints"] = (
        hand["keypoints"].copy()
    )

    result["box"] = (
        hand["box"].copy()
    )

    return result


def update_hand_points(
    hand,
    points
):
    updated = copy_hand(
        hand
    )

    updated["keypoints"] = (
        points.astype(
            np.float32
        )
    )

    updated["wrist"] = (
        float(points[0][0]),
        float(points[0][1])
    )

    updated["thumb"] = (
        float(points[4][0]),
        float(points[4][1])
    )

    updated["index"] = (
        float(points[8][0]),
        float(points[8][1])
    )

    updated["pinky"] = (
        float(points[20][0]),
        float(points[20][1])
    )

    x_values = points[:, 0]
    y_values = points[:, 1]

    updated["box"] = np.array(
        [
            np.min(x_values),
            np.min(y_values),
            np.max(x_values),
            np.max(y_values)
        ],
        dtype=np.float32
    )

    return updated


def smooth_detection(
    previous,
    detected
):
    if previous is None:
        return copy_hand(
            detected
        )

    old_points = (
        previous["keypoints"]
        .astype(np.float32)
    )

    new_points = (
        detected["keypoints"]
        .astype(np.float32)
    )

    result = new_points.copy()

    for i in range(
        len(result)
    ):
        if i in [4, 8, 20]:
            factor = TIP_SMOOTHING_FACTOR
        else:
            factor = SMOOTHING_FACTOR

        result[i] = (
            old_points[i]
            * (1.0 - factor)
            + new_points[i]
            * factor
        )

    return update_hand_points(
        detected,
        result
    )


def distance(
    point_a,
    point_b
):
    return float(
        np.linalg.norm(
            np.array(
                point_a,
                dtype=np.float32
            )
            -
            np.array(
                point_b,
                dtype=np.float32
            )
        )
    )


def match_hands(
    previous,
    detected
):
    if not detected:
        return []

    if not previous:
        return [
            copy_hand(hand)
            for hand in detected
        ]

    if len(previous) == 1:
        if len(detected) == 1:
            return [
                smooth_detection(
                    previous[0],
                    detected[0]
                )
            ]

        nearest_index = np.argmin(
            [
                distance(
                    previous[0]["wrist"],
                    hand["wrist"]
                )
                for hand in detected
            ]
        )

        result = [
            smooth_detection(
                previous[0],
                detected[nearest_index]
            )
        ]

        for i, hand in enumerate(
            detected
        ):
            if i != nearest_index:
                result.append(
                    copy_hand(hand)
                )

        return result[:2]

    if len(detected) == 1:
        nearest_index = np.argmin(
            [
                distance(
                    previous_hand["wrist"],
                    detected[0]["wrist"]
                )
                for previous_hand in previous
            ]
        )

        return [
            smooth_detection(
                previous[nearest_index],
                detected[0]
            )
        ]

    direct_distance = (
        distance(
            previous[0]["wrist"],
            detected[0]["wrist"]
        )
        +
        distance(
            previous[1]["wrist"],
            detected[1]["wrist"]
        )
    )

    crossed_distance = (
        distance(
            previous[0]["wrist"],
            detected[1]["wrist"]
        )
        +
        distance(
            previous[1]["wrist"],
            detected[0]["wrist"]
        )
    )

    if direct_distance <= crossed_distance:
        first = detected[0]
        second = detected[1]
    else:
        first = detected[1]
        second = detected[0]

    return [
        smooth_detection(
            previous[0],
            first
        ),
        smooth_detection(
            previous[1],
            second
        )
    ]


def track_hands(
    previous_gray,
    current_gray,
    hands
):
    if (
        previous_gray is None
        or current_gray is None
        or not hands
    ):
        return [
            copy_hand(hand)
            for hand in hands
        ]

    tracked_hands = []

    for hand in hands:
        old_points = (
            hand["keypoints"]
            .astype(np.float32)
            .reshape(-1, 1, 2)
        )

        try:
            new_points, status, error = (
                cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    current_gray,
                    old_points,
                    None,
                    winSize=FLOW_WIN_SIZE,
                    maxLevel=FLOW_MAX_LEVEL,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS
                        | cv2.TERM_CRITERIA_COUNT,
                        20,
                        0.03
                    )
                )
            )
        except Exception:
            tracked_hands.append(
                copy_hand(hand)
            )
            continue

        if (
            new_points is None
            or status is None
        ):
            tracked_hands.append(
                copy_hand(hand)
            )
            continue

        new_points = (
            new_points
            .reshape(-1, 2)
            .astype(np.float32)
        )

        status = (
            status
            .reshape(-1)
            .astype(bool)
        )

        tracked = (
            hand["keypoints"]
            .copy()
            .astype(np.float32)
        )

        valid_count = int(
            np.count_nonzero(status)
        )

        if valid_count < FLOW_MIN_POINTS:
            tracked_hands.append(
                copy_hand(hand)
            )
            continue

        for i in range(
            len(tracked)
        ):
            if not status[i]:
                continue

            movement = distance(
                tracked[i],
                new_points[i]
            )

            if movement > FLOW_MAX_DISTANCE:
                continue

            tracked[i] = (
                new_points[i]
            )

        tracked[:, 0] = np.clip(
            tracked[:, 0],
            0,
            current_gray.shape[1] - 1
        )

        tracked[:, 1] = np.clip(
            tracked[:, 1],
            0,
            current_gray.shape[0] - 1
        )

        tracked_hands.append(
            update_hand_points(
                hand,
                tracked
            )
        )

    return tracked_hands


def detect_gesture(hands):
    if len(hands) == 0:
        return False

    for hand in hands:
        x1, y1, x2, y2 = (
            hand["box"]
        )

        hand_width = max(
            1.0,
            float(x2 - x1)
        )

        hand_height = max(
            1.0,
            float(y2 - y1)
        )

        hand_size = max(
            hand_width,
            hand_height
        )

        thumb_pinky = distance(
            hand["thumb"],
            hand["pinky"]
        )

        ratio = (
            thumb_pinky
            / hand_size
        )

        if (
            thumb_pinky
            <= GESTURE_MIN_DISTANCE
            or ratio
            <= GESTURE_RATIO
        ):
            return True

    return False


def update_filter(hands):
    global current_filter_index
    global current_filter
    global gesture_triggered
    global last_gesture_time

    gesture = detect_gesture(
        hands
    )

    now = time.time()

    if (
        gesture
        and not gesture_triggered
        and now - last_gesture_time
        >= GESTURE_COOLDOWN
    ):
        current_filter_index += 1

        if (
            current_filter_index
            >= len(FILTERS)
        ):
            current_filter_index = 0

        current_filter = FILTERS[
            current_filter_index
        ]

        gesture_triggered = True

        last_gesture_time = now

    if not gesture:
        gesture_triggered = False


def get_portal_points(hands):
    if len(hands) < 2:
        return []

    ordered = sorted(
        hands,
        key=lambda hand: hand["wrist"][0]
    )

    return [
        tuple(
            np.round(
                ordered[0]["thumb"]
            ).astype(int)
        ),
        tuple(
            np.round(
                ordered[0]["index"]
            ).astype(int)
        ),
        tuple(
            np.round(
                ordered[1]["thumb"]
            ).astype(int)
        ),
        tuple(
            np.round(
                ordered[1]["index"]
            ).astype(int)
        )
    ]


def create_portal_mask(
    frame,
    points
):
    if len(points) < 4:
        return None

    height, width = frame.shape[:2]

    pts = np.array(
        points,
        dtype=np.float32
    )

    pts[:, 0] = np.clip(
        pts[:, 0],
        0,
        width - 1
    )

    pts[:, 1] = np.clip(
        pts[:, 1],
        0,
        height - 1
    )

    hull = cv2.convexHull(
        pts.astype(np.int32)
    )

    if (
        cv2.contourArea(hull)
        < 1000
    ):
        return None

    mask = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        mask,
        hull,
        255
    )

    mask = cv2.GaussianBlur(
        mask,
        (7, 7),
        0
    )

    return mask, hull


def apply_portal_filter(
    frame,
    filter_name,
    points
):
    portal_data = create_portal_mask(
        frame,
        points
    )

    if portal_data is None:
        return frame.copy(), None

    mask, hull = portal_data

    filtered = apply_filter(
        frame,
        filter_name
    )

    alpha = (
        mask.astype(
            np.float32
        ) / 255.0
    )

    alpha = alpha[:, :, None]

    result = (
        frame.astype(np.float32)
        * (1.0 - alpha)
        + filtered.astype(np.float32)
        * alpha
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return result, hull


def draw_hand_points(
    frame,
    hands
):
    result = frame.copy()

    for hand in hands:
        cv2.circle(
            result,
            (
                int(hand["thumb"][0]),
                int(hand["thumb"][1])
            ),
            6,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            result,
            (
                int(hand["index"][0]),
                int(hand["index"][1])
            ),
            6,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            result,
            (
                int(hand["pinky"][0]),
                int(hand["pinky"][1])
            ),
            5,
            (0, 200, 255),
            -1,
            cv2.LINE_AA
        )

    return result


def draw_portal_effect(
    frame,
    hull
):
    if hull is None:
        return frame

    result = frame.copy()

    glow = np.zeros_like(
        result
    )

    cv2.polylines(
        glow,
        [hull],
        True,
        (0, 220, 255),
        14,
        cv2.LINE_AA
    )

    glow = cv2.GaussianBlur(
        glow,
        (0, 0),
        8
    )

    result = cv2.addWeighted(
        result,
        1.0,
        glow,
        0.75,
        0
    )

    cv2.polylines(
        result,
        [hull],
        True,
        (255, 255, 255),
        4,
        cv2.LINE_AA
    )

    return result


def draw_portal_particles(
    frame,
    hull
):
    if hull is None:
        return frame

    result = frame.copy()

    contour = hull.reshape(
        -1,
        2
    )

    if len(contour) < 3:
        return result

    now = time.time()

    for i in range(30):
        edge_index = (
            i % len(contour)
        )

        p1 = contour[
            edge_index
        ]

        p2 = contour[
            (edge_index + 1)
            % len(contour)
        ]

        t = (
            now * 0.7
            + i * 0.073
        ) % 1.0

        x = int(
            p1[0] * (1.0 - t)
            + p2[0] * t
        )

        y = int(
            p1[1] * (1.0 - t)
            + p2[1] * t
        )

        x += int(
            np.sin(
                now * 4.0 + i
            ) * 5
        )

        y += int(
            np.cos(
                now * 4.0 + i
            ) * 5
        )

        cv2.circle(
            result,
            (x, y),
            2,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

    return result


def draw_ui(
    frame,
    hands,
    portal_active
):
    result = frame.copy()

    height, width = result.shape[:2]

    overlay = result.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (285, 108),
        (0, 0, 0),
        -1
    )

    result = cv2.addWeighted(
        overlay,
        0.45,
        result,
        0.55,
        0
    )

    cv2.putText(
        result,
        f"PORTAL: {current_filter}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        f"HANDS: {len(hands)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    status = (
        "ACTIVE"
        if portal_active
        else "WAITING"
    )

    cv2.putText(
        result,
        f"PORTAL: {status}",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        (
            width - 110,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        "1 tangan: Jempol + Kelingking = Ganti Filter",
        (
            20,
            height - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return result


def detector_worker():
    global latest_frame
    global latest_detected_hands
    global latest_detection_id
    global detector_running

    print(
        "Loading hand model..."
    )

    model = YOLO(
        HAND_MODEL
    )

    print(
        "Hand detector ready."
    )

    while detector_running:
        with detector_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        try:
            results = model.predict(
                frame,
                imgsz=HAND_IMGSZ,
                conf=HAND_CONF,
                iou=0.45,
                max_det=2,
                verbose=False,
                device="cpu"
            )

            if results:
                detected = get_hand_data(
                    results[0]
                )
            else:
                detected = []

            with detector_lock:
                latest_detected_hands = (
                    detected
                )

                latest_detection_id += 1

        except Exception:
            time.sleep(0.02)


if not os.path.exists(
    HAND_MODEL
):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {HAND_MODEL}"
    )

if not os.path.exists(
    SEG_MODEL
):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {SEG_MODEL}"
    )

print(
    "Starting hand detector..."
)

detector_thread = threading.Thread(
    target=detector_worker,
    daemon=True
)

detector_thread.start()

print(
    "Opening camera..."
)

camera = cv2.VideoCapture(
    0
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    30
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not camera.isOpened():
    detector_running = False

    raise RuntimeError(
        "Camera tidak bisa dibuka."
    )

print(
    "Camera ready."
)

print(
    "Camera mode: FLIPPED VERTICAL / NON-MIRROR"
)

print(
    "Tekan Q untuk keluar."
)

previous_gray = None
last_detection_id = 0

while True:
    success, frame = camera.read()

    if not success:
        continue

    frame_count += 1

    frame = cv2.flip(
        frame,
        0
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    with detector_lock:
        latest_frame = frame.copy()

        detected_hands = [
            copy_hand(hand)
            for hand in latest_detected_hands
        ]

        detection_id = (
            latest_detection_id
        )

    if last_hands:
        tracked_hands = track_hands(
            previous_gray,
            gray,
            last_hands
        )
    else:
        tracked_hands = []

    if (
        detection_id
        != last_detection_id
    ):
        last_detection_id = (
            detection_id
        )

        if detected_hands:
            if tracked_hands:
                last_hands = match_hands(
                    tracked_hands,
                    detected_hands
                )
            else:
                last_hands = [
                    copy_hand(hand)
                    for hand in detected_hands
                ]

            hand_lost_frames = 0

        else:
            hand_lost_frames += 1

    else:
        if tracked_hands:
            last_hands = tracked_hands

        if not last_hands:
            hand_lost_frames += 1
        else:
            hand_lost_frames = 0

    if (
        len(last_hands) > 0
    ):
        update_filter(
            last_hands
        )

    if len(last_hands) >= 2:
        hand_memory = [
            copy_hand(hand)
            for hand in last_hands
        ]

        last_hand_points = (
            get_portal_points(
                last_hands
            )
        )

    elif (
        len(hand_memory) >= 2
        and hand_lost_frames
        <= HAND_MEMORY_FRAMES
    ):
        last_hand_points = (
            get_portal_points(
                hand_memory
            )
        )

    else:
        last_hand_points = []

        if (
            hand_lost_frames
            > HAND_MEMORY_FRAMES
        ):
            hand_memory = []

    previous_gray = gray

    if current_filter == "GALAXY":
        if seg_model is None:
            print(
                "Loading segmentation model..."
            )

            seg_model = YOLO(
                SEG_MODEL
            )

            print(
                "Segmentation model ready."
            )

        if (
            last_segmentation is None
            or frame_count % 15 == 0
        ):
            try:
                segmentation_results = (
                    seg_model.predict(
                        frame,
                        imgsz=SEG_IMGSZ,
                        conf=SEG_CONF,
                        verbose=False,
                        device="cpu"
                    )
                )

                if segmentation_results:
                    last_segmentation = (
                        segmentation_results[0]
                    )

            except Exception:
                pass

        base_frame = (
            apply_galaxy_segmentation(
                frame,
                last_segmentation
            )
        )

    else:
        base_frame = frame.copy()

        last_segmentation = None

    portal_active = False
    portal_hull = None

    if len(last_hand_points) >= 4:
        portal_frame, portal_hull = (
            apply_portal_filter(
                frame,
                current_filter,
                last_hand_points
            )
        )

        if portal_hull is not None:
            portal_active = True

            filtered_frame = (
                draw_portal_effect(
                    portal_frame,
                    portal_hull
                )
            )

            filtered_frame = (
                draw_portal_particles(
                    filtered_frame,
                    portal_hull
                )
            )

        else:
            filtered_frame = base_frame

    else:
        filtered_frame = base_frame

    filtered_frame = draw_hand_points(
        filtered_frame,
        last_hands
    )

    fps_counter += 1

    current_time = time.time()

    elapsed = (
        current_time
        - fps_timer
    )

    if elapsed >= 1.0:
        fps = (
            fps_counter
            / elapsed
        )

        fps_counter = 0
        fps_timer = current_time

    output = draw_ui(
        filtered_frame,
        last_hands,
        portal_active
    )

    cv2.imshow(
        "Retrolens - Hand Tracking Filter & Portal",
        output
    )

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    if key == ord("q"):
        break

detector_running = False

camera.release()

cv2.destroyAllWindows()