from __future__ import annotations

from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageDraw

from adetailer import PredictOutput
from adetailer.common import create_bbox_from_mask, create_mask_from_bbox

# Check if mediapipe is available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[ADetailer] MediaPipe not available, using InsightFace as fallback")


def mediapipe_predict(
    model_type: str, image: Image.Image, confidence: float = 0.3
) -> PredictOutput:
    mapping = {
        "mediapipe_face_short": partial(mediapipe_face_detection, 0),
        "mediapipe_face_full": partial(mediapipe_face_detection, 1),
        "mediapipe_face_mesh": mediapipe_face_mesh,
        "mediapipe_face_mesh_eyes_only": mediapipe_face_mesh_eyes_only,
    }
    if model_type in mapping:
        func = mapping[model_type]
        try:
            return func(image, confidence)
        except Exception:
            return PredictOutput()
    msg = f"[-] ADetailer: Invalid mediapipe model type: {model_type}, Available: {list(mapping.keys())!r}"
    raise RuntimeError(msg)


def mediapipe_face_detection(
    model_type: int, image: Image.Image, confidence: float = 0.3
) -> PredictOutput[float]:
    if not MEDIAPIPE_AVAILABLE:
        return insightface_face_detection(image, confidence)
    
    import mediapipe as mp

    img_width, img_height = image.size

    mp_face_detection = mp.solutions.face_detection
    draw_util = mp.solutions.drawing_utils

    img_array = np.array(image)

    with mp_face_detection.FaceDetection(
        model_selection=model_type, min_detection_confidence=confidence
    ) as face_detector:
        pred = face_detector.process(img_array)

    if pred.detections is None:
        return PredictOutput()

    preview_array = img_array.copy()

    bboxes = []
    confidences = []
    for detection in pred.detections:
        draw_util.draw_detection(preview_array, detection)

        bbox = detection.location_data.relative_bounding_box
        x1 = bbox.xmin * img_width
        y1 = bbox.ymin * img_height
        w = bbox.width * img_width
        h = bbox.height * img_height
        x2 = x1 + w
        y2 = y1 + h

        confidences.append(detection.score)
        bboxes.append([x1, y1, x2, y2])

    masks = create_mask_from_bbox(bboxes, image.size)
    preview = Image.fromarray(preview_array)

    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def mediapipe_face_mesh(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    if not MEDIAPIPE_AVAILABLE:
        return insightface_face_mesh(image, confidence)
    
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    draw_util = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    w, h = image.size

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=20, min_detection_confidence=confidence
    ) as face_mesh:
        arr = np.array(image)
        pred = face_mesh.process(arr)

        if pred.multi_face_landmarks is None:
            return PredictOutput()

        preview = arr.copy()
        masks = []
        confidences = []

        for landmarks in pred.multi_face_landmarks:
            draw_util.draw_landmarks(
                image=preview,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            points = np.array(
                [[land.x * w, land.y * h] for land in landmarks.landmark], dtype=int
            )
            outline = cv2.convexHull(points).reshape(-1).tolist()

            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            draw.polygon(outline, fill="white")
            masks.append(mask)
            confidences.append(1.0)  # Confidence is unknown

        bboxes = create_bbox_from_mask(masks, image.size)
        preview = Image.fromarray(preview)
        return PredictOutput(
            bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
        )


def mediapipe_face_mesh_eyes_only(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    if not MEDIAPIPE_AVAILABLE:
        return insightface_face_mesh_eyes_only(image, confidence)
    
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh

    left_idx = np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
    right_idx = np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()

    w, h = image.size

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=20, min_detection_confidence=confidence
    ) as face_mesh:
        arr = np.array(image)
        pred = face_mesh.process(arr)

        if pred.multi_face_landmarks is None:
            return PredictOutput()

        preview = image.copy()
        masks = []
        confidences = []

        for landmarks in pred.multi_face_landmarks:
            points = np.array(
                [[land.x * w, land.y * h] for land in landmarks.landmark], dtype=int
            )
            left_eyes = points[left_idx]
            right_eyes = points[right_idx]
            left_outline = cv2.convexHull(left_eyes).reshape(-1).tolist()
            right_outline = cv2.convexHull(right_eyes).reshape(-1).tolist()

            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            for outline in (left_outline, right_outline):
                draw.polygon(outline, fill="white")
            masks.append(mask)
            confidences.append(1.0)  # Confidence is unknown

        bboxes = create_bbox_from_mask(masks, image.size)
        preview = draw_preview(preview, bboxes, masks)
        return PredictOutput(
            bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
        )


def draw_preview(
    preview: Image.Image, bboxes: list[list[int]], masks: list[Image.Image]
) -> Image.Image:
    red = Image.new("RGB", preview.size, "red")
    for mask in masks:
        masked = Image.composite(red, preview, mask)
        preview = Image.blend(preview, masked, 0.25)

    draw = ImageDraw.Draw(preview)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="red", width=2)

    return preview


# InsightFace fallback implementations
_insightface_app = None


def get_insightface_app():
    """Lazy load InsightFace app."""
    global _insightface_app
    if _insightface_app is None:
        try:
            from insightface.app import FaceAnalysis
            # Try GPU first, fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            _insightface_app = FaceAnalysis(name='buffalo_l', providers=providers)
            _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            print("[ADetailer] InsightFace initialized successfully (used as MediaPipe replacement)")
        except Exception as e:
            print(f"[ADetailer] Failed to initialize InsightFace: {e}")
            print("[ADetailer] Tip: For better results, use YOLO models (face_yolov8n.pt) instead of mediapipe_face_* models")
            _insightface_app = False
    return _insightface_app if _insightface_app is not False else None


def insightface_face_detection(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[float]:
    """InsightFace-based face detection as MediaPipe replacement."""
    app = get_insightface_app()
    if app is None:
        return PredictOutput()
    
    img_array = np.array(image)
    faces = app.get(img_array)
    
    # Filter by confidence
    faces = [f for f in faces if f.det_score >= confidence]
    
    if not faces:
        return PredictOutput()
    
    bboxes = []
    confidences = []
    preview_array = img_array.copy()
    
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        bboxes.append([float(x1), float(y1), float(x2), float(y2)])
        confidences.append(float(face.det_score))
        
        # Draw bounding box
        cv2.rectangle(preview_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks if available
        if hasattr(face, 'kps') and face.kps is not None:
            for kp in face.kps.astype(int):
                cv2.circle(preview_array, tuple(kp), 2, (255, 0, 0), -1)
    
    masks = create_mask_from_bbox(bboxes, image.size)
    preview = Image.fromarray(preview_array)
    
    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def insightface_face_mesh(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    """InsightFace-based face mesh as MediaPipe replacement."""
    app = get_insightface_app()
    if app is None:
        return PredictOutput()
    
    img_array = np.array(image)
    faces = app.get(img_array)
    
    # Filter by confidence
    faces = [f for f in faces if f.det_score >= confidence]
    
    if not faces:
        return PredictOutput()
    
    w, h = image.size
    preview = image.copy()
    masks = []
    confidences_list = []
    
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Create face mask from bbox (approximate face mesh with convex hull)
        if hasattr(face, 'kps') and face.kps is not None:
            # Use landmarks to create better face mask
            kps = face.kps.astype(int)
            # Expand landmarks to approximate face mesh
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Create elliptical mask for face
            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            draw.ellipse([x1, y1, x2, y2], fill="white")
            masks.append(mask)
        else:
            # Fallback to rectangular mask
            mask = Image.new("L", image.size, "black")
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill="white")
            masks.append(mask)
        
        confidences_list.append(1.0)
    
    bboxes = create_bbox_from_mask(masks, image.size)
    preview = draw_preview(preview, bboxes, masks)
    
    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences_list, preview=preview
    )


def insightface_face_mesh_eyes_only(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    """InsightFace-based eyes-only detection as MediaPipe replacement."""
    app = get_insightface_app()
    if app is None:
        return PredictOutput()
    
    img_array = np.array(image)
    faces = app.get(img_array)
    
    # Filter by confidence
    faces = [f for f in faces if f.det_score >= confidence]
    
    if not faces:
        return PredictOutput()
    
    w, h = image.size
    preview = image.copy()
    masks = []
    confidences_list = []
    
    for face in faces:
        if not hasattr(face, 'kps') or face.kps is None or len(face.kps) < 2:
            continue
        
        kps = face.kps.astype(int)
        # kps[0] = left_eye, kps[1] = right_eye
        
        # Create eye masks (approximate circular regions around eye landmarks)
        mask = Image.new("L", image.size, "black")
        draw = ImageDraw.Draw(mask)
        
        eye_radius = 30  # Approximate eye region radius
        
        # Left eye
        left_eye = kps[0]
        draw.ellipse([
            left_eye[0] - eye_radius, left_eye[1] - eye_radius,
            left_eye[0] + eye_radius, left_eye[1] + eye_radius
        ], fill="white")
        
        # Right eye
        right_eye = kps[1]
        draw.ellipse([
            right_eye[0] - eye_radius, right_eye[1] - eye_radius,
            right_eye[0] + eye_radius, right_eye[1] + eye_radius
        ], fill="white")
        
        masks.append(mask)
        confidences_list.append(1.0)
    
    if not masks:
        return PredictOutput()
    
    bboxes = create_bbox_from_mask(masks, image.size)
    preview = draw_preview(preview, bboxes, masks)
    
    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences_list, preview=preview
    )
