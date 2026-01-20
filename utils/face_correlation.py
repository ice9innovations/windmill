#!/usr/bin/env python3
"""
Face Correlation Utility
Correlates NudeNet face detections (with gender) with face service detections (with keypoints)
to create higher-confidence, gender-attributed face detections.
"""
import math
from utils.spatial_analysis import calculate_bbox_iou, normalize_bbox


def fix_negative_bbox(bbox):
    """
    Fix bboxes with negative dimensions (data quality issue from NudeNet).
    Negative width/height means the bbox coordinates were swapped.
    """
    bbox = normalize_bbox(bbox)

    x, y = bbox['x'], bbox['y']
    w, h = bbox['width'], bbox['height']

    # Fix negative width
    if w < 0:
        x = x + w  # Move x left by |w|
        w = abs(w)

    # Fix negative height
    if h < 0:
        y = y + h  # Move y up by |h|
        h = abs(h)

    return {'x': x, 'y': y, 'width': w, 'height': h}


def calculate_center_distance(bbox1, bbox2):
    """Calculate Euclidean distance between bbox centers."""
    bbox1 = fix_negative_bbox(bbox1)
    bbox2 = fix_negative_bbox(bbox2)

    center1_x = bbox1['x'] + bbox1['width'] / 2
    center1_y = bbox1['y'] + bbox1['height'] / 2

    center2_x = bbox2['x'] + bbox2['width'] / 2
    center2_y = bbox2['y'] + bbox2['height'] / 2

    return math.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)


def deduplicate_face_detections(faces, iou_threshold=0.7):
    """
    Deduplicate face detections that likely represent the same face.

    The face service sometimes returns multiple detections for the same face
    with slightly different bboxes. This merges them, keeping the highest
    confidence detection.

    Args:
        faces: list of face dicts with {bbox, confidence, ...}
        iou_threshold: IoU above which faces are considered duplicates

    Returns:
        list: deduplicated faces, keeping highest confidence for each group
    """
    if not faces:
        return []

    # Sort by confidence descending - we'll keep higher confidence ones
    sorted_faces = sorted(faces, key=lambda f: f.get('confidence', 0), reverse=True)

    kept_faces = []
    used_indices = set()

    for i, face in enumerate(sorted_faces):
        if i in used_indices:
            continue

        face_bbox = fix_negative_bbox(face['bbox'])

        # Mark any overlapping faces as duplicates
        for j, other_face in enumerate(sorted_faces[i+1:], i+1):
            if j in used_indices:
                continue

            other_bbox = fix_negative_bbox(other_face['bbox'])
            iou = calculate_bbox_iou(face_bbox, other_bbox)

            if iou >= iou_threshold:
                # This is a duplicate - mark it as used
                used_indices.add(j)

        kept_faces.append(face)
        used_indices.add(i)

    return kept_faces


def correlate_faces(nudenet_faces, face_service_faces, iou_threshold=0.2, center_distance_threshold=50):
    """
    Correlate face detections from NudeNet and face service.

    NudeNet provides: bbox, gender (FACE_FEMALE/FACE_MALE), confidence
    Face service provides: bbox, keypoints, confidence (no gender)

    When both detect the same face (IoU > threshold), we combine:
    - Gender from NudeNet
    - Keypoints from face service
    - Higher confidence from voting

    Args:
        nudenet_faces: list of dicts with {label, bbox, confidence}
        face_service_faces: list of dicts with {bbox, keypoints, confidence, ...}
        iou_threshold: minimum IoU to consider same face (default 0.2)
        center_distance_threshold: max pixel distance between centers (fallback when IoU low)

    Returns:
        dict: {
            'correlated_faces': list of faces detected by BOTH services,
            'nudenet_only_faces': list of faces detected only by NudeNet,
            'face_service_only_faces': list of faces detected only by face service,
            'total_faces': deduplicated face count,
            'correlation_stats': summary statistics
        }
    """
    # Deduplicate face service detections first (it sometimes returns duplicates)
    face_service_faces = deduplicate_face_detections(face_service_faces, iou_threshold=0.7)

    correlated_faces = []
    used_nudenet_indices = set()
    used_face_service_indices = set()

    # Find correlations
    for nn_idx, nn_face in enumerate(nudenet_faces):
        nn_bbox = fix_negative_bbox(nn_face['bbox'])
        nn_gender = extract_gender_from_label(nn_face['label'])

        best_match = None
        best_match_idx = None
        best_iou = 0
        best_center_dist = float('inf')

        for fs_idx, fs_face in enumerate(face_service_faces):
            if fs_idx in used_face_service_indices:
                continue

            fs_bbox = fix_negative_bbox(fs_face['bbox'])
            iou = calculate_bbox_iou(nn_bbox, fs_bbox)
            center_dist = calculate_center_distance(nn_bbox, fs_bbox)

            # Match if IoU is good OR if centers are close (fallback for different box sizes)
            is_match = iou >= iou_threshold or center_dist < center_distance_threshold

            if is_match:
                # Prefer higher IoU, then lower center distance
                if iou > best_iou or (iou == best_iou and center_dist < best_center_dist):
                    best_match = fs_face
                    best_match_idx = fs_idx
                    best_iou = iou
                    best_center_dist = center_dist

        if best_match is not None:
            fs_bbox = fix_negative_bbox(best_match['bbox'])
            correlated_face = {
                'face_id': len(correlated_faces),
                'source': 'both',
                'gender': nn_gender,
                'gender_source': 'nudenet',
                'bbox': {
                    'nudenet': nn_face['bbox'],
                    'face_service': best_match['bbox'],
                    'averaged': average_bboxes(nn_bbox, fs_bbox)
                },
                'keypoints': best_match.get('keypoints'),
                'confidence': {
                    'nudenet': nn_face.get('confidence', 0.0),
                    'face_service': best_match.get('confidence', 0.0),
                    'combined': calculate_combined_confidence(
                        nn_face.get('confidence', 0.0),
                        best_match.get('confidence', 0.0)
                    )
                },
                'iou': best_iou,
                'center_distance': best_center_dist,
                'vote_count': 2,
                'correlation_strength': 'strong' if best_iou > 0.5 else ('moderate' if best_iou > 0.2 else 'weak')
            }
            correlated_faces.append(correlated_face)
            used_nudenet_indices.add(nn_idx)
            used_face_service_indices.add(best_match_idx)

    # Collect unmatched faces
    nudenet_only_faces = []
    for nn_idx, nn_face in enumerate(nudenet_faces):
        if nn_idx not in used_nudenet_indices:
            nn_gender = extract_gender_from_label(nn_face['label'])
            nudenet_only_faces.append({
                'face_id': len(correlated_faces) + len(nudenet_only_faces),
                'source': 'nudenet_only',
                'gender': nn_gender,
                'gender_source': 'nudenet',
                'bbox': {
                    'nudenet': nn_face['bbox'],
                    'face_service': None,
                    'averaged': nn_face['bbox']
                },
                'keypoints': None,
                'confidence': {
                    'nudenet': nn_face.get('confidence', 0.0),
                    'face_service': None,
                    'combined': nn_face.get('confidence', 0.0)
                },
                'iou': None,
                'vote_count': 1,
                'correlation_strength': 'single_source'
            })

    face_service_only_faces = []
    for fs_idx, fs_face in enumerate(face_service_faces):
        if fs_idx not in used_face_service_indices:
            face_service_only_faces.append({
                'face_id': len(correlated_faces) + len(nudenet_only_faces) + len(face_service_only_faces),
                'source': 'face_service_only',
                'gender': 'unknown',  # Face service doesn't provide gender
                'gender_source': None,
                'bbox': {
                    'nudenet': None,
                    'face_service': fs_face['bbox'],
                    'averaged': fs_face['bbox']
                },
                'keypoints': fs_face.get('keypoints'),
                'confidence': {
                    'nudenet': None,
                    'face_service': fs_face.get('confidence', 0.0),
                    'combined': fs_face.get('confidence', 0.0)
                },
                'iou': None,
                'vote_count': 1,
                'correlation_strength': 'single_source'
            })

    # Calculate statistics
    total_faces = len(correlated_faces) + len(nudenet_only_faces) + len(face_service_only_faces)

    stats = {
        'total_faces': total_faces,
        'correlated_count': len(correlated_faces),
        'nudenet_only_count': len(nudenet_only_faces),
        'face_service_only_count': len(face_service_only_faces),
        'gender_breakdown': {
            'female': sum(1 for f in correlated_faces + nudenet_only_faces if f['gender'] == 'female'),
            'male': sum(1 for f in correlated_faces + nudenet_only_faces if f['gender'] == 'male'),
            'unknown': len(face_service_only_faces)  # Only face service doesn't know gender
        },
        'average_iou': (
            sum(f['iou'] for f in correlated_faces) / len(correlated_faces)
            if correlated_faces else 0.0
        ),
        'high_confidence_faces': sum(
            1 for f in correlated_faces + nudenet_only_faces + face_service_only_faces
            if f['confidence']['combined'] > 0.7
        )
    }

    return {
        'correlated_faces': correlated_faces,
        'nudenet_only_faces': nudenet_only_faces,
        'face_service_only_faces': face_service_only_faces,
        'all_faces': correlated_faces + nudenet_only_faces + face_service_only_faces,
        'total_faces': total_faces,
        'correlation_stats': stats
    }


def extract_gender_from_label(label):
    """
    Extract gender from NudeNet face label.

    Args:
        label: NudeNet label (e.g., 'FACE_FEMALE', 'FACE_MALE')

    Returns:
        str: 'female', 'male', or 'unknown'
    """
    if 'FEMALE' in label.upper():
        return 'female'
    elif 'MALE' in label.upper():
        return 'male'
    return 'unknown'


def average_bboxes(bbox1, bbox2):
    """
    Average two bboxes to get a combined bbox.

    Args:
        bbox1, bbox2: normalized bboxes {x, y, width, height}

    Returns:
        list: [x, y, width, height] averaged
    """
    return [
        int((bbox1['x'] + bbox2['x']) / 2),
        int((bbox1['y'] + bbox2['y']) / 2),
        int((bbox1['width'] + bbox2['width']) / 2),
        int((bbox1['height'] + bbox2['height']) / 2)
    ]


def calculate_combined_confidence(conf1, conf2):
    """
    Calculate combined confidence from two detections.

    When two independent detectors agree, confidence increases.
    Uses noisy-OR combination: 1 - (1-p1)(1-p2)

    Args:
        conf1, conf2: individual confidence scores (0-1)

    Returns:
        float: combined confidence score
    """
    if conf1 is None or conf2 is None:
        return conf1 or conf2 or 0.0

    # Noisy-OR: probability that at least one is correct
    # This increases confidence when both agree
    return 1 - (1 - conf1) * (1 - conf2)


def get_face_gender_attribution(face_correlations, person_bbox):
    """
    Get gender attribution for a person bbox based on face correlation.

    Finds the face (if any) that overlaps with the person bbox
    and returns its gender with confidence.

    Args:
        face_correlations: output from correlate_faces()
        person_bbox: dict with bbox {x, y, width, height} or list [x, y, w, h]

    Returns:
        dict: {
            'gender': str,
            'confidence': float,
            'face_id': int or None,
            'source': str
        }
    """
    person_bbox_norm = normalize_bbox(person_bbox)
    best_match = None
    best_iou = 0.0

    for face in face_correlations.get('all_faces', []):
        # Get the best available bbox for this face
        face_bbox = face['bbox'].get('averaged') or face['bbox'].get('nudenet') or face['bbox'].get('face_service')
        if not face_bbox:
            continue

        face_bbox_norm = normalize_bbox(face_bbox)
        iou = calculate_bbox_iou(person_bbox_norm, face_bbox_norm)

        # Face should be contained within or overlap significantly with person
        if iou > best_iou:
            best_iou = iou
            best_match = face

    if best_match and best_iou > 0.1:  # Low threshold since face is inside person bbox
        return {
            'gender': best_match['gender'],
            'confidence': best_match['confidence']['combined'],
            'face_id': best_match['face_id'],
            'source': best_match['source'],
            'overlap_iou': best_iou
        }

    return {
        'gender': 'unknown',
        'confidence': 0.0,
        'face_id': None,
        'source': None,
        'overlap_iou': 0.0
    }
