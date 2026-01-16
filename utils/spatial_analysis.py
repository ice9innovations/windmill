#!/usr/bin/env python3
"""
Spatial Analysis Utilities
Bounding box operations for activity inference and relationship detection
"""
import math


def normalize_bbox(bbox):
    """
    Normalize bbox to dict format.

    Args:
        bbox: Either list [x, y, w, h] or dict {x, y, width, height}

    Returns:
        dict: {x, y, width, height}
    """
    if isinstance(bbox, list):
        return {
            'x': bbox[0],
            'y': bbox[1],
            'width': bbox[2],
            'height': bbox[3]
        }
    elif isinstance(bbox, dict):
        # Already in dict format, but ensure correct keys
        if 'width' in bbox and 'height' in bbox:
            return bbox
        elif 'w' in bbox and 'h' in bbox:
            return {
                'x': bbox.get('x', 0),
                'y': bbox.get('y', 0),
                'width': bbox['w'],
                'height': bbox['h']
            }
    return bbox


def calculate_bbox_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1, bbox2: Either list [x, y, w, h] or dict {x, y, width, height}

    Returns:
        float: IoU score (0.0 to 1.0)
    """
    bbox1 = normalize_bbox(bbox1)
    bbox2 = normalize_bbox(bbox2)

    x1_1, y1_1 = bbox1['x'], bbox1['y']
    x2_1, y2_1 = x1_1 + bbox1['width'], y1_1 + bbox1['height']

    x1_2, y1_2 = bbox2['x'], bbox2['y']
    x2_2, y2_2 = x1_2 + bbox2['width'], y1_2 + bbox2['height']

    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_center_distance(bbox1, bbox2):
    """Calculate Euclidean distance between bbox centers"""
    bbox1 = normalize_bbox(bbox1)
    bbox2 = normalize_bbox(bbox2)

    center1_x = bbox1['x'] + bbox1['width'] / 2
    center1_y = bbox1['y'] + bbox1['height'] / 2

    center2_x = bbox2['x'] + bbox2['width'] / 2
    center2_y = bbox2['y'] + bbox2['height'] / 2

    distance = math.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
    return distance


def bbox_completely_contains(outer_bbox, inner_bbox):
    """
    Check if outer bbox completely contains inner bbox.

    Returns:
        bool: True if inner bbox is completely inside outer bbox
    """
    inner_bbox = normalize_bbox(inner_bbox)
    outer_bbox = normalize_bbox(outer_bbox)

    inner_x1 = inner_bbox['x']
    inner_y1 = inner_bbox['y']
    inner_x2 = inner_x1 + inner_bbox['width']
    inner_y2 = inner_y1 + inner_bbox['height']

    outer_x1 = outer_bbox['x']
    outer_y1 = outer_bbox['y']
    outer_x2 = outer_x1 + outer_bbox['width']
    outer_y2 = outer_y1 + outer_bbox['height']

    return (inner_x1 >= outer_x1 and inner_x2 <= outer_x2 and
            inner_y1 >= outer_y1 and inner_y2 <= outer_y2)


def detect_person_containment(person_bboxes):
    """
    Detect when one person bbox completely contains another.
    Returns containment relationships for deduplication.

    Args:
        person_bboxes: list of person bbox dicts with {id, bbox: {x, y, width, height}}

    Returns:
        list of containment relationship dicts
    """
    containments = []

    for i, bbox1 in enumerate(person_bboxes):
        for j, bbox2 in enumerate(person_bboxes[i+1:], i+1):
            if bbox_completely_contains(bbox2['bbox'], bbox1['bbox']):
                containments.append({
                    'contained_bbox_id': bbox1['id'],
                    'containing_bbox_id': bbox2['id'],
                    'likely_same_person': True,
                    'reason': 'complete_containment',
                    'confidence': 0.95
                })
            elif bbox_completely_contains(bbox1['bbox'], bbox2['bbox']):
                containments.append({
                    'contained_bbox_id': bbox2['id'],
                    'containing_bbox_id': bbox1['id'],
                    'likely_same_person': True,
                    'reason': 'complete_containment',
                    'confidence': 0.95
                })

    return containments


def detect_bbox_overlap(bbox1_data, bbox2_data, iou_threshold=0.2, proximity_threshold=50):
    """
    Detect if two bboxes overlap significantly.

    Args:
        bbox1_data, bbox2_data: dicts with {bbox: {x, y, width, height}, label: str}
        iou_threshold: minimum IoU to consider overlapping
        proximity_threshold: max pixel distance between centers

    Returns:
        dict: {
            'overlaps': bool,
            'iou': float,
            'proximity': float,
            'type': str
        }
    """
    bbox1 = bbox1_data['bbox']
    bbox2 = bbox2_data['bbox']

    iou = calculate_bbox_iou(bbox1, bbox2)
    proximity = calculate_center_distance(bbox1, bbox2)

    overlaps = iou > iou_threshold or proximity < proximity_threshold

    return {
        'overlaps': overlaps,
        'iou': iou,
        'proximity': proximity,
        'bbox1_label': bbox1_data['label'],
        'bbox2_label': bbox2_data['label']
    }


def detect_sexual_activities(anatomy_bboxes, person_bboxes, captions_text):
    """
    Detect sexual activities from bbox spatial relationships.

    Args:
        anatomy_bboxes: list of anatomy detection dicts {label, bbox, confidence}
        person_bboxes: list of person bbox dicts {id, bbox}
        captions_text: combined caption text (lowercase)

    Returns:
        dict: {
            'activities': list of activity strings,
            'spatial_relationships': list of relationship dicts,
            'scene_type': str,
            'intimacy_level': str
        }
    """
    activities = []
    spatial_relationships = []

    # Get all anatomy labels
    anatomy_labels = [d['label'] for d in anatomy_bboxes]

    # Detect genital-to-genital overlap (sexual intercourse)
    # Use startswith() to avoid 'MALE_GENITALIA' matching 'FEMALE_GENITALIA' (substring issue)
    male_genitals = [d for d in anatomy_bboxes if d['label'].startswith('MALE_GENITALIA')]
    female_genitals = [d for d in anatomy_bboxes if d['label'].startswith('FEMALE_GENITALIA')]
    buttocks = [d for d in anatomy_bboxes if 'BUTTOCKS' in d['label']]

    for mg in male_genitals:
        for fg in female_genitals:
            # Use higher proximity threshold (100px) for genital interactions
            # 50px was too strict and missed clear cases of sexual contact
            overlap = detect_bbox_overlap(mg, fg, iou_threshold=0.15, proximity_threshold=100)
            if overlap['overlaps']:
                activities.append('sexual_intercourse')
                activities.append('heterosexual_activity')
                spatial_relationships.append({
                    'type': 'genital_to_genital',
                    'bbox1': mg['label'],
                    'bbox2': fg['label'],
                    'iou': overlap['iou'],
                    'proximity': overlap['proximity']
                })

        # Genital-to-buttocks (anal sex)
        for bt in buttocks:
            # Use higher proximity threshold (100px) for genital interactions
            overlap = detect_bbox_overlap(mg, bt, iou_threshold=0.15, proximity_threshold=100)
            if overlap['overlaps']:
                activities.append('anal_intercourse')
                spatial_relationships.append({
                    'type': 'genital_to_buttocks',
                    'bbox1': mg['label'],
                    'bbox2': bt['label'],
                    'iou': overlap['iou'],
                    'proximity': overlap['proximity']
                })

    # Same-gender genital overlap
    if len(male_genitals) >= 2:
        activities.append('male_homosexual_activity')
        activities.append('same_gender_sexual_activity')
        spatial_relationships.append({'type': 'male_male_genital_overlap'})

    if len(female_genitals) >= 2:
        activities.append('female_homosexual_activity')
        activities.append('same_gender_sexual_activity')
        spatial_relationships.append({'type': 'female_female_genital_overlap'})

    # Face-to-genital overlap (oral sex)
    face_bboxes = [d for d in anatomy_bboxes if 'FACE' in d['label']]
    all_genitals = male_genitals + female_genitals

    for face in face_bboxes:
        for genital in all_genitals:
            overlap = detect_bbox_overlap(face, genital, iou_threshold=0.1, proximity_threshold=50)
            if overlap['overlaps']:
                # Check for breastfeeding exception
                if 'baby' not in captions_text and 'infant' not in captions_text and 'breastfeeding' not in captions_text:
                    activities.append('oral_sex')
                    spatial_relationships.append({
                        'type': 'face_to_genital',
                        'bbox1': face['label'],
                        'bbox2': genital['label'],
                        'iou': overlap['iou'],
                        'proximity': overlap['proximity']
                    })

    # Breastfeeding detection (face-to-breast with baby/infant keywords)
    # NOTE: breast_play detection removed - face-to-breast proximity is natural anatomy
    # and creates false positives. Will re-add when pose service provides hand keypoints
    # to detect actual hand-to-breast contact.
    female_breasts = [d for d in anatomy_bboxes if 'FEMALE_BREAST' in d['label']]
    for face in face_bboxes:
        for breast in female_breasts:
            overlap = detect_bbox_overlap(face, breast, iou_threshold=0.1, proximity_threshold=50)
            if overlap['overlaps']:
                # Only flag breastfeeding if keywords suggest it
                if any(kw in captions_text for kw in ['baby', 'infant', 'breastfeeding', 'nursing']):
                    activities.append('breastfeeding')
                    spatial_relationships.append({
                        'type': 'breastfeeding',
                        'bbox1': face['label'],
                        'bbox2': breast['label']
                    })

    # Threesome detection
    deduplicated_people_count = len(person_bboxes)
    female_count = sum(1 for label in anatomy_labels if 'FEMALE_BREAST' in label or label.startswith('FEMALE_GENITALIA'))
    male_count = sum(1 for label in anatomy_labels if label.startswith('MALE_GENITALIA'))

    if deduplicated_people_count == 3:
        if female_count >= 2 and male_count >= 1:
            activities.append('threesome')
            activities.append('ffm_threesome')
        elif male_count >= 2 and female_count >= 1:
            activities.append('threesome')
            activities.append('mmf_threesome')

    # Group sex
    if deduplicated_people_count >= 4 and (female_count + male_count) >= 4:
        activities.append('group_sex')
        activities.append('orgy')

    # Determine scene type based on anatomical evidence
    # Check for actual exposed anatomy (not just COVERED or FACE)
    exposed_labels = [label for label in anatomy_labels if 'EXPOSED' in label]
    has_exposed_anatomy = len(exposed_labels) > 0
    has_genitalia = any(label.startswith('FEMALE_GENITALIA') or label.startswith('MALE_GENITALIA')
                        for label in exposed_labels)
    has_exposed_breasts = any('BREAST_EXPOSED' in label for label in exposed_labels)
    has_exposed_buttocks = any('BUTTOCKS_EXPOSED' in label for label in exposed_labels)

    intimacy_level = 'solo' if deduplicated_people_count == 1 else 'multiple_people'

    if activities:
        # Has sexual activities detected
        if 'breastfeeding' in activities:
            scene_type = 'breastfeeding'
            intimacy_level = 'non_sexual'
        else:
            # Any sexual activity (except breastfeeding) = sexually explicit
            scene_type = 'sexually_explicit'
            intimacy_level = 'explicit_sexual'
            if deduplicated_people_count >= 3:
                intimacy_level = 'group'
    elif not has_exposed_anatomy:
        # No exposed anatomy detected - not nudity
        scene_type = 'sfw'
        intimacy_level = 'none'
    elif has_genitalia:
        # Exposed genitalia = softcore
        scene_type = 'softcore_pornography'
    else:
        # Exposed breasts/buttocks but no genitalia = simple nudity
        scene_type = 'simple_nudity'

    return {
        'activities': list(set(activities)),  # Deduplicate
        'spatial_relationships': spatial_relationships,
        'scene_type': scene_type,
        'intimacy_level': intimacy_level
    }

