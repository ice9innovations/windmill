#!/usr/bin/env python3
"""
Framing Analysis Utilities
Calculate bbox sizes relative to image dimensions for shot composition analysis
"""


def calculate_bbox_percentage(bbox, image_width, image_height):
    """
    Calculate what percentage of the image a bbox occupies.

    Args:
        bbox: list [x, y, width, height] or dict {x, y, width, height}
        image_width: int
        image_height: int

    Returns:
        float: percentage of image area (0-100)
    """
    if isinstance(bbox, list):
        x, y, w, h = bbox
    else:
        x = bbox.get('x', 0)
        y = bbox.get('y', 0)
        w = bbox.get('width', 0)
        h = bbox.get('height', 0)

    bbox_area = w * h
    image_area = image_width * image_height

    if image_area == 0:
        return 0.0

    return (bbox_area / image_area) * 100


def classify_framing(anatomy_bboxes, person_bboxes, image_width, image_height):
    """
    Classify shot framing based on bbox sizes relative to image dimensions.

    Args:
        anatomy_bboxes: list of anatomy detection dicts {label, bbox, confidence}
        person_bboxes: list of person bbox dicts {id, bbox}
        image_width: int
        image_height: int

    Returns:
        dict: {
            'framing_type': str,
            'bbox_percentages': list of dicts,
            'largest_anatomy_bbox': dict,
            'reasoning': str
        }
    """
    if image_width == 0 or image_height == 0:
        return {
            'framing_type': 'unknown',
            'bbox_percentages': [],
            'largest_anatomy_bbox': None,
            'reasoning': 'no_image_dimensions'
        }

    # Calculate percentages for all anatomy bboxes
    bbox_percentages = []
    for detection in anatomy_bboxes:
        percentage = calculate_bbox_percentage(
            detection['bbox'],
            image_width,
            image_height
        )
        bbox_percentages.append({
            'label': detection['label'],
            'percentage': round(percentage, 2),
            'bbox': detection['bbox']
        })

    # Find largest anatomy bbox
    largest_anatomy = None
    max_percentage = 0
    for bp in bbox_percentages:
        if bp['percentage'] > max_percentage:
            max_percentage = bp['percentage']
            largest_anatomy = bp

    # Calculate person bbox percentages
    person_percentages = []
    for person in person_bboxes:
        percentage = calculate_bbox_percentage(
            person['bbox'],
            image_width,
            image_height
        )
        person_percentages.append({
            'person_id': person['id'],
            'percentage': round(percentage, 2)
        })

    # Determine framing type
    framing_type = 'unknown'
    reasoning = ''

    # Check for face presence
    has_face = any('FACE' in bp['label'] for bp in bbox_percentages)

    # Genital bboxes
    genital_bboxes = [bp for bp in bbox_percentages if 'GENITALIA' in bp['label']]
    max_genital_pct = max([bp['percentage'] for bp in genital_bboxes], default=0)

    if max_genital_pct > 40 and not has_face:
        framing_type = 'extreme_closeup'
        reasoning = f'Genital bbox occupies {max_genital_pct:.1f}% of frame with no face detected'

    elif max_genital_pct > 25 and not has_face:
        framing_type = 'closeup'
        reasoning = f'Genital bbox occupies {max_genital_pct:.1f}% of frame with no face'

    elif max_percentage > 30 and not has_face:
        framing_type = 'closeup'
        reasoning = f'Anatomy bbox occupies {max_percentage:.1f}% of frame with no face'

    elif person_percentages and max([p['percentage'] for p in person_percentages]) > 50:
        framing_type = 'full_body'
        max_person_pct = max([p['percentage'] for p in person_percentages])
        reasoning = f'Person bbox occupies {max_person_pct:.1f}% of frame'

    elif len(person_percentages) >= 2:
        framing_type = 'wide_shot'
        reasoning = f'Multiple people detected ({len(person_percentages)} people)'

    elif has_face and max_percentage < 20:
        framing_type = 'medium_shot'
        reasoning = 'Face detected with moderate anatomy bbox sizes'

    else:
        framing_type = 'medium_shot'
        reasoning = 'Default classification based on bbox distribution'

    return {
        'framing_type': framing_type,
        'bbox_percentages': bbox_percentages,
        'person_percentages': person_percentages,
        'largest_anatomy_bbox': largest_anatomy,
        'max_anatomy_percentage': round(max_percentage, 2),
        'max_genital_percentage': round(max_genital_pct, 2),
        'has_face': has_face,
        'reasoning': reasoning,
        'image_dimensions': {
            'width': image_width,
            'height': image_height
        }
    }
