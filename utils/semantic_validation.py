#!/usr/bin/env python3
"""
Semantic Validation Utilities
Keyword extraction and matching for cross-validating spatial detections with VLM captions
"""

# Category-to-keyword mapping for NudeNet+ categories
CATEGORY_KEYWORDS = {
    'FEMALE_BREAST_EXPOSED': {
        'positive': ['topless', 'naked', 'nude', 'bare chest', 'exposed breast', 'shirtless', 'breasts'],
        'negative': ['shirt', 'top', 'dressed', 'clothed', 'bra', 'bikini top', 'covered'],
        'gender_hints': ['woman', 'female', 'girl', 'she', 'her']
    },
    'MALE_BREAST_EXPOSED': {
        'positive': ['shirtless', 'bare chest', 'topless', 'naked torso', 'nude'],
        'negative': ['shirt', 'top', 'dressed', 'covered'],
        'gender_hints': ['man', 'male', 'guy', 'he', 'him', 'his']
    },
    'FEMALE_GENITALIA_EXPOSED': {
        'positive': ['naked', 'nude', 'exposed', 'bottomless', 'vagina', 'vulva', 'genitals'],
        'negative': ['pants', 'underwear', 'panties', 'shorts', 'dressed', 'covered'],
        'gender_hints': ['woman', 'female', 'girl', 'she', 'her']
    },
    'MALE_GENITALIA_EXPOSED': {
        'positive': ['naked', 'nude', 'exposed', 'penis', 'erect', 'erection', 'genitals'],
        'negative': ['pants', 'underwear', 'shorts', 'dressed', 'covered'],
        'gender_hints': ['man', 'male', 'guy', 'he', 'him', 'his']
    },
    'BUTTOCKS_EXPOSED': {
        'positive': ['naked', 'nude', 'bare', 'exposed', 'butt', 'ass', 'buttocks', 'bottom'],
        'negative': ['pants', 'underwear', 'shorts', 'dressed', 'covered'],
        'gender_hints': []  # Can be any gender
    },
    'ANUS_EXPOSED': {
        'positive': ['anus', 'exposed', 'spread', 'bent over'],
        'negative': ['covered', 'clothed'],
        'gender_hints': []
    }
}

# Gender keyword mapping
GENDER_KEYWORDS = {
    'male': ['man', 'male', 'guy', 'he', 'him', 'his', 'boy'],
    'female': ['woman', 'female', 'girl', 'she', 'her', 'lady']
}

# Scene context keywords
ARTISTIC_KEYWORDS = ['portrait', 'art', 'artistic', 'photography', 'studio', 'standing', 'sitting', 'profile', 'pose', 'model']
SOFTCORE_KEYWORDS = ['bed', 'laying', 'sexy', 'seductive', 'bedroom', 'intimate', 'provocative']
SEXUAL_ACTIVITY_KEYWORDS = ['sex', 'intercourse', 'oral', 'penetration', 'fucking', 'sucking']
BREASTFEEDING_KEYWORDS = ['baby', 'infant', 'breastfeeding', 'nursing', 'feeding']


def extract_keywords_from_captions(captions):
    """
    Extract and normalize keywords from VLM captions.

    Args:
        captions: List of caption dictionaries with 'text' and 'service' keys
                  e.g., [{'service': 'blip', 'text': 'a naked woman...'}, ...]

    Returns:
        dict: {
            'combined_text': str,
            'positive_keywords': set,
            'negative_keywords': set,
            'gender_mentions': {'male': bool, 'female': bool},
            'scene_keywords': {'artistic': int, 'softcore': int, 'sexual': int, 'breastfeeding': int}
        }
    """
    combined_text = ' '.join([c.get('text', '').lower() for c in captions])

    # Extract all keywords mentioned
    all_positive = set()
    all_negative = set()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords['positive']:
            if kw in combined_text:
                all_positive.add(kw)
        for kw in keywords['negative']:
            if kw in combined_text:
                all_negative.add(kw)

    # Extract gender mentions
    gender_mentions = {
        'male': any(kw in combined_text for kw in GENDER_KEYWORDS['male']),
        'female': any(kw in combined_text for kw in GENDER_KEYWORDS['female'])
    }

    # Extract scene context keywords
    scene_keywords = {
        'artistic': sum(1 for kw in ARTISTIC_KEYWORDS if kw in combined_text),
        'softcore': sum(1 for kw in SOFTCORE_KEYWORDS if kw in combined_text),
        'sexual': sum(1 for kw in SEXUAL_ACTIVITY_KEYWORDS if kw in combined_text),
        'breastfeeding': sum(1 for kw in BREASTFEEDING_KEYWORDS if kw in combined_text)
    }

    return {
        'combined_text': combined_text,
        'positive_keywords': list(all_positive),
        'negative_keywords': list(all_negative),
        'gender_mentions': gender_mentions,
        'scene_keywords': scene_keywords
    }


def validate_category_with_captions(category, captions):
    """
    Cross-validate a NudeNet category detection with VLM captions.

    Args:
        category: str - NudeNet category label (e.g., 'FEMALE_BREAST_EXPOSED')
        captions: list - VLM caption dictionaries

    Returns:
        dict: {
            'category': str,
            'corroborated': bool,
            'conflicted': bool,
            'positive_hits': list,
            'negative_hits': list,
            'gender_corroboration': bool,
            'confidence': float
        }
    """
    if category not in CATEGORY_KEYWORDS:
        return {
            'category': category,
            'corroborated': False,
            'conflicted': False,
            'positive_hits': [],
            'negative_hits': [],
            'gender_corroboration': False,
            'confidence': 0.0
        }

    keywords = CATEGORY_KEYWORDS[category]
    extracted = extract_keywords_from_captions(captions)
    combined_text = extracted['combined_text']

    # Check positive evidence
    positive_hits = [kw for kw in keywords['positive'] if kw in combined_text]

    # Check negative evidence (conflicts)
    negative_hits = [kw for kw in keywords['negative'] if kw in combined_text]

    # Check gender corroboration
    gender_corroboration = False
    if keywords['gender_hints']:
        # Check if any gender hints match VLM gender mentions
        for hint in keywords['gender_hints']:
            if hint in combined_text:
                gender_corroboration = True
                break

    # Determine corroboration and conflict
    corroborated = len(positive_hits) > 0
    conflicted = len(negative_hits) > 0

    # Calculate confidence modifier
    confidence = 0.0
    if corroborated and not conflicted:
        confidence = min(0.9 + (len(positive_hits) * 0.05), 0.99)
    elif corroborated and conflicted:
        # Both positive and negative - ambiguous
        confidence = 0.5
    elif conflicted and not corroborated:
        # Only negative - strong conflict
        confidence = 0.1

    return {
        'category': category,
        'corroborated': corroborated,
        'conflicted': conflicted,
        'positive_hits': positive_hits,
        'negative_hits': negative_hits,
        'gender_corroboration': gender_corroboration,
        'confidence': confidence
    }


def infer_gender_from_anatomy(anatomy_detections):
    """
    Infer gender from anatomical detections with confidence scoring.
    Spatial evidence (NudeNet) is more reliable than semantic (VLM).

    Args:
        anatomy_detections: list of detection dicts with 'label' key

    Returns:
        dict: {
            'gender': str ('female', 'male', 'mixed', 'unknown'),
            'confidence': float,
            'spatial_markers': int,
            'reasoning': str
        }
    """
    female_markers = [
        'FACE_FEMALE',
        'FEMALE_BREAST_EXPOSED',
        'FEMALE_GENITALIA_EXPOSED'
    ]
    male_markers = [
        'FACE_MALE',
        'MALE_BREAST_EXPOSED',
        'MALE_GENITALIA_EXPOSED'
    ]

    female_count = sum(1 for d in anatomy_detections
                      if d.get('label') in female_markers)
    male_count = sum(1 for d in anatomy_detections
                    if d.get('label') in male_markers)

    if female_count > 0 and male_count == 0:
        # Only female anatomy detected
        confidence = min(0.95 + (female_count * 0.01), 0.99)
        return {
            'gender': 'female',
            'confidence': confidence,
            'spatial_markers': female_count,
            'reasoning': 'spatial_anatomy_only'
        }
    elif male_count > 0 and female_count == 0:
        # Only male anatomy detected
        confidence = min(0.95 + (male_count * 0.01), 0.99)
        return {
            'gender': 'male',
            'confidence': confidence,
            'spatial_markers': male_count,
            'reasoning': 'spatial_anatomy_only'
        }
    elif female_count > 0 and male_count > 0:
        # Both anatomies detected - multiple people or edge case
        return {
            'gender': 'mixed',
            'confidence': 0.8,
            'spatial_markers': female_count + male_count,
            'reasoning': 'both_anatomies_detected'
        }
    else:
        # No gendered anatomy detected
        return {
            'gender': 'unknown',
            'confidence': 0.0,
            'spatial_markers': 0,
            'reasoning': 'no_spatial_evidence'
        }


def detect_gender_hallucination(captions, spatial_gender_inference):
    """
    Detect when VLMs mention genders not supported by spatial evidence.

    Args:
        captions: list of VLM caption dicts
        spatial_gender_inference: dict from infer_gender_from_anatomy()

    Returns:
        list of hallucination dicts
    """
    import re

    hallucinations = []
    spatial_gender = spatial_gender_inference['gender']

    # Don't flag hallucinations if we have no spatial evidence
    if spatial_gender == 'unknown':
        return hallucinations

    # Extract gender mentions from each caption using word boundaries
    for caption in captions:
        service = caption.get('service', 'unknown')
        text = caption.get('text', '').lower()

        # Check for male mentions using word boundaries
        male_mentioned = any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in GENDER_KEYWORDS['male'])
        # Check for female mentions using word boundaries
        female_mentioned = any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in GENDER_KEYWORDS['female'])

        # Male mentioned but no spatial evidence
        if male_mentioned and spatial_gender != 'male' and spatial_gender != 'mixed':
            hallucinations.append({
                'vlm': service,
                'hallucinated_gender': 'male',
                'spatial_evidence': spatial_gender,
                'confidence': 'hallucination_likely',
                'caption_snippet': text[:100]
            })

        # Female mentioned but no spatial evidence
        if female_mentioned and spatial_gender != 'female' and spatial_gender != 'mixed':
            hallucinations.append({
                'vlm': service,
                'hallucinated_gender': 'female',
                'spatial_evidence': spatial_gender,
                'confidence': 'hallucination_likely',
                'caption_snippet': text[:100]
            })

    return hallucinations


def classify_female_solo_nudity(anatomy, captions, context_objects):
    """
    Distinguish artistic nudity from softcore pornography.
    Female-only nudity without sexual activity.

    Args:
        anatomy: list of anatomy labels detected
        captions: combined caption text (lowercase)
        context_objects: list of emoji objects detected

    Returns:
        str: 'artistic_nudity', 'softcore_pornography', or 'simple_nudity'
    """
    artistic_objects = ['🖼', '📷', '🎨']
    softcore_objects = ['🛏', '💤']

    artistic_score = sum(1 for kw in ARTISTIC_KEYWORDS if kw in captions)
    artistic_score += sum(1 for obj in artistic_objects if obj in context_objects)

    softcore_score = sum(1 for kw in SOFTCORE_KEYWORDS if kw in captions)
    softcore_score += sum(1 for obj in softcore_objects if obj in context_objects)

    # Genitalia visibility = strong softcore signal
    if 'FEMALE_GENITALIA_EXPOSED' in anatomy:
        softcore_score += 2
    # Only breast = could be artistic
    elif 'FEMALE_BREAST_EXPOSED' in anatomy:
        artistic_score += 1

    if artistic_score > softcore_score:
        return 'artistic_nudity'
    elif softcore_score > artistic_score:
        return 'softcore_pornography'
    else:
        return 'simple_nudity'
