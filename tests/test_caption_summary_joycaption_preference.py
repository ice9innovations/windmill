from workers.caption_summary_worker import (
    _nsfw2_flags_nsfw,
    _nudenet_flags_nsfw,
    _select_preferred_caption_service,
)


def test_select_preferred_caption_service_prefers_joycaption_for_nsfw():
    captions = {
        "blip": "a person in a room",
        "joycaption": "NSFW: a detailed JoyCaption description",
        "qwen": "a portrait",
    }

    preferred_service = _select_preferred_caption_service(captions, nsfw=True)

    assert preferred_service == "joycaption"


def test_select_preferred_caption_service_returns_none_for_sfw():
    captions = {
        "blip": "a person in a room",
        "joycaption": "SFW: a detailed JoyCaption description",
        "qwen": "a portrait",
    }

    preferred_service = _select_preferred_caption_service(captions, nsfw=False)

    assert preferred_service is None


def test_select_preferred_caption_service_returns_none_when_joycaption_missing():
    captions = {
        "blip": "a person in a room",
        "qwen": "a portrait",
    }

    preferred_service = _select_preferred_caption_service(captions, nsfw=True)

    assert preferred_service is None


def test_nsfw_flags_accept_nsfw2_and_nudenet_payloads():
    assert _nsfw2_flags_nsfw({"predictions": [{"nsfw": True}]}) is True
    assert _nsfw2_flags_nsfw({"predictions": [{"nsfw": False}]}) is False

    assert _nudenet_flags_nsfw({
        "predictions": [{"label": "FEMALE_BREAST_EXPOSED", "bbox": [0, 0, 1, 1]}]
    }) is True
    assert _nudenet_flags_nsfw({
        "predictions": [{"label": "FACE_FEMALE", "bbox": [0, 0, 1, 1]}]
    }) is False
