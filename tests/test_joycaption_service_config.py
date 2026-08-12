import os

from core.dispatch import resolve_services
from workers.service_config import ServiceConfig


def test_joycaption_is_premium_extra_vlm():
    config = ServiceConfig(os.path.join(os.path.dirname(__file__), "..", "service_config.yaml"))

    joycaption = config.get_service_config("primary.joycaption")

    assert joycaption["queue_name"] == "joycaption"
    assert joycaption["port"] == 7798
    assert joycaption["endpoint"] == "/analyze"
    assert config.is_vlm_service("primary.joycaption")
    assert config.should_trigger_consensus("primary.joycaption")
    assert "joycaption" in config.get_vlm_service_names()
    assert "joycaption" in resolve_services("premium", config)
    assert "joycaption" in resolve_services("extra", config)
    assert "joycaption" not in resolve_services("basic", config)
