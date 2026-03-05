"""
Service resolution and downstream dispatch logic.

No Flask dependency. All functions take a ServiceConfig instance as a parameter
so callers can pass their own config path (e.g. ice9-api pointing at a different
service_config.yaml location).
"""


def resolve_services(services_param, tier, config):
    """Resolve a comma-separated service list, or return tier-appropriate primary services.

    Returns (service_names, error_string_or_None).
    On success error is None; on failure service_names is None.
    """
    primary   = config.get_services_by_category('primary')
    available = sorted(name.split('.', 1)[1] for name in primary.keys())

    if not services_param:
        tier_services = config.get_services_by_tier(tier)
        return sorted(
            name.split('.', 1)[1]
            for name in tier_services.keys()
            if name.startswith('primary.')
        ), None

    requested = [s.strip() for s in services_param.split(',')]
    invalid   = [s for s in requested if s not in available]
    if invalid:
        return None, f"Unknown services: {', '.join(invalid)}. Available: {', '.join(available)}"
    return requested, None


def compute_expected_downstream(services_submitted, config):
    """Determine which downstream services are expected based on submitted primary services.

    Returns a dict of {downstream_name: bool} indicating whether each downstream
    service is expected to eventually produce a result for this image.
    Uses service_config so the logic adapts when services change.
    """
    if not services_submitted:
        return {}

    has_consensus_service = any(
        config.should_trigger_consensus(f'primary.{s}')
        for s in services_submitted
    )

    has_spatial_service = any(
        config.is_spatial_service(f'primary.{s}')
        for s in services_submitted
    )

    vlm_services = [
        s for s in services_submitted
        if config.is_vlm_service(f'primary.{s}')
    ]
    has_vlm       = len(vlm_services) > 0
    has_multi_vlm = len(vlm_services) >= 2

    return {
        'consensus':        has_consensus_service,
        'content_analysis': has_spatial_service,
        'noun_consensus':   has_vlm,
        'verb_consensus':   has_vlm,
        'sam3':             has_vlm,
        'caption_summary':  has_multi_vlm,
    }
