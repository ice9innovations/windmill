"""
Service resolution and downstream dispatch logic.

No Flask dependency. All functions take a ServiceConfig instance as a parameter
so callers can pass their own config path (e.g. ice9-api pointing at a different
service_config.yaml location).
"""


def resolve_services(tier, config):
    """Return the tier-appropriate primary services.

    Tier is the sole control over which services run. There is no mechanism
    for callers to request a specific service list — that would allow tier bypass.

    Returns a sorted list of short service names (e.g. ['blip', 'colors', ...]).
    """
    tier_services = config.get_services_by_tier(tier)
    return sorted(
        name.split('.', 1)[1]
        for name in tier_services.keys()
        if name.startswith('primary.')
    )


def compute_expected_downstream(services_submitted, config, tier='free'):
    """Determine which downstream services are expected based on submitted primary services and tier.

    Returns a dict of {downstream_name: bool} indicating whether each downstream
    service is expected to eventually produce a result for this image.

    SAM3 and caption_summary are basic+ only — they are never triggered for
    free-tier images regardless of which primary services ran.
    content_analysis availability is read from service_config.yaml so that
    tier membership changes there automatically flow through here.
    """
    if not services_submitted:
        return {}

    has_consensus_service = any(
        config.should_trigger_consensus(f'primary.{s}')
        for s in services_submitted
    )

    vlm_services = [
        s for s in services_submitted
        if config.is_vlm_service(f'primary.{s}')
    ]
    has_vlm       = len(vlm_services) > 0
    has_multi_vlm = len(vlm_services) >= 2

    return {
        'consensus':        has_consensus_service and config.is_available_for_tier('system.harmony', tier),
        'content_analysis': config.is_available_for_tier('system.content_analysis', tier) and 'nudenet' in services_submitted,
        'noun_consensus':   has_vlm,
        'verb_consensus':   has_vlm,
        'sam3':             config.is_available_for_tier('system.sam3', tier) and has_vlm,
        'caption_summary':  config.is_available_for_tier('system.caption_summary', tier) and has_multi_vlm,
        'florence2_grounding': 'florence2' in services_submitted,
    }
