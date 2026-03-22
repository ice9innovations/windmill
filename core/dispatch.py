"""
Service resolution and downstream dispatch logic.

No Flask dependency. All functions take a ServiceConfig instance as a parameter
so callers can pass their own config path (e.g. ice9-api pointing at a different
service_config.yaml location).
"""

from core.workflow import compute_expected_downstream


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
