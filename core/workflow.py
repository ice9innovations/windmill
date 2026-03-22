"""
Code-owned workflow metadata and downstream expectation logic.

This module is the single source of truth for Windmill's internal dependency
map. It is intentionally defined in Python rather than operator configuration.
"""

WORKFLOW_VERSION = 1

PREDICATE_DESCRIPTIONS = {
    'has_consensus_service': 'At least one submitted primary service has consensus=true.',
    'has_spatial_primary': 'At least one submitted primary service is spatial.',
    'has_vlm': 'At least one submitted primary service is a VLM.',
    'has_multi_vlm': 'At least two submitted primary services are VLMs.',
    'has_nudenet': 'nudenet is present in services_submitted.',
    'has_florence2': 'florence2 is present in services_submitted.',
}

WORKFLOW_STAGES = {
    'harmony': {
        'kind': 'system',
        'expected_when': ['has_spatial_primary', 'tier_allows:system.harmony'],
        'triggered_by': [
            {
                'source': 'primary.spatial',
                'worker': 'workers/base_worker.py',
            }
        ],
    },
    'consensus': {
        'kind': 'system',
        'expected_when': ['has_consensus_service', 'tier_allows:system.harmony'],
        'triggered_by': [
            {
                'source': 'primary.consensus_eligible',
                'worker': 'workers/base_worker.py',
            },
            {
                'source': 'system.harmony',
                'worker': 'workers/harmony_worker.py',
                'note': 'harmony retriggers consensus after merged_boxes update',
            },
        ],
    },
    'noun_consensus': {
        'kind': 'system',
        'expected_when': ['has_vlm'],
        'triggered_by': [
            {
                'source': 'primary.vlm',
                'worker': 'workers/base_worker.py',
            }
        ],
    },
    'verb_consensus': {
        'kind': 'system',
        'expected_when': ['has_vlm'],
        'triggered_by': [
            {
                'source': 'primary.vlm',
                'worker': 'workers/base_worker.py',
            }
        ],
    },
    'sam3': {
        'kind': 'system',
        'expected_when': ['has_vlm', 'tier_allows:system.sam3'],
        'triggered_by': [
            {
                'source': 'system.florence2_grounding',
                'worker': 'workers/florence2_grounding_worker.py',
            }
        ],
    },
    'caption_summary': {
        'kind': 'system',
        'expected_when': ['has_multi_vlm', 'tier_allows:system.caption_summary'],
        'triggered_by': [
            {
                'source': 'system.noun_consensus',
                'worker': 'workers/noun_consensus_worker.py',
                'note': 'progressive; retriggered as additional captions arrive',
            }
        ],
    },
    'content_analysis': {
        'kind': 'system',
        'expected_when': ['has_nudenet', 'tier_allows:system.content_analysis'],
        'triggered_by': [
            {
                'source': 'system.consensus',
                'worker': 'workers/consensus_worker.py',
            }
        ],
    },
    'florence2_grounding': {
        'kind': 'system',
        'expected_when': ['has_florence2'],
        'triggered_by': [
            {
                'source': 'system.noun_consensus',
                'worker': 'workers/noun_consensus_worker.py',
                'note': 'progressive; may trigger multiple times as noun consensus updates',
            }
        ],
    },
}


def build_workflow_context(services_submitted, config, tier='free'):
    """Build the normalized predicate context for one submission."""
    services_submitted = services_submitted or []
    has_consensus_service = any(
        config.should_trigger_consensus(f'primary.{s}')
        for s in services_submitted
    )
    vlm_services = [
        s for s in services_submitted
        if config.is_vlm_service(f'primary.{s}')
    ]
    return {
        'services_submitted': services_submitted,
        'tier': tier,
        'has_consensus_service': has_consensus_service,
        'has_spatial_primary': any(
            config.is_spatial_service(f'primary.{s}')
            for s in services_submitted
        ),
        'has_vlm': len(vlm_services) > 0,
        'has_multi_vlm': len(vlm_services) >= 2,
        'has_nudenet': 'nudenet' in services_submitted,
        'has_florence2': 'florence2' in services_submitted,
    }


def evaluate_condition(condition, context, config):
    """Evaluate one symbolic workflow condition."""
    if condition.startswith('tier_allows:'):
        service_full_name = condition.split(':', 1)[1]
        return config.is_available_for_tier(service_full_name, context['tier'])
    return bool(context.get(condition, False))


def compute_expected_downstream(services_submitted, config, tier='free'):
    """Return downstream expectation booleans for one submission."""
    if not services_submitted:
        return {}

    context = build_workflow_context(services_submitted, config, tier)
    return {
        stage_name: all(
            evaluate_condition(condition, context, config)
            for condition in stage_def.get('expected_when', [])
        )
        for stage_name, stage_def in WORKFLOW_STAGES.items()
    }


def get_workflow_definition():
    """Return the machine-readable workflow contract."""
    return {
        'version': WORKFLOW_VERSION,
        'predicates': PREDICATE_DESCRIPTIONS,
        'stages': WORKFLOW_STAGES,
    }
