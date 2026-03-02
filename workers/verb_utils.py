#!/usr/bin/env python3
"""
Verb utilities for VLM verb consensus in Windmill.

Provides WordNet-based synonym collapsing for verbs: maps surface forms like
"sprint", "dash", "run" to a single canonical form so that cross-VLM agreement
can be measured on meaning rather than exact wording.

Parallel to noun_utils.py — same connected-component approach, same design
principles. WordNet verb coverage is shallower than noun coverage, so false
groupings via secondary senses are a real risk. Like noun_utils, we use only
the first (most common) synset per verb to keep groupings conservative.
"""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

_wordnet_available = None


def warmup_wordnet() -> bool:
    """Load WordNet corpus at startup so the first message doesn't pay the
    disk-load penalty mid-processing. Safe to call multiple times."""
    return _check_wordnet()


def _check_wordnet() -> bool:
    global _wordnet_available
    if _wordnet_available is not None:
        return _wordnet_available
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('run', pos=wn.VERB)  # confirm corpus is downloaded
        _wordnet_available = True
        logger.info("verb_utils: WordNet available")
    except Exception as e:
        _wordnet_available = False
        logger.warning(f"verb_utils: WordNet unavailable ({e}) - falling back to surface forms")
    return _wordnet_available


def _get_synsets(verb: str) -> List[str]:
    """
    Return synset names for a verb in WordNet's order (most common sense first).
    Returns [] if WordNet unavailable or verb not found.

    Only the first synset is used — verb polysemy in WordNet is high (e.g.
    "run" has 41 senses). Using all synsets causes false groupings via rare
    secondary senses.
    """
    if not _check_wordnet():
        return []
    from nltk.corpus import wordnet as wn
    verb_clean = verb.lower().strip()
    synsets = wn.synsets(verb_clean, pos=wn.VERB)
    return [s.name() for s in synsets[:1]]


def _synset_canonical(synset_name: str) -> str:
    """First lemma name of a synset — used as the label for a matched group."""
    if not _check_wordnet():
        return synset_name
    from nltk.corpus import wordnet as wn
    return wn.synset(synset_name).lemma_names()[0].replace('_', ' ')


def _find_groups(all_verbs: Set[str],
                 verb_synset_list: Dict[str, List[str]]) -> List[Set[str]]:
    """
    Find connected components of verbs that share at least one synset.
    Returns a list of sets, each set being a group of synonym verbs.
    """
    synset_to_verbs: Dict[str, Set[str]] = {}
    for verb, synsets in verb_synset_list.items():
        for s in synsets:
            synset_to_verbs.setdefault(s, set()).add(verb)

    neighbors: Dict[str, Set[str]] = {v: set() for v in all_verbs}
    for verbs_sharing in synset_to_verbs.values():
        if len(verbs_sharing) > 1:
            for v1 in verbs_sharing:
                for v2 in verbs_sharing:
                    if v1 != v2:
                        neighbors[v1].add(v2)

    visited: Set[str] = set()
    groups: List[Set[str]] = []
    for verb in sorted(all_verbs):  # sorted for determinism
        if verb not in visited:
            component: Set[str] = set()
            queue = [verb]
            while queue:
                v = queue.pop()
                if v not in visited:
                    visited.add(v)
                    component.add(v)
                    queue.extend(neighbors[v] - visited)
            groups.append(component)

    return groups


def collapse_synonyms(service_verb_map: Dict[str, List[str]]) -> List[dict]:
    """
    Collapse synonym variants across VLM services and count votes.

    Groups verbs that share a WordNet synset (connected components), then
    labels each group using the most common surface form — same strategy as
    noun_utils to avoid WordNet lemma quirks producing unintuitive canonicals.

    WordNet is only used to label a confirmed match, never to transform a lone
    verb whose sense is ambiguous.

    Args:
        service_verb_map: {service_name: [verb, ...]} from each VLM

    Returns:
        List of dicts sorted by vote_count descending, each containing:
        {
            "canonical":     str,   # most common surface form in the group
            "surface_forms": list,  # what each model actually said
            "services":      list,  # which services contributed
            "vote_count":    int,   # number of services that saw it
            "confidence":    float  # vote_count / total services (0.0-1.0)
        }
    """
    total_services = len(service_verb_map)
    if total_services == 0:
        return []

    all_verbs: Set[str] = set()
    for verbs in service_verb_map.values():
        all_verbs.update(verbs)

    verb_synset_list: Dict[str, List[str]] = {
        verb: _get_synsets(verb) for verb in all_verbs
    }

    groups = _find_groups(all_verbs, verb_synset_list)

    # Pre-count how many services mentioned each verb (used for canonical selection)
    verb_service_count: Dict[str, int] = {}
    for verb in all_verbs:
        verb_service_count[verb] = sum(
            1 for verbs in service_verb_map.values() if verb in verbs
        )

    noun_to_canonical: Dict[str, str] = {}
    for group in groups:
        if len(group) == 1:
            verb = next(iter(group))
            noun_to_canonical[verb] = verb.lower().strip()
        else:
            canonical = min(
                group,
                key=lambda v: (-verb_service_count.get(v, 0), v.lower().strip())
            ).lower().strip()
            for verb in group:
                noun_to_canonical[verb] = canonical

    canonical_map: Dict[str, Dict] = {}
    for service, verbs in service_verb_map.items():
        for verb in verbs:
            canonical = noun_to_canonical.get(verb, verb.lower().strip())
            if canonical not in canonical_map:
                canonical_map[canonical] = {'services': set(), 'surface_forms': set()}
            canonical_map[canonical]['services'].add(service)
            canonical_map[canonical]['surface_forms'].add(verb)

    results = []
    for canonical, info in canonical_map.items():
        vote_count = len(info['services'])
        results.append({
            'canonical': canonical,
            'surface_forms': sorted(info['surface_forms']),
            'services': sorted(info['services']),
            'vote_count': vote_count,
            'confidence': round(vote_count / total_services, 3),
        })

    results.sort(key=lambda x: (-x['vote_count'], x['canonical']))
    return results
