#!/usr/bin/env python3
"""
Noun utilities for VLM noun consensus in Windmill.

Synonym collapsing strategy: WordNet primary, ConceptNet second pass.

1. collapse_synonyms() — WordNet-based connected components, with ConceptNet
   as a second pass for pairs WordNet misses (e.g. "truck"/"lorry",
   "sofa"/"couch"). ConceptNet data lives in the conceptnet_edges table.
   Pass db_conn to enable ConceptNet; omit for WordNet-only fallback.

2. collapse_synonyms_clip() — DEAD CODE. CLIP text embeddings have visual
   co-occurrence bias ("dog"/"man" scores higher than "dog"/"bulldog") that
   makes them fundamentally unsuitable for synonym detection. Do not use.

Key design principle: canonicalization is only applied when two or more nouns
are confirmed equivalent. Lone nouns always keep their surface form.
"""

import logging
import os
from functools import lru_cache
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# Manual synonym overrides for content moderation.
# WordNet's primary senses for these terms are non-anatomical (cock=rooster,
# pussy=cat) so it will not link them to their clinical equivalents.
# Applied as a pre-pass before WordNet so votes from different VLM vocabularies
# are correctly merged.
CONTENT_MODERATION_SYNONYMS = {
    'cock':      'penis',
    'dick':      'penis',
    'pussy':     'vagina',
    'cunt':      'vagina',
    'marijuana': 'cannabis',
    'panty':     'underwear',
    'panties':   'underwear',
    'bra top':   'bra',
    'bikini top': 'bra',
}

_wordnet_available = None

# MWE (Multi-Word Expression) set, loaded once at startup.
# Contains underscore-joined lowercase strings for recognized compound nouns,
# e.g. {'dirt_track', 'rain_forest', 'coral_reef'}.
# Source: Animal Farm's curated mwe.txt (64k entries).
# Compounds NOT in this set are stripped to their head noun at consensus time.
_mwe_set: set = set()
_mwe_loaded: bool = False

_MWE_GITHUB_URL = "https://raw.githubusercontent.com/ice9innovations/animal-farm/refs/heads/main/config/mwe.txt"
_MWE_CACHE = os.path.join(os.path.dirname(__file__), '..', 'config', 'mwe.txt')
_MWE_AUTO_UPDATE = os.environ.get('AUTO_UPDATE', 'True').lower() == 'true'
_MWE_TIMEOUT = float(os.environ.get('TIMEOUT', '10.0'))


def load_mwe() -> int:
    """Load the MWE list into memory at worker startup.

    Tries GitHub first (if AUTO_UPDATE is enabled), caches locally, then
    falls back to the local cache if GitHub is unavailable. Follows the
    same pattern as Animal Farm's LAVIS/REST.py load_mwe_mappings().

    Entries are stored as underscore-joined lowercase strings matching
    the format in mwe.txt (e.g. 'dirt_track', 'rain_forest').
    Safe to call multiple times — subsequent calls are no-ops.

    Returns the number of entries loaded.
    """
    import requests as _requests

    global _mwe_set, _mwe_loaded
    if _mwe_loaded:
        return len(_mwe_set)

    mwe_text = []

    if _MWE_AUTO_UPDATE:
        try:
            response = _requests.get(_MWE_GITHUB_URL, timeout=_MWE_TIMEOUT)
            response.raise_for_status()
            mwe_text = response.text.splitlines()
            try:
                with open(_MWE_CACHE, 'w') as f:
                    f.write(response.text)
            except Exception as e:
                logger.warning(f"noun_utils: failed to cache MWE ({e})")
            logger.info(f"noun_utils: loaded {len(mwe_text)} MWE lines from GitHub")
        except Exception as e:
            logger.warning(f"noun_utils: GitHub MWE fetch failed ({e}), trying local cache")

    if not mwe_text:
        try:
            with open(_MWE_CACHE) as f:
                mwe_text = f.read().splitlines()
            logger.info(f"noun_utils: loaded {len(mwe_text)} MWE lines from local cache")
        except Exception as e:
            logger.warning(f"noun_utils: failed to load MWE cache ({e}), MWE normalization disabled")

    for line in mwe_text:
        entry = line.strip().lower()
        if entry:
            _mwe_set.add(entry)

    _mwe_loaded = True
    logger.info(f"noun_utils: {len(_mwe_set)} MWE entries ready")
    return len(_mwe_set)


def is_mwe(phrase: str) -> bool:
    """Return True if phrase is a recognized multi-word expression.

    Accepts space- or underscore-separated input, e.g. 'qr code' or 'qr_code'.
    Returns False if the MWE list has not been loaded yet.
    """
    if not _mwe_loaded:
        return False
    return '_'.join(phrase.lower().split()) in _mwe_set


def _mwe_normalize(noun: str) -> str:
    """Strip adjective modifiers from compound nouns not in the MWE list.

    Single-word nouns are returned unchanged.
    Multi-word nouns in MWE are returned unchanged (recognized compounds),
    with one exception: plural MWEs are normalized to their singular form
    when the singular is also in the MWE list (e.g. 'ram modules' → 'ram module').
    Multi-word nouns NOT in MWE are reduced to their head noun (last word).

    Examples (with MWE loaded):
        'dirt path'   -> 'path'       (not in MWE)
        'dirt road'   -> 'road'       (not in MWE)
        'dirt track'  -> 'dirt track' (in MWE as dirt_track)
        'rain forest' -> 'rain forest'(in MWE as rain_forest)
        'ram modules' -> 'ram module' (plural MWE → singular MWE)
        'pangolin'    -> 'pangolin'   (single word, unchanged)
    """
    parts = noun.replace('_', ' ').split()
    if len(parts) <= 1:
        return noun
    if not _mwe_loaded:
        return noun
    mwe_key = '_'.join(p.lower() for p in parts)
    if mwe_key in _mwe_set:
        # Prefer singular form when the last word is plural and singular MWE exists
        last = parts[-1].lower()
        if last.endswith('s') and len(last) > 2:
            singular_key = '_'.join(p.lower() for p in parts[:-1]) + '_' + last[:-1]
            if singular_key in _mwe_set:
                return ' '.join(parts[:-1] + [last[:-1]])
        return noun
    return parts[-1]


def apply_mwe_normalization(
    service_noun_map: Dict[str, List[str]],
    service_category_map: Dict[str, dict],
) -> tuple:
    """Apply MWE head-noun stripping to both the noun map and category map.

    Normalizes compound nouns that are not in MWE to their head nouns,
    updating both maps so category lookups remain consistent.

    Deduplicates noun lists per service after normalization (e.g. if a
    service output both 'dirt path' and 'path', both become 'path').

    Returns (norm_noun_map, norm_category_map).
    """
    norm_noun_map = {}
    norm_cat_map = {}
    for service, nouns in service_noun_map.items():
        # Normalize and deduplicate, preserving first-seen order
        seen = {}
        for n in nouns:
            normed = _mwe_normalize(n)
            if normed not in seen:
                seen[normed] = True
        norm_noun_map[service] = list(seen.keys())

        # Remap category map keys to their normalized forms
        cats = service_category_map.get(service, {})
        norm_cats = {}
        for noun, cat in cats.items():
            norm_noun = _mwe_normalize(noun)
            # Don't overwrite if a more-specific entry already exists
            if norm_noun not in norm_cats:
                norm_cats[norm_noun] = cat
        norm_cat_map[service] = norm_cats

    return norm_noun_map, norm_cat_map


# ConceptNet adjacency set, loaded once at startup.
# Each entry is a frozenset of two normalized URIs that share a
# /r/Synonym edge. Bidirectional by construction.
# IsA is intentionally excluded: IsA is hierarchy (woman IS-A person),
# not equivalence. Loading IsA as synonyms destroys gender/specificity signals.
_conceptnet_edges: set = set()
_conceptnet_loaded = False

_CONCEPTNET_CACHE = os.path.join(os.path.dirname(__file__), 'conceptnet_cache.pkl')


def load_conceptnet(db_conn) -> int:
    """Load ConceptNet edges into memory at worker startup.

    Loads from a local pickle cache if available (fast), otherwise fetches
    from the DB and writes the cache for future startups. Subsequent calls
    within the same process are no-ops.

    Returns the number of edges loaded.
    """
    import os
    import pickle

    global _conceptnet_edges, _conceptnet_loaded
    if _conceptnet_loaded:
        return len(_conceptnet_edges)

    # Try local cache first
    if os.path.exists(_CONCEPTNET_CACHE):
        try:
            with open(_CONCEPTNET_CACHE, 'rb') as f:
                _conceptnet_edges = pickle.load(f)
            _conceptnet_loaded = True
            logger.info(f"noun_utils: loaded {len(_conceptnet_edges)} ConceptNet edges from cache")
            return len(_conceptnet_edges)
        except Exception as e:
            logger.warning(f"noun_utils: cache load failed ({e}), fetching from DB")

    # Fetch from DB and write cache
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            """
            SELECT start_uri, end_uri FROM conceptnet_edges
            WHERE relation = '/r/Synonym'
            """
        )
        for start_uri, end_uri in cursor.fetchall():
            _conceptnet_edges.add(frozenset((start_uri, end_uri)))
        cursor.close()
        _conceptnet_loaded = True
        logger.info(f"noun_utils: loaded {len(_conceptnet_edges)} ConceptNet edges from DB")

        try:
            with open(_CONCEPTNET_CACHE, 'wb') as f:
                pickle.dump(_conceptnet_edges, f)
            logger.info(f"noun_utils: wrote ConceptNet cache to {_CONCEPTNET_CACHE}")
        except Exception as e:
            logger.warning(f"noun_utils: failed to write cache ({e})")

    except Exception as e:
        logger.warning(f"noun_utils: failed to load ConceptNet ({e}), skipping")

    return len(_conceptnet_edges)


def warmup_wordnet() -> bool:
    """Load WordNet corpus at startup so the first message doesn't pay the
    disk-load penalty mid-processing.  Safe to call multiple times."""
    return _check_wordnet()


def _check_wordnet() -> bool:
    global _wordnet_available
    if _wordnet_available is not None:
        return _wordnet_available
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('cat', pos=wn.NOUN)  # confirm corpus is downloaded
        _wordnet_available = True
        logger.info("noun_utils: WordNet available")
    except Exception as e:
        _wordnet_available = False
        logger.warning(f"noun_utils: WordNet unavailable ({e}) - falling back to surface forms")
    return _wordnet_available


def _head_noun(noun: str) -> str:
    """Return the head noun of a compound noun phrase.

    English compound nouns are right-headed: the last word is the head.
    e.g. "cast iron pan" -> "pan", "frying pan" -> "pan".
    Returns the noun unchanged if it is a single word.
    """
    parts = noun.replace('_', ' ').split()
    return parts[-1] if len(parts) > 1 else noun


def _conceptnet_uri(noun: str) -> str:
    """Convert a noun to its ConceptNet English URI prefix."""
    return '/c/en/' + noun.lower().strip().replace(' ', '_')


def _are_conceptnet_synonyms(noun1: str, noun2: str) -> bool:
    """Return True if ConceptNet has a direct /r/Synonym edge between
    noun1 and noun2 (either direction).

    URI matching uses exact-or-slash-suffix to avoid compound word false
    matches (e.g. /c/en/dog matches /c/en/dog and /c/en/dog/n/wn/animal
    but NOT /c/en/dogfish).

    For compound nouns that have no ConceptNet entry, falls back to the
    head noun (last word) before giving up.
    """
    for n1, n2 in [(noun1, noun2), (_head_noun(noun1), _head_noun(noun2))]:
        if n1 == n2:
            continue
        if frozenset((_conceptnet_uri(n1), _conceptnet_uri(n2))) in _conceptnet_edges:
            return True
        if _head_noun(noun1) == noun1 and _head_noun(noun2) == noun2:
            break
    return False


@lru_cache(maxsize=2048)
def _get_synsets(noun: str) -> List[str]:
    """
    Return synset names for a noun in WordNet's order (most common sense first).
    Returns [] if WordNet unavailable or noun not found.
    """
    if not _check_wordnet():
        return []
    from nltk.corpus import wordnet as wn
    noun_clean = noun.lower().strip()
    synsets = wn.synsets(noun_clean.replace(' ', '_'), pos=wn.NOUN)
    if not synsets:
        synsets = wn.synsets(noun_clean, pos=wn.NOUN)
    # Only the first (most common) synset — using all synsets causes false
    # groupings via secondary senses (e.g. cap+jacket share crown.n.11,
    # the dental/hat-brim sense, even though both are clothing items in context).
    return [s.name() for s in synsets[:1]]


@lru_cache(maxsize=2048)
def _synset_canonical(synset_name: str) -> str:
    """First lemma name of a synset - used as the label for a matched group."""
    if not _check_wordnet():
        return synset_name
    from nltk.corpus import wordnet as wn
    return wn.synset(synset_name).lemma_names()[0].replace('_', ' ')


def _find_groups(all_nouns: Set[str],
                 noun_synset_list: Dict[str, List[str]],
                 db_conn=None) -> List[Set[str]]:
    """
    Find connected components of nouns that share at least one synset,
    with ConceptNet as a second pass for pairs WordNet misses.

    Returns a list of sets, each set being a group of synonym nouns.
    """
    # Build: synset_name → set of nouns that have it
    synset_to_nouns: Dict[str, Set[str]] = {}
    for noun, synsets in noun_synset_list.items():
        for s in synsets:
            synset_to_nouns.setdefault(s, set()).add(noun)

    # Build adjacency from shared WordNet synsets
    neighbors: Dict[str, Set[str]] = {n: set() for n in all_nouns}
    for nouns_sharing in synset_to_nouns.values():
        if len(nouns_sharing) > 1:
            for n1 in nouns_sharing:
                for n2 in nouns_sharing:
                    if n1 != n2:
                        neighbors[n1].add(n2)

    # ConceptNet second pass: pure in-memory set lookups, no DB round-trip.
    if _conceptnet_loaded:
        nouns_list = sorted(all_nouns)
        for i, n1 in enumerate(nouns_list):
            for n2 in nouns_list[i + 1:]:
                if n2 not in neighbors[n1] and _are_conceptnet_synonyms(n1, n2):
                    neighbors[n1].add(n2)
                    neighbors[n2].add(n1)

    # MWE head-word pass: connect a single-word noun to any recognized MWE
    # whose last word it is (e.g. "module" ↔ "ram module", "code" ↔ "qr code").
    # Handles the case where some VLMs extract the full compound and others
    # extract only the head noun — both sets of votes merge into one entry.
    single_nouns = {n for n in all_nouns if ' ' not in n}
    mwe_nouns = {n for n in all_nouns if ' ' in n and is_mwe(n)}
    for mwe in mwe_nouns:
        head = mwe.split()[-1]
        if head in single_nouns and head not in neighbors[mwe]:
            neighbors[mwe].add(head)
            neighbors[head].add(mwe)

    # BFS connected components
    visited: Set[str] = set()
    groups: List[Set[str]] = []
    for noun in sorted(all_nouns):  # sorted for determinism
        if noun not in visited:
            component: Set[str] = set()
            queue = [noun]
            while queue:
                n = queue.pop()
                if n not in visited:
                    visited.add(n)
                    component.add(n)
                    queue.extend(neighbors[n] - visited)
            groups.append(component)

    return groups


def _normalize_slang(n: str) -> str:
    """Normalize content moderation slang to clinical equivalents.

    WordNet's primary senses for these terms are non-anatomical (cock=rooster,
    pussy=cat) so it will not link them to their clinical equivalents.
    Applied as a pre-pass before any collapsing strategy.

    Checks for multi-word synonyms and single-word slang as word-bounded
    substrings to handle compounds with modifiers (e.g. "lace bra top"
    contains "bra top" → "bra") and spaCy compound-dep extractions where
    a slang term is embedded (e.g. "man cock" contains "cock" → "penis").
    """
    key = n.lower().strip()

    # Exact match first
    if key in CONTENT_MODERATION_SYNONYMS:
        return CONTENT_MODERATION_SYNONYMS[key]

    # Check for synonyms as word-bounded substrings (longest match first,
    # including single-word scan for slang embedded in spaCy compound nouns)
    words = key.split()
    for length in range(len(words), 0, -1):
        for i in range(len(words) - length + 1):
            phrase = ' '.join(words[i:i+length])
            if phrase in CONTENT_MODERATION_SYNONYMS:
                return CONTENT_MODERATION_SYNONYMS[phrase]

    return n


def _apply_slang_normalization(service_noun_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Apply content moderation pre-pass to all nouns in a service map."""
    return {
        service: [_normalize_slang(n) for n in nouns]
        for service, nouns in service_noun_map.items()
    }


def _count_noun_services(service_noun_map: Dict[str, List[str]], all_nouns: set) -> Dict[str, int]:
    """Count how many services mentioned each noun — used for canonical selection."""
    return {
        noun: sum(1 for nouns in service_noun_map.values() if noun in nouns)
        for noun in all_nouns
    }


def _select_canonical(group: set, noun_service_count: Dict[str, int]) -> str:
    """Pick the canonical form for a group: most frequent surface form.

    Ties broken by preferring the more specific (longer) form — e.g. "ram module"
    over "module" — then alphabetically.
    """
    return min(
        group,
        key=lambda n: (-noun_service_count.get(n, 0), -len(n.split()), n.lower().strip())
    ).lower().strip()


def _build_results(service_noun_map: Dict[str, List[str]],
                   noun_to_canonical: Dict[str, str],
                   total_services: int) -> List[dict]:
    """Aggregate votes per canonical group and return sorted result list."""
    canonical_map: Dict[str, Dict] = {}
    for service, nouns in service_noun_map.items():
        for noun in nouns:
            canonical = noun_to_canonical.get(noun, noun.lower().strip())
            if canonical not in canonical_map:
                canonical_map[canonical] = {'services': set(), 'surface_forms': set()}
            canonical_map[canonical]['services'].add(service)
            canonical_map[canonical]['surface_forms'].add(noun)

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


def collapse_synonyms_clip(service_noun_map: Dict[str, List[str]],
                           embeddings: Dict[str, List[float]],
                           threshold: float = 0.85) -> List[dict]:
    """Collapse synonym variants using CLIP embedding cosine similarity.

    Groups nouns whose CLIP text embeddings are within `threshold` cosine
    similarity using connected components — the same graph structure as the
    WordNet path, with CLIP distance replacing synset overlap as the
    adjacency criterion.

    Since CLIP embeddings are L2-normalized, cosine similarity equals the
    dot product.

    Args:
        service_noun_map: {service_name: [noun, ...]} from each VLM
        embeddings:       {noun: embedding} — normalized float lists from
                          the clip-score /v3/embed/text endpoint
        threshold:        cosine similarity floor for merging (tune via
                          NOUN_CLIP_SIMILARITY_THRESHOLD env var)

    Returns:
        Same format as collapse_synonyms().
    """
    total_services = len(service_noun_map)
    if total_services == 0:
        return []

    service_noun_map = _apply_slang_normalization(service_noun_map)

    all_nouns: set = set()
    for nouns in service_noun_map.values():
        all_nouns.update(nouns)

    noun_service_count = _count_noun_services(service_noun_map, all_nouns)
    nouns_list = sorted(all_nouns)  # deterministic order

    # Build adjacency from cosine similarity
    neighbors: Dict[str, set] = {n: set() for n in all_nouns}
    for i, n1 in enumerate(nouns_list):
        emb1 = embeddings.get(n1)
        if emb1 is None:
            continue
        for n2 in nouns_list[i + 1:]:
            emb2 = embeddings.get(n2)
            if emb2 is None:
                continue
            sim = sum(x * y for x, y in zip(emb1, emb2))
            if sim >= threshold:
                neighbors[n1].add(n2)
                neighbors[n2].add(n1)

    # BFS connected components
    visited: set = set()
    groups: List[set] = []
    for noun in nouns_list:
        if noun not in visited:
            component: set = set()
            queue = [noun]
            while queue:
                n = queue.pop()
                if n not in visited:
                    visited.add(n)
                    component.add(n)
                    queue.extend(neighbors[n] - visited)
            groups.append(component)

    noun_to_canonical: Dict[str, str] = {}
    for group in groups:
        canonical = _select_canonical(group, noun_service_count)
        for noun in group:
            noun_to_canonical[noun] = canonical

    return _build_results(service_noun_map, noun_to_canonical, total_services)


def collapse_synonyms(service_noun_map: Dict[str, List[str]]) -> List[dict]:
    """Collapse synonym variants using WordNet synset overlap, with ConceptNet
    as a second pass for pairs WordNet misses.

    WordNet is only used to label a confirmed match — lone nouns always keep
    their surface form so WordNet quirks never corrupt unambiguous terms.

    ConceptNet is used automatically if load_conceptnet() was called at startup.

    Args:
        service_noun_map: {service_name: [noun, ...]} from each VLM

    Returns:
        List of dicts sorted by vote_count descending, each containing:
        {
            "canonical":     str,
            "surface_forms": list,
            "services":      list,
            "vote_count":    int,
            "confidence":    float,
        }
    """
    total_services = len(service_noun_map)
    if total_services == 0:
        return []

    service_noun_map = _apply_slang_normalization(service_noun_map)

    all_nouns: Set[str] = set()
    for nouns in service_noun_map.values():
        all_nouns.update(nouns)

    noun_synset_list: Dict[str, List[str]] = {
        noun: _get_synsets(noun) for noun in all_nouns
    }
    noun_service_count = _count_noun_services(service_noun_map, all_nouns)

    groups = _find_groups(all_nouns, noun_synset_list)

    noun_to_canonical: Dict[str, str] = {}
    for group in groups:
        if len(group) == 1:
            noun = next(iter(group))
            noun_to_canonical[noun] = noun.lower().strip()
        else:
            # Use most common surface form — WordNet's first lemma is unreliable
            # as a label (e.g. "hat"+"cap" share crown.n.06 → "crown").
            noun_to_canonical.update(
                {noun: _select_canonical(group, noun_service_count) for noun in group}
            )

    return _build_results(service_noun_map, noun_to_canonical, total_services)
