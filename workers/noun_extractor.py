#!/usr/bin/env python3
"""
Noun Extractor - spaCy-based noun extraction from VLM text output.

Used by all VLM services (BLIP2, Moondream, llama-cpp) to extract a
structured noun list from free-text image descriptions. Returns a
deduplicated, lowercase list suitable for cross-service reconciliation
in Windmill v2.

Also provides categorize_nouns() which maps each noun to a semantic
category (animal, human, vehicle, food, plant, clothing, structure,
body part, furniture, object) via WordNet hypernym traversal.
"""

import re
import logging
from collections import deque
from typing import Dict, List, Optional
from noun_utils import is_mwe

logger = logging.getLogger(__name__)

# Loaded once at import time - en_core_web_sm is CPU-only, ~12MB
_nlp = None


def _load_model():
    global _nlp
    if _nlp is not None:
        return True
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("noun_extractor: en_core_web_sm loaded")
        return True
    except OSError:
        logger.error("noun_extractor: en_core_web_sm not found - run: "
                     "python -m spacy download en_core_web_sm")
        return False
    except ImportError:
        logger.error("noun_extractor: spacy not installed")
        return False


def warmup_noun_extractor() -> bool:
    """Load the spaCy model at worker startup so the first message pays no
    disk-load penalty mid-processing. Safe to call multiple times."""
    return _load_model()


# Noise words that slip through POS tagging but aren't useful content nouns.
# Includes purely directional/positional words that describe placement, not objects.
_STOP_NOUNS = {
    "image", "photo", "picture", "scene", "view", "shot",
    "background", "foreground", "area", "way", "lot", "kind",
    "type", "sort", "thing", "something", "anything", "nothing",
    "everything", "side", "part", "top", "bottom", "front", "back",
    "left", "right", "text", "content", "nudity",
}

# Words that are always plural and must not be lemmatized.
# Lemmatizing these produces wrong or misleading singular forms:
#   glasses → glass, pants → pant, jeans → jean (a fabric), scissors → scissor
# Noun chunks preserve their surface form already; this guard applies only
# to the standalone-noun pass where token.lemma_ is used.
_PLURALE_TANTUM = {
    # Eyewear / optical
    "glasses", "spectacles", "goggles", "binoculars",
    # Clothing
    "pants", "trousers", "jeans", "shorts", "tights", "leggings",
    "overalls", "pajamas", "pyjamas", "knickers",
    # Food (plurale tantum — "fries" must not lemmatize to "fry", a fish)
    "fries",
    # Tools / implements
    "scissors", "pliers", "tongs", "tweezers", "shears", "clippers",
    # Surroundings
    "surroundings",
}

# Dependency labels that mark purely adjectival or functional modifiers.
# A multi-word chunk whose non-root tokens are ALL in this set is an
# adjective+noun phrase — reduce it to the root noun only:
#   "larger lamp"        (amod)     → "lamp"
#   "light blue background" (amod)  → "background"  (then filtered by _STOP_NOUNS)
#   "round shade"        (amod)     → "shade"
# Chunks with ANY token labelled 'compound' contain a noun modifier and keep
# their full form:
#   "metal arm"          (compound) → "metal arm"
#   "animation studios"  (compound) → "animation studios"
_ADJ_DEPS = frozenset({
    'amod',     # adjectival modifier       "larger lamp"
    'advmod',   # adverbial modifier        "very large thing"
    'nummod',   # numeric modifier          "three lamps"
    'quantmod', # quantifier modifier
    'det',      # determiner                "the lamp" (after leading-strip pass)
    'poss',     # possessive                "its shade"
    'case',     # case marker               "'s"
})

# A valid extracted noun contains only letters, digits, spaces, and hyphens.
# Rejects candidates that still contain punctuation after edge-stripping.
_VALID_NOUN_RE = re.compile(r'^[a-z0-9][a-z0-9 \-]*$')


def _has_noun_synsets(lemma: str) -> bool:
    """Return True if lemma has at least one WordNet NOUN synset.

    Guards against spaCy en_core_web_sm misclassifying adjectives as
    compound-dep NOUN tokens (e.g. 'crispy' in 'crispy fries' gets
    tag_=NN despite being purely adjectival in WordNet). Fails open
    (True) if WordNet is unavailable so extraction still works without NLTK.
    """
    try:
        from nltk.corpus import wordnet as wn
        return bool(wn.synsets(lemma, pos=wn.NOUN))
    except Exception:
        return True


def _clean_surface(s: str) -> str:
    """Strip leading/trailing non-alphanumeric characters and normalise spaces."""
    s = re.sub(r'^[^a-z0-9]+', '', s)
    s = re.sub(r'[^a-z0-9]+$', '', s)
    return re.sub(r'\s+', ' ', s).strip()


# Manual overrides: noun → category.  Add entries here to hard-wire
# edge cases that WordNet gets wrong for this domain.
CATEGORY_OVERRIDES: Dict[str, str] = {
    "people": "human",
    "crowd": "human",
    "pangolid": "animal",   # Moondream hallucination for "pangolin"
    "fries": "food",
    "fry": "food",          # spaCy lemmatizes "fries" → "fry" when "French" is stripped as amod
}

# WordNet synset names that anchor each category.  BFS stops at the first
# anchor it encounters climbing the hypernym graph, so more-specific anchors
# listed here shadow broader ones automatically (e.g. footwear.n.02 is found
# before artifact.n.01, so boots → clothing rather than object).
_CATEGORY_ANCHORS: Dict[str, str] = {
    "animal.n.01":     "animal",
    "person.n.01":     "human",
    "vehicle.n.01":    "vehicle",
    "conveyance.n.03": "vehicle",   # wheeled_vehicle parent — catches car, bike
    "food.n.01":       "food",
    "food.n.02":       "food",
    "plant.n.02":      "plant",
    "clothing.n.01":   "clothing",
    "footwear.n.02":   "clothing",
    "structure.n.01":  "structure",
    "body_part.n.01":  "body part",
    "furniture.n.01":  "furniture",
}


def _find_category_for_synset(synset, max_depth: int = 12) -> Optional[str]:
    """BFS through the hypernym graph from synset to any anchor.

    Handles multiple inheritance correctly: all hypernym paths are
    explored in breadth-first order so the closest anchor wins.
    Returns None if no anchor is reached within max_depth hops.
    """
    queue = deque([(synset, 0)])
    visited: set = set()
    while queue:
        current, depth = queue.popleft()
        name = current.name()
        if depth > max_depth or name in visited:
            continue
        visited.add(name)
        if name in _CATEGORY_ANCHORS:
            return _CATEGORY_ANCHORS[name]
        for hypernym in current.hypernyms():
            if hypernym.name() not in visited:
                queue.append((hypernym, depth + 1))
    return None


def categorize_nouns(nouns: List[str]) -> Dict[str, str]:
    """Map each noun to a semantic category via WordNet hypernym traversal.

    Categories: animal, human, vehicle, food, plant, clothing,
                structure, body part, furniture, object (fallback).

    Nouns matched by CATEGORY_OVERRIDES bypass WordNet entirely.
    Nouns with at least one WordNet synset but no anchor match fall
    back to "object".  Nouns with no WordNet synsets are omitted.

    Args:
        nouns: List of nouns from extract_nouns()

    Returns:
        Dict mapping noun → category string
    """
    if not nouns:
        return {}

    try:
        from nltk.corpus import wordnet as wn
        # Trigger a cheap lookup to surface LookupError if data is missing
        wn.synsets("dog")
    except ImportError:
        logger.error("noun_extractor: nltk not installed")
        return {}
    except LookupError:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            from nltk.corpus import wordnet as wn  # noqa: F811 — re-import after download
        except Exception as e:
            logger.error(f"noun_extractor: failed to download wordnet corpus: {e}")
            return {}

    result: Dict[str, str] = {}
    for noun in nouns:
        if noun in CATEGORY_OVERRIDES:
            result[noun] = CATEGORY_OVERRIDES[noun]
            continue

        # WordNet uses underscores for multi-word entries
        wn_key = noun.replace(" ", "_")
        synsets = wn.synsets(wn_key, pos=wn.NOUN)
        if not synsets and " " in noun:
            # Compound phrase not in WordNet — walk component words right-to-left.
            # The last word is the head noun in English compounds, so try it first:
            #   "retriever dog" → "dog" → animal
            #   "city park"     → "park" → object
            for word in reversed(noun.split()):
                synsets = wn.synsets(word, pos=wn.NOUN)
                if synsets:
                    break
        if not synsets:
            continue  # no WordNet entry — omit from result

        category = _find_category_for_synset(synsets[0])
        result[noun] = category if category else "object"

    return result


def _extract_subject_from_doc(doc) -> Optional[str]:
    """Extract grammatical subject from a pre-parsed spaCy doc."""
    # Work on the first sentence only — VLM captions are usually one
    # sentence, and the main subject is always in the first clause.
    first_sent = next(doc.sents, None)
    if first_sent is None:
        return None

    # Find the root verb of the first sentence
    root = next((t for t in first_sent if t.dep_ == 'ROOT'), None)
    if root is None:
        return None

    # Find the nominal subject (nsubj) of the root
    nsubj = next((t for t in root.children if t.dep_ == 'nsubj'), None)

    if nsubj is not None and nsubj.pos_ == 'PRON':
        # Skip pronoun subjects ("it", "they", "something")
        nsubj = None

    if nsubj is not None:
        # Action sentence path: use the nsubj noun chunk head.
        # e.g. "A horse runs through a field" → nsubj=horse → "horse"
        for chunk in doc.noun_chunks:
            if nsubj in chunk:
                root_token = chunk.root
                surface = root_token.text.lower()
                candidate = surface if surface in _PLURALE_TANTUM else root_token.lemma_.lower()
                candidate = _clean_surface(candidate)
                if candidate and len(candidate) > 1 and candidate not in _STOP_NOUNS:
                    return candidate
                return None

        # No noun chunk found for nsubj — use the token itself
        surface = nsubj.text.lower()
        candidate = surface if surface in _PLURALE_TANTUM else nsubj.lemma_.lower()
        candidate = _clean_surface(candidate)
        if candidate and len(candidate) > 1 and candidate not in _STOP_NOUNS:
            return candidate
        return None

    # Nominal sentence fallback: no nsubj means there is likely no verb.
    # spaCy promotes the head noun to ROOT in nominal sentences, e.g.:
    #   "A bowl of soup with chives" → ROOT=bowl (NOUN)
    # Only fire if the ROOT is itself a noun — a verb ROOT with a missing
    # nsubj is a parse failure, not a nominal sentence.
    if root.pos_ == 'NOUN':
        for chunk in doc.noun_chunks:
            if root in chunk:
                root_token = chunk.root
                surface = root_token.text.lower()
                candidate = surface if surface in _PLURALE_TANTUM else root_token.lemma_.lower()
                candidate = _clean_surface(candidate)
                if candidate and len(candidate) > 1 and candidate not in _STOP_NOUNS:
                    return candidate
                return None

    return None


def extract_subject(text: str) -> Optional[str]:
    """
    Extract the grammatical subject from a VLM caption.

    Primary path: finds the nsubj of the root verb in the first sentence and
    returns the head noun lemma of that noun chunk.

    Fallback for nominal sentences (no verb): if the ROOT token is itself a
    noun, uses it as the subject. This handles food, product, and still-life
    captions such as "A bowl of soup with chives" where there is no action
    verb and spaCy promotes the head noun to ROOT.

    Returns None if no clear subject is found (pronoun subject, stop noun,
    verb ROOT with no nsubj, sentinel strings, etc.).

    Args:
        text: Free-text description from a VLM

    Returns:
        Lowercase lemma of the subject noun, or None
    """
    if not text or not text.strip():
        return None

    if not _load_model():
        return None

    try:
        doc = _nlp(text)
        return _extract_subject_from_doc(doc)
    except Exception as e:
        logger.error(f"noun_extractor: subject extraction failed: {e}")
        return None


def _extract_nouns_from_doc(doc) -> List[str]:
    """Extract nouns from a pre-parsed spaCy doc."""
    seen = set()          # full phrases already added
    seen_lemmas = set()   # individual lemmas consumed by a chunk
    nouns = []

    _LEADING_STRIP = {"a", "an", "the", "some", "any",
                      "this", "that", "these", "those"}
    _QUANTIFIERS = {"one", "two", "three", "four", "five", "six",
                    "seven", "eight", "nine", "ten", "many", "several",
                    "few", "multiple", "various", "numerous"}
    # Determiners that negate the noun — skip the whole chunk and
    # blacklist its lemmas so the standalone pass doesn't re-add them.
    # Handles "no people", "no text", "no nudity" from VLMs that
    # explicitly report the absence of content.
    _NEGATION_DETS = {"no", "without"}

    # Noun chunks first - preserves genuine compound nouns ("park bench",
    # "metal arm") as single entries rather than splitting them.
    for chunk in doc.noun_chunks:
        # Skip pronoun-headed chunks ("it", "they", "something")
        if chunk.root.pos_ == 'PRON':
            continue

        # Skip verb-rooted chunks ("looking at", "standing in") — spaCy
        # occasionally produces noun chunks whose root is a verb or gerund.
        if chunk.root.pos_ == 'VERB':
            continue

        raw_tokens = chunk.text.lower().strip().split()

        # Skip negated chunks ("no people", "no text", "no nudity").
        # Still mark their lemmas as seen so the standalone pass doesn't
        # re-add them (spaCy processes tokens independently of chunks).
        if raw_tokens and raw_tokens[0] in _NEGATION_DETS:
            for token in chunk:
                seen_lemmas.add(token.lemma_.lower())
                if token.text.lower() in _PLURALE_TANTUM:
                    seen_lemmas.add(token.text.lower())
            continue

        # Strip leading determiners and quantifiers
        while raw_tokens and raw_tokens[0] in _LEADING_STRIP | _QUANTIFIERS:
            raw_tokens = raw_tokens[1:]

        if not raw_tokens:
            continue

        if len(raw_tokens) == 1:
            # Single-word chunk: lemmatize (with plurale tantum guard)
            # so "cats" → "cat", "glasses" → "glasses"
            surface = raw_tokens[0]
            candidate = surface if surface in _PLURALE_TANTUM else chunk.root.lemma_.lower()

        else:
            # Multi-word chunk: check MWE list before any modifier stripping.
            # Recognized compounds (e.g. "qr code") bypass the synset filter
            # entirely — acronyms and technical terms often lack WordNet entries
            # but are still valid compound nouns.
            if is_mwe(' '.join(raw_tokens)):
                candidate = ' '.join(raw_tokens)
            else:
                # Inspect modifier dependency types.
                # If every non-root token is a purely adjectival/functional
                # modifier, reduce the chunk to just the root noun — this
                # strips colour/size/comparative adjectives that add noise
                # without semantic distinctiveness across VLMs.
                # If any non-root token has dep_='compound' (noun modifying
                # noun), strip adjectival tokens and rebuild from compound+root
                # tokens only:
                #   "white filing cabinet"  → "filing cabinet"
                #   "beige cardboard boxes" → "cardboard boxes"
                #   "metal arm"             → "metal arm"  (no amod, unchanged)
                non_root_tokens = [t for t in chunk if t != chunk.root]
                all_adjectival = all(t.dep_ in _ADJ_DEPS for t in non_root_tokens)

                if all_adjectival:
                    root = chunk.root
                    surface = root.text.lower()
                    candidate = surface if surface in _PLURALE_TANTUM else root.lemma_.lower()
                else:
                    # Keep only compound modifiers and the root — drop amod etc.
                    content_tokens = [t for t in chunk
                                      if (t.dep_ == 'compound'
                                          and _has_noun_synsets(t.lemma_.lower()))
                                      or t == chunk.root]
                    if len(content_tokens) == 1:
                        # Only the root survived stripping; reduce to root lemma
                        root = chunk.root
                        surface = root.text.lower()
                        candidate = surface if surface in _PLURALE_TANTUM else root.lemma_.lower()
                    else:
                        phrase = " ".join(t.text.lower() for t in content_tokens)
                        # Drop phrases with punctuation bleed or disjunctions
                        if any(c in phrase for c in ('"', "'", ',')):
                            continue
                        if ' or ' in phrase or ' and ' in phrase:
                            continue
                        candidate = phrase

        # Strip any surviving leading/trailing punctuation and validate.
        candidate = _clean_surface(candidate)

        if (not candidate
                or len(candidate) < 2
                or not _VALID_NOUN_RE.match(candidate)
                or candidate in seen
                or candidate in _STOP_NOUNS):
            continue

        seen.add(candidate)
        nouns.append(candidate)

        # Mark every lemma AND every plurale tantum surface form in this
        # chunk as consumed so the standalone pass doesn't re-add them.
        for token in chunk:
            seen_lemmas.add(token.lemma_.lower())
            surface = token.text.lower()
            if surface in _PLURALE_TANTUM:
                seen_lemmas.add(surface)

    # Sliding window MWE scan: catch recognized compounds that spaCy's chunker
    # missed. This is the sentence-level pass that was present in Animal Farm's
    # individual VLM services (via NLTK MWETokenizer) but was not carried over
    # when noun extraction was centralized into Windmill.
    #
    # Scans windows of 2–4 tokens against _mwe_set. Found compounds are added
    # to nouns and their component token lemmas are marked as seen so the
    # standalone pass below does not re-add bare head words (e.g. "retriever"
    # after "golden retriever" is found here).
    _MAX_MWE_LEN = 4  # no entry in the MWE set exceeds 4 words
    doc_tokens = list(doc)
    for start in range(len(doc_tokens)):
        for length in range(_MAX_MWE_LEN, 1, -1):
            end = start + length
            if end > len(doc_tokens):
                continue
            window = doc_tokens[start:end]
            phrase = ' '.join(t.text.lower() for t in window)
            # Only accept MWE matches that are noun-headed: the last token
            # must be a NOUN or PROPN. Filters verb-phrase MWEs like
            # "looking_at" that exist in the list but aren't noun compounds.
            if (is_mwe(phrase)
                    and phrase not in seen
                    and not all(t.lemma_.lower() in seen_lemmas for t in window)
                    and window[-1].pos_ in ('NOUN', 'PROPN')):
                seen.add(phrase)
                nouns.append(phrase)
                for t in window:
                    seen_lemmas.add(t.lemma_.lower())
                    seen_lemmas.add(t.text.lower())
                break  # don't try shorter windows starting at this position

    # Words that spaCy en_core_web_sm incorrectly tags as non-NOUN.
    # These are forced through the noun extraction pipeline regardless of POS tag.
    _MISCLASSIFIED_NOUNS = frozenset({
        "tattoo",  # spaCy tags as ADV in all contexts
    })

    # Capture standalone nouns not already covered by a chunk or MWE scan.
    for token in doc:
        if (token.pos_ == "NOUN" or token.text.lower() in _MISCLASSIFIED_NOUNS) and not token.is_stop:
            surface = token.text.lower().strip()
            # Preserve plurale tantum as-is; lemmatize everything else
            candidate = surface if surface in _PLURALE_TANTUM else token.lemma_.lower().strip()
            candidate = _clean_surface(candidate)
            if (not candidate
                    or len(candidate) < 2
                    or not _VALID_NOUN_RE.match(candidate)
                    or candidate in seen
                    or candidate in seen_lemmas
                    or candidate in _STOP_NOUNS):
                continue

            seen.add(candidate)
            seen_lemmas.add(candidate)
            nouns.append(candidate)

    return nouns


def extract_nouns(text: str) -> List[str]:
    """
    Extract nouns from VLM output text.

    Combines noun chunks (multi-word phrases like "park bench") with
    standalone nouns, deduplicates, and filters noise. Returns lowercase.

    Multi-word chunks are handled in two ways:
    - Adjective+noun ("larger lamp", "white shade") → reduced to root lemma
    - Compound noun  ("metal arm", "animation studios") → kept as full phrase

    Args:
        text: Free-text description from a VLM

    Returns:
        Deduplicated list of lowercase nouns/noun phrases, or [] on failure
    """
    if not text or not text.strip():
        return []

    if not _load_model():
        return []

    try:
        doc = _nlp(text)
        return _extract_nouns_from_doc(doc)
    except Exception as e:
        logger.error(f"noun_extractor: extraction failed: {e}")
        return []


def extract_nouns_and_subject(text: str):
    """Parse text once and return both nouns and subject.

    Equivalent to calling extract_nouns(text) + extract_subject(text) but
    with a single spaCy parse instead of two.

    Returns:
        (nouns: List[str], subject: Optional[str])
    """
    if not text or not text.strip():
        return [], None

    if not _load_model():
        return [], None

    try:
        doc = _nlp(text)
        return _extract_nouns_from_doc(doc), _extract_subject_from_doc(doc)
    except Exception as e:
        logger.error(f"noun_extractor: combined extraction failed: {e}")
        return [], None
