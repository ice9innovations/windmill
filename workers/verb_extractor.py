#!/usr/bin/env python3
"""
Verb Extractor - spaCy-based verb and SVO triple extraction from VLM captions.

Used by noun_consensus_worker to extract verbs and SVO triples from raw captions
stored in the results table. Parallel to noun_extractor.py — extraction runs in
the windmill layer, not in individual VLM services.

The primary entry point is extract_verbs_and_svo() which runs a single spaCy
parse and returns both results in one pass.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Loaded once at import time - en_core_web_lg, ~560MB, CPU-only
_nlp = None


def _load_model():
    global _nlp
    if _nlp is not None:
        return True
    try:
        import spacy
        _nlp = spacy.load("en_core_web_lg")
        logger.info("verb_extractor: en_core_web_lg loaded")
        return True
    except OSError:
        logger.error("verb_extractor: en_core_web_lg not found - run: "
                     "python -m spacy download en_core_web_lg")
        return False
    except ImportError:
        logger.error("verb_extractor: spacy not installed")
        return False


def warmup_verb_extractor() -> bool:
    """Load the spaCy model at worker startup so the first message pays no
    disk-load penalty mid-processing. Safe to call multiple times."""
    return _load_model()


# Verbs that carry no scene content — grammatical function words and
# verbs so generic they communicate nothing about what is happening.
_STOP_VERBS = {
    "be", "have", "do", "say", "make", "go", "know", "get",
    "see", "look", "seem", "appear", "become",
}

# Dependency labels that mark grammatical (non-content) verb roles.
_STOP_DEPS = frozenset({'aux', 'auxpass', 'cop'})

# Past participles (VBN) used as modifiers are not content verbs:
#   dep_=acl:   "a meadow dotted with flowers"
#   dep_=amod:  "forested hills"
#   dep_=relcl: relative clause modifiers
#   dep_=advcl: adverbial participial phrases — "set against a backdrop"
# Present participles (VBG) with these same dep_ labels ARE content verbs
# when dep_=acl ("a horse running/galloping") and are kept.
_VBN_MODIFIER_DEPS = frozenset({'acl', 'amod', 'relcl', 'advcl'})

# Present participles used purely as adjectives (not actions):
#   dep_=amod: "rolling hills", "running water"
# VBG with dep_=acl is kept — "a horse running/galloping" is content.
_VBG_ADJ_DEPS = frozenset({'amod'})

# Dependency labels used to locate subjects and objects in SVO extraction.
_SUBJECT_DEPS = frozenset({'nsubj', 'nsubjpass'})
_OBJECT_DEPS  = frozenset({'dobj', 'pobj', 'attr'})

# A valid verb lemma contains only lowercase letters and hyphens.
# Rejects tokens with digits, punctuation bleed, or whitespace.
_VALID_VERB_RE = re.compile(r'^[a-z][a-z\-]*$')


def extract_verbs_and_svo(text: str) -> Tuple[List[str], List[List[str]]]:
    """
    Extract content verbs and SVO triples from VLM output text in a single parse.

    Verbs: lemmatized, deduplicated, auxiliaries/copulas/stop verbs removed.
    SVO triples: [subject, verb, object] lists from dependency parse. Only
    emitted when both subject and object are present — partial pairs are noise.
    All elements are lemmatized. Triples are deduplicated.

    Args:
        text: Free-text description from a VLM

    Returns:
        (verbs, svo_triples) where verbs is List[str] and
        svo_triples is List[List[str]]. Both are [] on failure.
    """
    if not text or not text.strip():
        return [], []

    if not _load_model():
        return [], []

    try:
        doc = _nlp(text)

        seen_verbs = set()
        verbs = []
        seen_triples = set()
        triples = []

        for token in doc:
            if token.pos_ != "VERB":
                continue
            if token.dep_ in _STOP_DEPS:
                continue
            # Past participles used as modifiers are not content verbs.
            if token.tag_ == 'VBN' and token.dep_ in _VBN_MODIFIER_DEPS:
                continue
            # Present participles used purely as adjectives are not content verbs.
            # VBG+acl is kept — "a horse running/galloping" IS content.
            if token.tag_ == 'VBG' and token.dep_ in _VBG_ADJ_DEPS:
                continue

            lemma = token.lemma_.lower().strip()

            if (not lemma
                    or lemma in _STOP_VERBS
                    or not _VALID_VERB_RE.match(lemma)):
                continue

            if lemma not in seen_verbs:
                seen_verbs.add(lemma)
                verbs.append(lemma)

            subjects = [w for w in token.lefts  if w.dep_ in _SUBJECT_DEPS]
            objects  = [w for w in token.rights if w.dep_ in _OBJECT_DEPS]

            if not subjects or not objects:
                continue

            for subj in subjects:
                for obj in objects:
                    key = (subj.lemma_.lower(), lemma, obj.lemma_.lower())
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    triples.append(list(key))

        return verbs, triples

    except Exception as e:
        logger.error(f"verb_extractor: extraction failed: {e}")
        return [], []
