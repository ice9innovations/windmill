#!/usr/bin/env python3
"""
Compare spaCy models (sm vs md vs lg) for noun extraction accuracy.

Tests both systematic test words and production caption sample.
"""

import os
import sys
import psycopg2
import spacy
from collections import defaultdict, Counter
from dotenv import load_dotenv

# Load environment
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, '.env'))

# Systematic test words (from previous testing)
SYSTEMATIC_TEST_WORDS = {
    "tattoo", "piercing", "pendant", "ornament", "bedspread",
    "scar", "bruise", "bikini", "necklace", "watch"
}

# Test contexts
TEST_CONTEXTS = [
    "The {word} is beautiful",
    "She has a {word}",
    "A woman with a {word}",
]

# Models to compare
MODELS_TO_TEST = [
    ("en_core_web_sm", "Small (12MB)"),
    ("en_core_web_md", "Medium (48MB)"),
]

# Try to add large model if available
try:
    import en_core_web_lg
    MODELS_TO_TEST.append(("en_core_web_lg", "Large (560MB)"))
except:
    pass


def test_model_on_systematic_words(nlp, model_name):
    """Test model on known problem words."""
    errors = 0

    for word in SYSTEMATIC_TEST_WORDS:
        word_errors = 0
        for context_template in TEST_CONTEXTS:
            context = context_template.format(word=word)
            doc = nlp(context)

            # Find the word
            for token in doc:
                if token.text.lower() == word.lower():
                    if token.pos_ not in ('NOUN', 'PROPN'):
                        word_errors += 1
                    break

        if word_errors > 0:
            errors += 1

    accuracy = (len(SYSTEMATIC_TEST_WORDS) - errors) / len(SYSTEMATIC_TEST_WORDS)
    return accuracy, errors


def test_model_on_production_captions(nlp, model_name, sample_size=100):
    """Test model on production VLM captions."""
    print(f"  Fetching production captions...")

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        sslmode=os.getenv('DB_SSLMODE', 'prefer')
    )
    cursor = conn.cursor()

    # Get sample of VLM captions
    query = """
        SELECT data->'predictions'->0->>'text' as caption
        FROM results
        WHERE service IN ('blip', 'moondream', 'qwen', 'haiku', 'ollama')
        AND status = 'success'
        AND data->'predictions'->0->>'text' IS NOT NULL
        ORDER BY result_id DESC
        LIMIT %s
    """
    cursor.execute(query, (sample_size,))

    captions = []
    for row in cursor.fetchall():
        if row[0]:
            captions.append(row[0])

    cursor.close()
    conn.close()

    # Analyze captions
    noun_count = 0
    propn_count = 0
    total_content_words = 0

    for caption in captions:
        doc = nlp(caption)
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 1:
                total_content_words += 1
                if token.pos_ == 'NOUN':
                    noun_count += 1
                elif token.pos_ == 'PROPN':
                    propn_count += 1

    noun_ratio = (noun_count + propn_count) / total_content_words if total_content_words > 0 else 0

    return {
        'captions': len(captions),
        'noun_count': noun_count,
        'propn_count': propn_count,
        'total_content_words': total_content_words,
        'noun_ratio': noun_ratio
    }


print("=" * 80)
print("SPACY MODEL COMPARISON")
print("=" * 80)
print()

results = {}

for model_id, model_desc in MODELS_TO_TEST:
    print(f"\nTesting {model_desc} ({model_id})...")
    print("-" * 80)

    try:
        # Load model
        print(f"  Loading model...")
        nlp = spacy.load(model_id)

        # Test on systematic words
        print(f"  Testing {len(SYSTEMATIC_TEST_WORDS)} known problem words...")
        sys_accuracy, sys_errors = test_model_on_systematic_words(nlp, model_id)

        # Test on production captions
        prod_stats = test_model_on_production_captions(nlp, model_id, sample_size=100)

        results[model_id] = {
            'desc': model_desc,
            'systematic_accuracy': sys_accuracy,
            'systematic_errors': sys_errors,
            'production': prod_stats,
            'available': True
        }

        print(f"  ✓ Systematic test: {sys_accuracy:.1%} accuracy ({sys_errors} errors)")
        print(f"  ✓ Production test: {prod_stats['noun_ratio']:.1%} nouns of content words")

    except OSError:
        print(f"  ✗ Model not installed")
        results[model_id] = {'available': False, 'desc': model_desc}

# Summary comparison
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print()

print(f"{'Model':<20} {'Size':<15} {'Systematic':<15} {'Noun Ratio':<15} {'Status'}")
print("-" * 80)

for model_id, model_desc in MODELS_TO_TEST:
    if results[model_id]['available']:
        print(f"{model_id:<20} {model_desc:<15} "
              f"{results[model_id]['systematic_accuracy']:.1%} ({results[model_id]['systematic_errors']} err)    "
              f"{results[model_id]['production']['noun_ratio']:.1%}            "
              f"✓")
    else:
        print(f"{model_id:<20} {model_desc:<15} {'N/A':<15} {'N/A':<15} Not installed")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

# Find best model
best_model = None
best_accuracy = 0

for model_id, data in results.items():
    if data['available'] and data['systematic_accuracy'] > best_accuracy:
        best_accuracy = data['systematic_accuracy']
        best_model = model_id

if best_model:
    improvement = (best_accuracy - results['en_core_web_sm']['systematic_accuracy']) * 100
    if improvement > 5:
        print(f"✓ {best_model} shows {improvement:.1f}% improvement - RECOMMENDED")
    elif improvement > 0:
        print(f"~ {best_model} shows {improvement:.1f}% improvement - marginal gain")
    else:
        print(f"✗ No significant improvement - stick with en_core_web_sm")
else:
    print("Unable to determine best model")

print()
