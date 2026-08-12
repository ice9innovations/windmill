#!/usr/bin/env python3
"""
Systematic audit of spaCy noun misclassification patterns.

Tests common nouns across multiple categories and contexts to identify
which words are being tagged incorrectly by spaCy's en_core_web_sm model.
"""

import spacy
from collections import defaultdict

# Load spaCy model
print("Loading spaCy en_core_web_sm model...")
nlp = spacy.load("en_core_web_sm")
print("✓ Model loaded\n")

# Test words organized by category
TEST_WORDS = {
    "body_modifications": [
        "tattoo", "piercing", "scar", "bruise", "marking", "birthmark"
    ],
    "clothing_details": [
        "sequin", "zipper", "button", "collar", "cuff", "hem", "sleeve"
    ],
    "accessories": [
        "bracelet", "necklace", "earring", "ring", "watch", "pendant"
    ],
    "body_parts": [
        "torso", "limb", "abdomen", "thigh", "calf", "ankle", "wrist"
    ],
    "objects": [
        "gadget", "device", "trinket", "ornament", "decoration"
    ],
    "clothing": [
        "outfit", "costume", "uniform", "garment", "apparel"
    ],
    "personal_care": [
        "makeup", "hairstyle", "manicure", "grooming"
    ],
}

# Test contexts to check word in different grammatical positions
TEST_CONTEXTS = [
    "The {word} is beautiful",           # Subject
    "She has a {word}",                  # Object
    "A woman with a {word}",             # Prepositional object
    "The {word} on her arm",             # Subject with modifier
    "I see the {word}",                  # Direct object
]


def test_word_classification(word, contexts):
    """Test if word is correctly tagged as NOUN in various contexts.

    Returns dict with POS tags found across contexts.
    """
    pos_tags = defaultdict(int)
    deps = defaultdict(int)

    for context_template in contexts:
        context = context_template.format(word=word)
        doc = nlp(context)

        # Find the word in the parsed doc
        for token in doc:
            if token.text.lower() == word.lower():
                pos_tags[token.pos_] += 1
                deps[token.dep_] += 1
                break

    return {
        'pos_tags': dict(pos_tags),
        'deps': dict(deps),
        'is_misclassified': 'NOUN' not in pos_tags or pos_tags['NOUN'] < len(contexts)
    }


def main():
    print("=" * 80)
    print("SPACY NOUN MISCLASSIFICATION AUDIT")
    print("=" * 80)
    print()

    # Track results
    all_misclassified = []
    category_results = {}

    # Test each category
    for category, words in TEST_WORDS.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        print("-" * 80)

        misclassified_in_category = []

        for word in words:
            results = test_word_classification(word, TEST_CONTEXTS)

            if results['is_misclassified']:
                misclassified_in_category.append(word)
                all_misclassified.append({
                    'word': word,
                    'category': category,
                    'pos_tags': results['pos_tags'],
                    'deps': results['deps']
                })

                # Show details
                pos_str = ", ".join(f"{pos}({count})" for pos, count in results['pos_tags'].items())
                print(f"  ✗ {word:<20} → {pos_str}")
            else:
                print(f"  ✓ {word:<20} → NOUN (correct)")

        category_results[category] = {
            'total': len(words),
            'correct': len(words) - len(misclassified_in_category),
            'misclassified': len(misclassified_in_category),
            'accuracy': (len(words) - len(misclassified_in_category)) / len(words)
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    total_words = sum(len(words) for words in TEST_WORDS.values())
    total_correct = total_words - len(all_misclassified)
    overall_accuracy = total_correct / total_words

    print(f"Total words tested: {total_words}")
    print(f"Correctly classified: {total_correct} ({overall_accuracy:.1%})")
    print(f"Misclassified: {len(all_misclassified)} ({1-overall_accuracy:.1%})")
    print()

    # Category breakdown
    print("Accuracy by category:")
    for category, results in category_results.items():
        print(f"  {category:<25} {results['accuracy']:.1%} "
              f"({results['correct']}/{results['total']} correct)")

    # Detailed misclassifications
    if all_misclassified:
        print("\n" + "=" * 80)
        print("MISCLASSIFIED WORDS (need manual override)")
        print("=" * 80)
        print()

        for item in all_misclassified:
            pos_tags = ", ".join(f"{pos}({count})" for pos, count in item['pos_tags'].items())
            print(f"  '{item['word']}' ({item['category']}) → {pos_tags}")

        print("\n" + "=" * 80)
        print("ADD TO _MISCLASSIFIED_NOUNS:")
        print("=" * 80)
        print()
        print("_MISCLASSIFIED_NOUNS = frozenset({")
        for item in all_misclassified:
            pos_str = ", ".join(item['pos_tags'].keys())
            print(f'    "{item["word"]}",  # spaCy tags as {pos_str}')
        print("})")

    print()


if __name__ == '__main__':
    main()
