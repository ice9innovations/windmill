#!/usr/bin/env python3
"""
Analyze all production VLM captions to find spaCy misclassifications.

Collects all words tagged as non-NOUN but appearing in noun positions,
then ranks by frequency to identify systematic misclassifications.
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

# VLM services to analyze
VLM_SERVICES = ['blip', 'ollama', 'haiku', 'moondream', 'qwen', 'gemini', 'gpt_nano', 'florence2']

# Noun-like dependency labels (positions where nouns typically appear)
NOUN_POSITIONS = {
    'nsubj',      # nominal subject
    'nsubjpass',  # passive nominal subject
    'dobj',       # direct object
    'pobj',       # object of preposition
    'attr',       # attribute
    'appos',      # appositional modifier
}

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("✓ Model loaded\n")

print("Connecting to database...")
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    sslmode=os.getenv('DB_SSLMODE', 'prefer')
)
cursor = conn.cursor()
print("✓ Connected\n")

# Fetch all VLM captions
print("Fetching VLM captions from production...")
query = """
    SELECT service, data
    FROM results
    WHERE service = ANY(%s)
    AND status = 'success'
    ORDER BY result_id
"""
cursor.execute(query, (VLM_SERVICES,))
results = cursor.fetchall()
print(f"✓ Found {len(results)} VLM results\n")

# Extract captions
captions = []
for service, data in results:
    predictions = data.get('predictions', [])
    for pred in predictions:
        if 'text' in pred:
            captions.append({
                'service': service,
                'text': pred['text']
            })

print(f"Total captions to analyze: {len(captions)}\n")
print("=" * 80)
print("Analyzing captions for misclassified nouns...")
print("=" * 80)

# Track non-nouns in noun positions
non_noun_in_noun_position = defaultdict(lambda: {
    'count': 0,
    'pos_tags': Counter(),
    'deps': Counter(),
    'contexts': []
})

# Analyze each caption
for i, caption_data in enumerate(captions):
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(captions)} captions...")

    caption = caption_data['text']

    try:
        doc = nlp(caption)

        for token in doc:
            # Skip stopwords and punctuation
            if token.is_stop or token.is_punct:
                continue

            # Look for non-NOUN words in noun positions
            if token.pos_ != 'NOUN' and token.dep_ in NOUN_POSITIONS:
                word = token.text.lower()

                # Skip single characters and numbers
                if len(word) < 2 or word.isdigit():
                    continue

                non_noun_in_noun_position[word]['count'] += 1
                non_noun_in_noun_position[word]['pos_tags'][token.pos_] += 1
                non_noun_in_noun_position[word]['deps'][token.dep_] += 1

                # Save example contexts (up to 3 per word)
                if len(non_noun_in_noun_position[word]['contexts']) < 3:
                    non_noun_in_noun_position[word]['contexts'].append(caption)

    except Exception as e:
        print(f"  Error processing caption: {e}")
        continue

print(f"  Processed {len(captions)}/{len(captions)} captions\n")

# Sort by frequency
sorted_words = sorted(
    non_noun_in_noun_position.items(),
    key=lambda x: x[1]['count'],
    reverse=True
)

# Display results
print("=" * 80)
print("NON-NOUN WORDS IN NOUN POSITIONS (Top 50)")
print("=" * 80)
print(f"\n{'Word':<20} {'Count':<8} {'POS Tags':<30} {'Deps'}")
print("-" * 80)

for word, data in sorted_words[:50]:
    pos_str = ", ".join(f"{pos}({count})" for pos, count in data['pos_tags'].most_common(3))
    dep_str = ", ".join(f"{dep}({count})" for dep, count in data['deps'].most_common(3))
    print(f"{word:<20} {data['count']:<8} {pos_str:<30} {dep_str}")

# Identify likely misclassifications (high frequency, consistent POS tag)
print("\n" + "=" * 80)
print("LIKELY MISCLASSIFICATIONS (appearing 5+ times)")
print("=" * 80)
print()

likely_misclassified = []
for word, data in sorted_words:
    if data['count'] >= 5:
        # Get most common POS tag
        most_common_pos = data['pos_tags'].most_common(1)[0]
        pos_tag, pos_count = most_common_pos

        # If consistently tagged as same non-NOUN, likely misclassified
        if pos_count / data['count'] >= 0.7:  # 70% consistency
            likely_misclassified.append((word, data))

            print(f"'{word}' - appears {data['count']} times as {pos_tag}")
            print(f"  Example: {data['contexts'][0][:80]}...")
            print()

if likely_misclassified:
    print("=" * 80)
    print("RECOMMENDED ADDITIONS TO _MISCLASSIFIED_NOUNS:")
    print("=" * 80)
    print()
    for word, data in likely_misclassified[:20]:  # Top 20
        most_common_pos = data['pos_tags'].most_common(1)[0][0]
        print(f'    "{word}",  # spaCy tags as {most_common_pos} ({data["count"]} occurrences)')

cursor.close()
conn.close()
print()
