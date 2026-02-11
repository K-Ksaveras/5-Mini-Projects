"""
Named Entity Recognition - Starter Code

"""

# 1: Import NER pipeline from transformers
from transformers import pipeline

# 2: Load NER pipeline

def load_ner_model():
    """Load a pre-trained Named Entity Recognition model."""
    ner_model = pipeline("token-classification", model="dslim/bert-base-NER")
    return ner_model


# 3: Create test sentences with multiple entities

def get_test_sentences():
    """Get test sentences containing named entities."""
    sentences = [
    "Steve Jobs founded Apple in California.",
    "Elon Musk works at Tesla in San Francisco.",
    "Google CEO Sundar Pichai is based in Mountain View.",
    "Jeff Bezos created Amazon and is from New Mexico.",
    "The World Health Organization is headquartered in Geneva."
]
    return sentences

# 4: Extract entities from each sentence

def extract_entities(model, sentences):
    """Extract named entities from a list of sentences."""
    results = {}
    # 4: Loop through sentences and extract entities
    for sentence in sentences:           
        entities = model(sentence)
        results[sentence] = entities
    return results

# 5: Organize entities by type

def group_entities_by_type(entities):
    """Group extracted entities by their type."""
    grouped = {}
    # 5: Group entities by their type (PERSON, ORG, LOC, etc)
    for sentence in entities:
        for entity in entities[sentence]:
            entity_type = entity['entity']
            word = entity['word']
            if entity_type not in grouped:
                grouped[entity_type] = []  # Create list if doesn't exist
            grouped[entity_type].append(word)
    return grouped


# 6: Display results clearly organized by entity type

def display_results(grouped_entities):
    """Display entities organized by their type."""
    for entity_type in grouped_entities:
        print(f"{entity_type}: {grouped_entities[entity_type]}")



if __name__ == "__main__":
    print("\nMODULE 3: NAMED ENTITY RECOGNITION\n")

    model = load_ner_model()
    sentences = get_test_sentences()
    entities = extract_entities(model, sentences)
    grouped = group_entities_by_type(entities)
    display_results(grouped)