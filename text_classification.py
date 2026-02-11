"""
Text Classification - Starter Code

"""

# 1: Import libraries
# You need: transformers, torch, sys
from transformers import pipeline

print('Imports successful!')


# 2: Create a simple text classification pipeline


def create_classifier():
    """Create a zero-shot classification pipeline."""
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier


# 3: Test the classifier with a simple sentence

def test_simple_classification(classifier):
    """Test the classifier with a simple sentence."""
    test_text = "This is a great product!"
    labels = ["positive", "negative", "neutral"]
    result = classifier(test_text, labels)

    print(result)


# 4: Process multiple texts at once

def classify_multiple_texts(classifier):
    """Classify multiple texts and display predictions with confidence scores."""
    texts = [
        "I absolutely love this product!",
        "This is the worst experience ever.",
        "The quality is decent, nothing special.",
        "Amazing! Exceeded all my expectations!",
        "Terrible waste of money, very disappointed."
    ]
    labels = ["positive", "negative", "neutral"]
    for text in texts:
        result = classifier(text, labels, multi_label=False)
        print(f"Text: {text}")
        print(f"Prediction: {result['labels'][0]} ({result['scores'][0]:.2f})")
        print()


# 5: Extract and display predictions clearly

def format_prediction(text, result):
    """Format classification result as a readable string."""
    top_label = result['labels'][0]
    confidence = result['scores'][0]
    formatted_output = f"Text: {text} | Prediction: {top_label} ({confidence:.2f})"
    return formatted_output
    


# Main execution
if __name__ == "__main__":
    print("  MODULE 1: TEXT CLASSIFICATION STARTER")

    classifier = create_classifier()
    test_simple_classification(classifier)
    classify_multiple_texts(classifier)