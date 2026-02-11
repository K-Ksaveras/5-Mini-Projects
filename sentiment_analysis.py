"""
Sentiment Analysis - Starter Code

"""

# 1: Import transformers pipeline
from transformers import pipeline

# 2: Load sentiment analysis pipeline

def load_sentiment_model():
    """Load a pre-trained sentiment analysis model."""
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_model


# 3: Create sample movie reviews dataset

def get_sample_reviews():
    """Get a collection of sample movie reviews for testing."""
    reviews = [
    "This movie was absolutely amazing! Best film I've seen all year!",
    "Terrible waste of time. The plot was boring and predictable.",
    "Great cinematography but the story was a bit confusing.",
    "I loved every minute of it! The acting was superb!",
    "Disappointing. Expected much better from this director.",
    "Fantastic! The soundtrack alone made it worth watching.",
    "Average movie. Not bad, not great either.",
    "One of the worst films I've ever seen. Completely disappointed."
]
    return reviews


# 4a: Classify each review

def analyze_reviews(model, reviews):
    """Analyze sentiment for a list of reviews using batch inference."""
    
    predictions = model(reviews)

    results = []

    # 4b: Loop over reviews and predictions together

    for review, pred in zip(reviews, predictions):
        sentiment = pred['label']
        score = pred['score']
        if score < 0.7:
            sentiment = "NEUTRAL"
        results.append((review, sentiment, score))

    return results


# 5: Display results formatted nicely

def display_results(results):
    """Display sentiment analysis results formatted with sentiment labels and scores."""
    for review, sentiment, score in results:
        print(f"Review: {review} | Sentiment: {sentiment} ({score:.2f})")


# Main execution block to run the sentiment analysis pipeline

if __name__ == "__main__":
    print("\nMODULE 2: SENTIMENT ANALYSIS\n")

    print("Loading model...")
    model = load_sentiment_model()
    print("Model loaded!")

    print("Getting reviews...")
    reviews = get_sample_reviews()
    print(f"Loaded {len(reviews)} reviews")

    print("Analyzing reviews...")
    results = analyze_reviews(model, reviews)
    print("Analysis complete!")

    print("Displaying results:\n")
    display_results(results)
