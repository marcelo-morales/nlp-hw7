from transformers import pipeline

# Load BLOOMZ model
model_name = "bloom-z/distilbert-base-uncased"
classifier = pipeline("text-classification", model=model_name)

# Example dataset
examples = [
    {"text": "I really enjoyed this movie!"},
    {"text": "This product is terrible."},
    {"text": "I'm not sure what to think about this book."},
    # Add more examples here...
]

# Generate predictions
predictions = classifier([ex["text"] for ex in examples])

# Print results
for i, ex in enumerate(examples):
    print(f"Example {i+1}:")
    print(f"  Text: {ex['text']}")
    print(f"  Prediction: {predictions[i]['label']}")
    print(f"  Score: {predictions[i]['score']:.2f}")
