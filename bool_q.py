import openai
import random

from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("API_KEY")


# Set up the BoolQ dataset
dataset = openai.Dataset.retrieve("boolq")

# Select 8 random examples from BoolQ
examples = random.sample(dataset["data"], 8)

# Create the prompt by interleaving the examples
prompt = ""
for i, example in enumerate(examples):
    prompt += example["paragraph"] + "\n"
    prompt += "Q: " + example["question"] + "\n"
    if i < 4:
        prompt += "A: Yes\n"
    else:
        prompt += "A: No\n"
    prompt += "\n"

# Print the prompt for verification
print(prompt)

# Evaluate GPT3 Davinci on a small subset of BoolQ instances
response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=64,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print the model's response
print(response.choices[0].text.strip())
