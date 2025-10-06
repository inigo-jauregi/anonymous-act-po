import os
import re

from openai import OpenAI
import numpy as np
from tqdm import tqdm


def extract_bracket_content(text):
    pattern = r'\{([^}]+)\}'  # Matches content between curly brackets
    match = re.search(pattern, text)
    if match:
        return match.group(1)  # Returns only the content inside brackets
    else:
        raise ValueError(f"No bracketed content found in: {text}")


def gpt_formality_evaluation(predictions):

    # Get the api_key as environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    system_prompt = """You are a formality evaluation expert. Analyze the given text and determine if it is formal or informal.
    Respond with only {formal} or {informal} using the brackets."""

    formality_scores = []
    print('Evaluating with GPT-as-judge...')
    for pred, label in tqdm(zip(predictions['prediction_list'], predictions['attribute_label_list']), total=len(predictions['attribute_label_list'])):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text: {pred}"}
                ],
                temperature=0,
            )

            # Get GPT's evaluation
            raw_eval = response.choices[0].message.content.lower().strip()
            gpt_eval = extract_bracket_content(raw_eval).lower()

            # Compare with intended formality
            matches_intended = (gpt_eval == "formal" and label == "formal") or \
                             (gpt_eval == "informal" and label == "informal")

            formality_scores.append(1 if matches_intended else 0)

        except Exception as e:
            print(f"Error evaluating text: {e}")
            formality_scores.append(0)

    # Calculate average accuracy
    gpt_formality_accuracy = np.mean(formality_scores)

    return gpt_formality_accuracy
