from sklearn.metrics import f1_score
from typing import List
import re
from tqdm import tqdm
from models.inference import generate_response

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)  # Remove articles
    text = re.sub(r'[^a-z0-9]', ' ', text)       # Remove punctuation
    text = ' '.join(text.split())                # Remove extra whitespace
    return text

def compute_exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate_model(dataset, retriever, model, tokenizer):
    total_em = 0
    total_f1 = 0
    count = 0

    for example in tqdm(dataset):
        question = example['question']
        ground_truths = [ans['text'] for ans in example['answers']]

        # Generate the answer using the RAG system
        generated_answer = generate_response(question, retriever, model, tokenizer)

        # Compute metrics for each ground truth answer
        example_em = max(compute_exact_match(generated_answer, gt) for gt in ground_truths)
        example_f1 = max(compute_f1(generated_answer, gt) for gt in ground_truths)

        total_em += example_em
        total_f1 += example_f1
        count += 1

    # Calculate average scores
    metrics = {
        'Exact Match': 100.0 * total_em / count,
        'F1 Score': 100.0 * total_f1 / count
    }

    return metrics
