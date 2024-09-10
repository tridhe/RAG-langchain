from sklearn.metrics import f1_score
from typing import List
import re
from tqdm import tqdm
from models.inference import generate_response
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)  # Remove articles
    text = re.sub(r'[^a-z0-9]', ' ', text)       # Remove punctuation
    text = ' '.join(text.split())                # Remove extra whitespace
    return text

def compute_bleu(prediction: str, ground_truth: str) -> float:

    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
        
    weights = (0.5, 0.3, 0.15, 0.05)
    
    return sentence_bleu([ground_truth_tokens], prediction_tokens, weights=weights)

def compute_rouge(prediction: str, ground_truth: str) -> dict:

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ground_truth, prediction)['rougeL'].fmeasure

def evaluate_model(dataset, retriever, model, tokenizer):
    total_bleu = 0
    total_rouge = 0
    count = 0

    for example in tqdm(dataset):
        question = example['question']
        ground_truths = [ans['text'] for ans in example['answers']]

        # Generate the answer using the RAG system
        generated_answer = generate_response(question, retriever, model, tokenizer)

        # Compute metrics for each ground truth answer
        max_bleu = max(compute_bleu(generated_answer, gt) for gt in ground_truths)
        max_rouge = max(compute_rouge(generated_answer, gt) for gt in ground_truths)

        total_bleu += max_bleu
        total_rouge += max_rouge
        count += 1

    # Calculate average scores
    metrics = {
        'Average BLEU': 100.0 * total_bleu / count,
        'Average ROUGE-L': 100.0 * total_rouge / count
    }

    return metrics
