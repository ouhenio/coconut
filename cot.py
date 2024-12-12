from vllm import LLM, SamplingParams
from datasets import load_dataset
import re
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import json

def setup_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> LLM:
    """Initialize the vLLM model."""
    return LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=4096
    )

def extract_answer(response: str) -> Optional[float]:
    """Extract the numerical answer from the model's response."""
    # find the last number that appears after "Answer:" or "Therefore"
    answer_patterns = [
        r"(?:Answer|Therefore):?\s*\$?(\d+(?:\.\d+)?)",
        r"(?:Answer|Therefore):?\s*(\d+(?:\.\d+)?)",
        r"The final answer is:?\s*\$?(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*$"  # Last number in the string
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    
    # if no patterns match, try to find the last number in the text
    numbers = re.findall(r'\d+(?:\.\d+)?', response)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    
    return None

def prepare_prompt(question: str) -> str:
    """Prepare the prompt for the model."""
    return f"""Question: {question}
Let's solve this step by step:
1)"""

def evaluate_gsm8k(
    model: LLM,
    batch_size: int = 8,
    num_samples: Optional[int] = None,
) -> Tuple[float, List[dict]]:
    """
    Evaluate model on GSM8K test set using vLLM for efficient batched inference.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    prompts = [prepare_prompt(example['question']) for example in dataset]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        presence_penalty=0,
        frequency_penalty=0
    )
    
    print("\nGenerating responses in batches...")
    outputs = model.generate(prompts, sampling_params)
    
    results = []
    correct = 0
    total = len(dataset)
    
    print("\nProcessing results...")
    for i, (output, example) in enumerate(zip(outputs, dataset)):
        response = output.outputs[0].text
        target_answer = float(example['answer'].split()[-1])
        predicted_answer = extract_answer(response)
        
        is_correct = False
        if predicted_answer is not None:
            is_correct = abs(predicted_answer - target_answer) < 1e-6
            if is_correct:
                correct += 1
        
        results.append({
            'question': example['question'],
            'target_answer': target_answer,
            'model_response': response,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct
        })
        
        if (i + 1) % 10 == 0:
            print(f"\nCurrent accuracy: {(correct / (i + 1)) * 100:.2f}% ({correct}/{i + 1})")
    
    final_accuracy = (correct / total) * 100
    return final_accuracy, results

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    print(f"Initializing {model_name} with vLLM...")
    model = setup_model(model_name)
    
    print("\nStarting evaluation on GSM8K test set...")
    accuracy, results = evaluate_gsm8k(model)
    
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    
    print("\nExample predictions:")
    for i in range(min(5, len(results))):
        result = results[i]
        print(f"\nQuestion: {result['question']}")
        print(f"Target Answer: {result['target_answer']}")
        print(f"Predicted Answer: {result['predicted_answer']}")
        print(f"Correct: {result['is_correct']}")
    
    output_file = 'gsm8k_vllm_results.json'
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'results': results
        }, f, indent=2)

if __name__ == "__main__":
    main()