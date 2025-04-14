import datasets
from datasets import load_dataset
import random
import re
import numpy as np

def load_gsm8k_dataset(split="train", cache_dir=None):
    """
    Load the GSM8K dataset for math word problems
    
    Args:
        split (str): Dataset split to load ('train' or 'test')
        cache_dir (str, optional): Directory to cache dataset
        
    Returns:
        dataset: Loaded dataset
    """
    print(f"Loading GSM8K {split} dataset")
    
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    print(f"Loaded {len(dataset)} examples")
    
    return dataset

def extract_answer(solution):
    """
    Extract the final numerical answer from the solution
    
    Args:
        solution (str): The full solution text
        
    Returns:
        float: The extracted answer
    """
    # Look for the final answer pattern (typically "The answer is X")
    last_line = solution.strip().split('\n')[-1]
    
    # Try to find any number in the last line
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", last_line)
    if matches:
        return float(matches[-1])
    
    # If no number found in the last line, search the entire solution
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", solution)
    if matches:
        return float(matches[-1])
    
    return None

def format_with_reasoning_prompt(example):
    """
    Format an example with a system prompt that encourages structured reasoning
    
    Args:
        example (dict): Dataset example
        
    Returns:
        dict: Formatted example with prompts
    """
    system_prompt = """You are a reasoning assistant. When solving problems, you must:
1. Break down the problem into steps
2. Think through each step carefully
3. Show all your work in a clear, organized manner
4. Always verify your answer before finalizing it
5. Provide your final answer in the format: <answer>X</answer> where X is the numerical value

Let me solve this step-by-step:"""

    question = example['question']
    
    # Format the answer to include step-by-step reasoning
    solution = example['answer']
    answer = extract_answer(solution)
    
    if answer is not None:
        formatted_solution = f"{solution}\n\nTherefore, the final answer is <answer>{answer}</answer>"
    else:
        formatted_solution = solution
    
    return {
        "system_prompt": system_prompt,
        "question": question,
        "answer": formatted_solution
    }

def prepare_training_dataset(dataset, max_samples=None, seed=42):
    """
    Prepare the dataset for training by formatting examples
    
    Args:
        dataset: The loaded dataset
        max_samples (int, optional): Maximum number of samples to use
        seed (int): Random seed for reproducibility
        
    Returns:
        list: Formatted examples ready for training
    """
    if max_samples and max_samples < len(dataset):
        random.seed(seed)
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    
    print(f"Preparing {len(dataset)} examples for training")
    
    formatted_dataset = dataset.map(format_with_reasoning_prompt)
    
    # Convert to the format expected by the training loop
    training_examples = []
    for example in formatted_dataset:
        training_examples.append({
            "system_prompt": example["system_prompt"],
            "question": example["question"],
            "answer": example["answer"]
        })
    
    return training_examples

def get_training_messages(examples):
    """
    Convert examples to the message format required for training
    
    Args:
        examples (list): List of formatted examples
        
    Returns:
        list: Messages in the format required by the training function
    """
    messages = []
    
    for example in examples:
        message = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]
        messages.append(message)
    
    return messages 