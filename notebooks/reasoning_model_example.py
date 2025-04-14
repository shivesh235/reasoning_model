#!/usr/bin/env python
# coding: utf-8

# # Reasoning Model Example
# 
# This script demonstrates how to train and use the reasoning model.

import os
import sys
import torch
import re

# Add the src directory to the path
sys.path.append(os.path.join('..', 'src'))

# Import the necessary modules
from model import load_base_model, create_lora_config, save_model, load_finetuned_model
from data import load_gsm8k_dataset, prepare_training_dataset, get_training_messages
from rewards import extract_model_answer, combined_reward

print("# 1. Load the Base Model")
print("First, we'll load the Llama 3.1-8B model using UnslothAI.")

# Load the base model
model_name = "meta-llama/Meta-Llama-3.1-8B"  # Replace with actual accessible model if needed
print(f"Loading model: {model_name}")
# Uncomment to actually load the model:
# model, tokenizer = load_base_model(model_name)
print("(Model loading is commented out in this demo script)")

print("\n# 2. Apply LoRA Configuration")
print("Next, we'll apply LoRA to allow for efficient fine-tuning.")

# Apply LoRA
print("Applying LoRA configuration...")
# Uncomment to actually apply LoRA:
# model = create_lora_config(
#     model,
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05
# )
print("(LoRA configuration is commented out in this demo script)")

print("\n# 3. Load and Prepare the Dataset")
print("We'll use the GSM8K dataset for training on mathematical reasoning.")

# Load a small portion of the dataset for demonstration
print("Loading GSM8K dataset...")
# Uncomment to actually load the dataset:
# gsm8k_dataset = load_gsm8k_dataset("train")
# examples = prepare_training_dataset(gsm8k_dataset, max_samples=10, seed=42)
# messages = get_training_messages(examples)

# Sample data for demonstration
sample_system_prompt = """You are a reasoning assistant. When solving problems, you must:
1. Break down the problem into steps
2. Think through each step carefully
3. Show all your work in a clear, organized manner
4. Always verify your answer before finalizing it
5. Provide your final answer in the format: <answer>X</answer> where X is the numerical value

Let me solve this step-by-step:"""

sample_user_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four eggs in each batch of muffins. She bakes 7 batches of muffins every week. How many eggs does she have left after 4 weeks?"

sample_assistant_answer = """To solve this problem, I need to calculate how many eggs Janet has left after 4 weeks.

1. First, I'll find out how many eggs her ducks lay in 4 weeks:
   Daily eggs = 16 eggs per day
   Total eggs in 4 weeks = 16 eggs per day × 7 days per week × 4 weeks = 448 eggs

2. Next, I'll calculate how many eggs Janet eats for breakfast in 4 weeks:
   Daily breakfast eggs = 3 eggs
   Total breakfast eggs in 4 weeks = 3 eggs per day × 7 days per week × 4 weeks = 84 eggs

3. Now I'll find out how many eggs she uses for muffins in 4 weeks:
   Weekly muffin eggs = 7 batches × 4 eggs per batch = 28 eggs per week
   Total muffin eggs in 4 weeks = 28 eggs per week × 4 weeks = 112 eggs

4. Finally, I'll calculate how many eggs she has left:
   Eggs left = Total eggs - Breakfast eggs - Muffin eggs
   Eggs left = 448 - 84 - 112 = 252 eggs

Therefore, the final answer is <answer>252</answer>"""

print("\nSample System Prompt:")
print(sample_system_prompt)
print("\nSample User Question:")
print(sample_user_question)
print("\nSample Expected Answer (excerpt):")
print(sample_assistant_answer[:200] + "...")

print("\n# 4. Generate a Response with the Base Model")
print("Let's see how the base model responds to a math problem.")

def format_prompt(prompt):
    system_prompt = """You are a reasoning assistant. When solving problems, you must:
1. Break down the problem into steps
2. Think through each step carefully
3. Show all your work in a clear, organized manner
4. Always verify your answer before finalizing it
5. Provide your final answer in the format: <answer>X</answer> where X is the numerical value

Let me solve this step-by-step:"""
    
    return f"{system_prompt}\n\nQuestion: {prompt}\n\n"

# Sample problem
sample_problem = "If John has 5 apples and Mary has 3 more apples than John, how many apples do they have in total?"

# Format the prompt
formatted_prompt = format_prompt(sample_problem)
print("Formatted prompt created.")

# Generate a response with the base model - simulated for demo purposes
print("Simulated base model response:")
simulated_base_response = """
To solve this problem, I need to find the total number of apples.

John has 5 apples.
Mary has 3 more apples than John, so she has 5 + 3 = 8 apples.

The total number of apples is 5 + 8 = 13 apples.

So John and Mary have 13 apples in total.
"""
print(simulated_base_response)

print("\n# 5. Brief Overview of GRPO Training")
print("""
In practice, we would now train the model using GRPO with our reward functions. For training, we would run the `train.py` script.

The key steps in GRPO training are:

1. Generate responses with the current model
2. Compute rewards using our defined reward functions
3. Update the model weights using the GRPO loss function
4. Repeat for multiple epochs

Since training takes several hours, we skip it in this demo script.
""")

print("\n# 6. Using a Trained Model (Simulated Example)")
print("This is how we would use the trained model for inference. Below we're simulating what a trained model's output might look like.")

# This is what a reasoning model's response would look like after training
simulated_trained_response = """
To solve this problem, I need to find the total number of apples that John and Mary have together.

1. Given:
   - John has 5 apples
   - Mary has 3 more apples than John

2. First, I'll calculate how many apples Mary has:
   Mary's apples = John's apples + 3
   Mary's apples = 5 + 3 = 8

3. Now, I'll find the total number of apples:
   Total apples = John's apples + Mary's apples
   Total apples = 5 + 8 = 13

4. Let me verify: John has 5 apples, Mary has 8 apples, and 5 + 8 = 13.

Therefore, the final answer is <answer>13</answer>
"""

print("Simulated trained model response:")
print(simulated_trained_response)

# Extract the answer
match = re.search(r"<answer>(.*?)</answer>", simulated_trained_response)
if match:
    print("\nExtracted answer:", match.group(1))

print("\n# 7. Comparison: Before and After Training")
print("""
The key differences in the responses:

1. Structured reasoning: The trained model shows clear step-by-step reasoning
2. Format compliance: The trained model provides the answer in the specified format
3. Verification step: The trained model explicitly verifies its answer
4. Clear final answer: The answer is prominently displayed in the correct format

These improvements help users follow the reasoning process and easily extract the final answer.
""")

print("\n# 8. Conclusion")
print("""
By fine-tuning a base LLM with GRPO and carefully designed reward functions, we can create a reasoning model that:

1. Shows its work in a clear, step-by-step manner
2. Follows a consistent format for presenting answers
3. Verifies its results before providing a final answer
4. Produces more accurate responses for mathematical and reasoning tasks

This approach can be extended to other types of reasoning tasks beyond mathematics.
""") 