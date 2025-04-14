import argparse
import torch
from model import load_finetuned_model

# Initialize parser
parser = argparse.ArgumentParser(description="Run inference with a fine-tuned reasoning model")
parser.add_argument("--model_path", type=str, default="models/reasoning_model",
                   help="Path to the fine-tuned model")
parser.add_argument("--prompt", type=str, required=True,
                   help="The prompt to generate a response for")
parser.add_argument("--max_new_tokens", type=int, default=512,
                   help="Maximum number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=0.9,
                   help="Top-p sampling parameter")

def format_prompt(prompt):
    """
    Format the user prompt with the system prompt for reasoning
    
    Args:
        prompt (str): User prompt
        
    Returns:
        str: Formatted prompt
    """
    system_prompt = """You are a reasoning assistant. When solving problems, you must:
1. Break down the problem into steps
2. Think through each step carefully
3. Show all your work in a clear, organized manner
4. Always verify your answer before finalizing it
5. Provide your final answer in the format: <answer>X</answer> where X is the numerical value

Let me solve this step-by-step:"""
    
    formatted_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\n"
    return formatted_prompt

def generate_reasoning(model, tokenizer, prompt, args):
    """
    Generate a reasoned response for the given prompt
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt (str): User prompt
        args: Command line arguments
        
    Returns:
        str: Generated response
    """
    # Format the prompt
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_p=args.top_p,
            top_k=50,
            repetition_penalty=1.1,
        )
    
    # Decode the response and remove the prompt
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    return response

def extract_answer(response):
    """
    Extract the answer from the response if available
    
    Args:
        response (str): The model's response
        
    Returns:
        str: The extracted answer if found, or None
    """
    import re
    
    # Try to extract answer from the XML format
    match = re.search(r"<answer>(.*?)</answer>", response)
    if match:
        return match.group(1).strip()
    
    return None

def main():
    args = parser.parse_args()
    
    # Load the fine-tuned model
    model, tokenizer = load_finetuned_model(args.model_path)
    
    # Generate reasoning response
    response = generate_reasoning(model, tokenizer, args.prompt, args)
    
    # Print the response
    print("\n--- Model Response ---")
    print(response)
    
    # Extract and print the answer if available
    answer = extract_answer(response)
    if answer:
        print("\n--- Extracted Answer ---")
        print(answer)

if __name__ == "__main__":
    main() 