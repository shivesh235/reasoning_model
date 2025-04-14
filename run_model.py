#!/usr/bin/env python3
"""
Reasoning Model CLI
------------------
This script provides a command-line interface to interact with the reasoning model.
"""

import argparse
import os
import sys
from src.model import load_finetuned_model
from src.inference import generate_reasoning, extract_answer

def parse_args():
    parser = argparse.ArgumentParser(description="Interact with the reasoning model")
    parser.add_argument("--model_path", type=str, default="models/reasoning_model",
                        help="Path to the fine-tuned model")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--prompt", type=str,
                        help="The prompt to generate a response for")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("You need to train the model first using src/train.py")
        sys.exit(1)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_finetuned_model(args.model_path)
    print("Model loaded successfully")
    
    if args.interactive:
        # Interactive mode
        print("\nReasoning Model Interactive Session")
        print("Type 'exit', 'quit', or Ctrl+C to end the session")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nEnter your question: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                if not user_input.strip():
                    continue
                
                print("\nThinking...")
                response = generate_reasoning(model, tokenizer, user_input, args)
                
                # Print the response
                print("\n--- Model Response ---")
                print(response)
                
                # Extract and print the answer if available
                answer = extract_answer(response)
                if answer:
                    print("\n--- Extracted Answer ---")
                    print(answer)
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.prompt:
        # Single prompt mode
        response = generate_reasoning(model, tokenizer, args.prompt, args)
        
        # Print the response
        print("\n--- Model Response ---")
        print(response)
        
        # Extract and print the answer if available
        answer = extract_answer(response)
        if answer:
            print("\n--- Extracted Answer ---")
            print(answer)
    
    else:
        print("Either --prompt or --interactive flag must be provided")
        print("Example: python run_model.py --interactive")
        print("Example: python run_model.py --prompt 'If John has 5 apples...'")

if __name__ == "__main__":
    main() 