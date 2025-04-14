import torch
from unsloth import FastLanguageModel
from peft import LoraConfig
import os

def load_base_model(model_name="meta-llama/Meta-Llama-3.1-8B", cache_dir=None):
    """
    Load the base LLM model with UnslothAI
    
    Args:
        model_name (str): HuggingFace model name
        cache_dir (str, optional): Directory to cache model weights
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        cache_dir=cache_dir,
        load_in_4bit=True,
    )
    
    print("Base model loaded successfully")
    
    return model, tokenizer

def create_lora_config(model, 
                       r=16, 
                       lora_alpha=32, 
                       lora_dropout=0.05, 
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
    """
    Create a LoRA configuration for efficient fine-tuning
    
    Args:
        model: The loaded model
        r (int): Rank of the low-rank adaptation matrices
        lora_alpha (int): Scaling factor for the LoRA adaptations
        lora_dropout (float): Dropout probability for LoRA layers
        target_modules (list): List of module names to apply LoRA to
        
    Returns:
        model: The model with LoRA applied
    """
    print("Applying LoRA configuration")
    
    model = FastLanguageModel.get_peft_model(
        model,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        target_modules=target_modules,
        use_gradient_checkpointing=True,
        random_state=42,
    )
    
    print("LoRA configuration applied successfully")
    
    return model

def save_model(model, tokenizer, output_dir="models/reasoning_model"):
    """
    Save the fine-tuned model and tokenizer
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        output_dir (str): Directory to save the model to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully")

def load_finetuned_model(model_path="models/reasoning_model"):
    """
    Load a fine-tuned model for inference
    
    Args:
        model_path (str): Path to the fine-tuned model
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading fine-tuned model from {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    print("Fine-tuned model loaded successfully")
    
    return model, tokenizer 