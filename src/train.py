import os
import argparse
import torch
import wandb
from tqdm import tqdm
import random
import numpy as np
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments, default_data_collator
from accelerate import Accelerator

from model import load_base_model, create_lora_config, save_model
from data import load_gsm8k_dataset, prepare_training_dataset, get_training_messages
from rewards import extract_model_answer, combined_reward, extract_model_answer

# Initialize parser
parser = argparse.ArgumentParser(description="Train a reasoning model using GRPO")
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1", 
                    help="Base model to use")
parser.add_argument("--max_samples", type=int, default=1000, 
                    help="Maximum number of training samples to use")
parser.add_argument("--output_dir", type=str, default="models/reasoning_model", 
                    help="Directory to save the model")
parser.add_argument("--num_epochs", type=int, default=3, 
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4, 
                    help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=2e-5, 
                    help="Learning rate")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed")
parser.add_argument("--use_wandb", action="store_true", 
                    help="Whether to use Weights & Biases for logging")
parser.add_argument("--wandb_project", type=str, default="reasoning-model", 
                    help="Weights & Biases project name")
parser.add_argument("--cache_dir", type=str, default=None, 
                    help="Directory to cache the model and dataset")

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GRPOTrainer:
    """
    Trainer class for Generative Reinforcement Policy Optimization (GRPO)
    """
    def __init__(
        self,
        model,
        tokenizer,
        args,
        dataset,
        reference_answers
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.dataset = dataset
        self.reference_answers = reference_answers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GRPO-specific parameters
        self.kl_coef = 0.1  # KL divergence coefficient 
        self.entropy_coef = 0.01  # Entropy coefficient
        self.clip_range = 0.2  # PPO clip range
        
        # Create accelerator
        self.accelerator = Accelerator()
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
        )
        
        # Prepare for training
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
    
    def get_model_output(self, message_batch):
        """
        Get model outputs for a batch of messages
        
        Args:
            message_batch (list): Batch of message sequences
            
        Returns:
            list: Model responses for each message
        """
        formatted_responses = []
        
        for messages in message_batch:
            # Format the prompt (system + user message only)
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            
            prompt = f"{system_msg}\n\nQuestion: {user_msg}\n\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )
            
            # Decode the response and remove the prompt
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            formatted_responses.append(response)
        
        return formatted_responses
    
    def compute_rewards(self, responses, reference_answers):
        """
        Compute rewards for model responses
        
        Args:
            responses (list): Model responses
            reference_answers (list): Reference answers
            
        Returns:
            list: Reward values for each response
        """
        rewards = []
        
        for response, ref_answer in zip(responses, reference_answers):
            reward = combined_reward(response, ref_answer)
            rewards.append(reward)
        
        return rewards
    
    def compute_grpo_loss(self, old_logprobs, current_logprobs, advantages):
        """
        Compute the GRPO loss
        
        Args:
            old_logprobs (tensor): Log probabilities from the old policy
            current_logprobs (tensor): Log probabilities from the current policy
            advantages (tensor): Advantage values
            
        Returns:
            tensor: The GRPO loss
        """
        # Calculate the ratio of new/old probabilities
        ratio = torch.exp(current_logprobs - old_logprobs)
        
        # Clipped surrogate objective
        policy_loss1 = advantages * ratio
        policy_loss2 = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        return policy_loss
    
    def train(self):
        """
        Train the model using GRPO
        """
        print(f"Training on {len(self.dataset)} examples for {self.args.num_epochs} epochs")
        
        # Initialize wandb if requested
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project)
        
        # Main training loop
        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            # Shuffle the dataset for each epoch
            random.shuffle(self.dataset)
            
            # Process in batches
            for i in tqdm(range(0, len(self.dataset), self.args.batch_size)):
                # Get the batch
                batch = self.dataset[i:i+self.args.batch_size]
                batch_answers = self.reference_answers[i:i+self.args.batch_size]
                
                # Get current model responses
                responses = self.get_model_output(batch)
                
                # Compute rewards
                rewards = torch.tensor(self.compute_rewards(responses, batch_answers), device=self.device)
                
                # Process responses for training
                response_inputs = self.tokenizer(responses, return_tensors="pt", padding=True).to(self.device)
                
                # Save old policy logprobs
                with torch.no_grad():
                    # Execute forward pass to get log probabilities
                    outputs = self.model(
                        input_ids=response_inputs["input_ids"],
                        attention_mask=response_inputs["attention_mask"],
                    )
                    old_logprobs = outputs.logits.log_softmax(-1).mean()
                
                # Forward pass with current policy
                outputs = self.model(
                    input_ids=response_inputs["input_ids"],
                    attention_mask=response_inputs["attention_mask"],
                    labels=response_inputs["input_ids"],
                )
                current_logprobs = outputs.logits.log_softmax(-1).mean()
                
                # Compute GRPO loss
                policy_loss = self.compute_grpo_loss(old_logprobs, current_logprobs, rewards)
                
                # Backward pass and optimization
                self.accelerator.backward(policy_loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Log metrics
                if self.args.use_wandb:
                    wandb.log({
                        "policy_loss": policy_loss.item(),
                        "mean_reward": rewards.mean().item(),
                    })
            
            # Save checkpoint after each epoch
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch+1}")
            save_model(self.model, self.tokenizer, checkpoint_dir)
            
        # Save final model
        save_model(self.model, self.tokenizer, self.args.output_dir)
        
        # Close wandb
        if self.args.use_wandb:
            wandb.finish()
        
        print("Training completed")

def main():
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load the base model
    model, tokenizer = load_base_model(args.model_name, args.cache_dir)
    
    # Apply LoRA
    model = create_lora_config(model)
    
    # Load the dataset
    gsm8k_dataset = load_gsm8k_dataset("train", args.cache_dir)
    
    # Prepare training data
    examples = prepare_training_dataset(gsm8k_dataset, args.max_samples, args.seed)
    messages = get_training_messages(examples)
    
    # Extract reference answers
    reference_answers = []
    for example in examples:
        answer = extract_model_answer(example["answer"])
        if answer is not None:
            reference_answers.append(answer)
        else:
            # Fallback to a dummy value if extraction fails
            reference_answers.append(0.0)
    
    # Create and run the trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        dataset=messages,
        reference_answers=reference_answers
    )
    
    # Enable logits for Unsloth
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    trainer.train()

if __name__ == "__main__":
    main() 