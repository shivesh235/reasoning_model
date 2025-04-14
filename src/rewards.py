import re
import numpy as np

def extract_model_answer(response):
    """
    Extract the numerical answer from the model's response
    
    Args:
        response (str): Model's response text
        
    Returns:
        float or None: The extracted answer, or None if no answer found
    """
    # Look for answer in the XML format: <answer>X</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", response)
    if answer_match:
        try:
            return float(answer_match.group(1).strip())
        except ValueError:
            pass
    
    # If XML format not found, try to extract from the last line
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1]
        # Look for patterns like "the answer is X" or "X is the answer"
        number_matches = re.findall(r"[-+]?\d*\.\d+|\d+", last_line)
        if number_matches:
            try:
                return float(number_matches[-1])
            except ValueError:
                pass
    
    return None

def format_compliance_reward(response, expected_format="<answer>"):
    """
    Reward function for compliance with the expected answer format
    
    Args:
        response (str): Model's response text
        expected_format (str): Expected format for the answer
        
    Returns:
        float: Reward score between 0 and 1
    """
    if expected_format in response:
        # Check if format is properly closed
        opening = f"<{expected_format.strip('<>')}>"
        closing = f"</{expected_format.strip('<>')}>"
        
        if opening in response and closing in response:
            return 1.0
        else:
            return 0.5
    
    return 0.0

def correctness_reward(response, reference_answer, tolerance=1e-6):
    """
    Reward function for numerical correctness of the answer
    
    Args:
        response (str): Model's response text
        reference_answer (float): The expected correct answer
        tolerance (float): Numerical tolerance for floating-point comparison
        
    Returns:
        float: Reward score between 0 and 1
    """
    model_answer = extract_model_answer(response)
    
    if model_answer is None:
        return 0.0
    
    # Check for exact match within tolerance
    if abs(model_answer - reference_answer) < tolerance:
        return 1.0
    
    # Partial credit for being close (within 5%)
    relative_error = abs((model_answer - reference_answer) / reference_answer)
    if relative_error < 0.05:
        return 0.5
    
    return 0.0

def reasoning_structure_reward(response, min_steps=3):
    """
    Reward function for structured reasoning with multiple steps
    
    Args:
        response (str): Model's response text
        min_steps (int): Minimum number of expected reasoning steps
        
    Returns:
        float: Reward score between 0 and 1
    """
    # Count the number of identifiable reasoning steps
    # We look for step indicators like numbered points, calculations, etc.
    
    # Check for numbered steps
    numbered_steps = re.findall(r"^\s*\d+\.", response, re.MULTILINE)
    
    # Check for calculation steps with equals signs
    calculation_steps = re.findall(r"=\s*[-+]?\d*\.\d+|\d+", response)
    
    # Check for step keywords
    step_keywords = ["first", "second", "third", "next", "then", "finally"]
    keyword_steps = sum(1 for keyword in step_keywords if keyword.lower() in response.lower())
    
    # Total number of steps detected
    total_steps = max(len(numbered_steps), len(calculation_steps), keyword_steps)
    
    if total_steps >= min_steps:
        return 1.0
    elif total_steps > 0:
        return float(total_steps) / min_steps
    else:
        return 0.0

def integer_formatting_reward(response):
    """
    Reward proper formatting of integers (no trailing decimals for whole numbers)
    
    Args:
        response (str): Model's response text
        
    Returns:
        float: Reward score between 0 and 1
    """
    # Check for decimal numbers that should be integers
    decimal_matches = re.findall(r"\d+\.\d+", response)
    
    for match in decimal_matches:
        value = float(match)
        if value.is_integer():
            # Found a decimal that should be an integer
            return 0.0
    
    return 1.0

def combined_reward(response, reference_answer, weights=None):
    """
    Combine multiple reward functions with weighted importance
    
    Args:
        response (str): Model's response text
        reference_answer (float): The expected correct answer
        weights (dict): Dictionary of weights for each reward function
        
    Returns:
        float: Combined reward score between 0 and 1
    """
    if weights is None:
        weights = {
            "correctness": 0.5,
            "format": 0.2,
            "structure": 0.2,
            "integer_format": 0.1
        }
    
    rewards = {
        "correctness": correctness_reward(response, reference_answer),
        "format": format_compliance_reward(response),
        "structure": reasoning_structure_reward(response),
        "integer_format": integer_formatting_reward(response)
    }
    
    total_reward = sum(reward * weights[name] for name, reward in rewards.items())
    
    return total_reward 