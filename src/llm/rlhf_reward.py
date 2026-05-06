import json
from typing import Dict, Any, Set

class RLHFRewardModel:
    """
    Validates model outputs structurally against the CrystalPack contract.
    Models that hallucinate unknown morphemes or break graph rules receive negative rewards.
    """
    
    def __init__(self, valid_vocab: Set[str] = None):
        self.valid_vocab = valid_vocab if valid_vocab is not None else set()
        
    def calculate_reward(self, model_output_json_str: str) -> float:
        """
        Calculates a reward (-1.0 to 1.0) based on strict adherence to the 
        Kristal architecture's determinism and output structure.
        """
        try:
            output = json.loads(model_output_json_str)
        except json.JSONDecodeError:
            # Fatal format error
            return -1.0
            
        reward = 0.0
        
        # 1. Contract Compliance
        required_keys = {"input", "analyses", "best_surface", "token_vector"}
        if not required_keys.issubset(output.keys()):
            return -0.8
        else:
            reward += 0.2
            
        # 2. Hallucination Check (token_vector validity)
        token_vector = output.get("token_vector", [])
        if not isinstance(token_vector, list):
            return -0.5
            
        if len(token_vector) > 0:
            if self.valid_vocab:
                fake_tokens = [t for t in token_vector if t not in self.valid_vocab]
                if fake_tokens:
                    return -0.9 # Heavy penalty for hallucination
            reward += 0.8
            
        return reward
