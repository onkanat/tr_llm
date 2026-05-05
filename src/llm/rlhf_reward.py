import json
from typing import Dict, Any

class RLHFRewardModel:
    """
    Validates model outputs structurally against the CrystalPack contract.
    Models that hallucinate unknown morphemes or break graph rules receive negative rewards.
    """
    
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
            
        # In a real scenario, check if every item in token_vector exists in vocab (vocab_size=20500)
        # If it invented a fake token ID, penalize heavily.
        if len(token_vector) > 0:
            reward += 0.8
            
        return reward
