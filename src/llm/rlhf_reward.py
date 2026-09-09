import json
from typing import Any, List, Optional, Set, Union

class RLHFRewardModel:
    """
    Validates model outputs against the Kristal architecture contract.
    Supports both structured JSON outputs and raw token ID sequences.
    """

    def __init__(
        self,
        valid_vocab: Optional[Set[Union[str, int]]] = None,
        vocab=None,
    ):
        self.valid_vocab = valid_vocab if valid_vocab is not None else set()
        self.vocab = vocab

    def calculate_reward(self, model_output: Any) -> float:
        """
        Calculates a reward (-1.0 to 1.0) from structured model output.

        Accepts:
        - JSON strings that contain `token_vector`
        - dicts that contain `token_vector`
        - raw token ID lists
        - whitespace-separated token ID strings
        """
        if isinstance(model_output, str):
            try:
                parsed = json.loads(model_output)
                if isinstance(parsed, dict) and "token_vector" in parsed:
                    return self.calculate_reward_from_tokens(
                        parsed["token_vector"]
                    )
            except json.JSONDecodeError:
                pass

            token_ids = self._parse_token_list_string(model_output)
            return self.calculate_reward_from_tokens(token_ids)

        if isinstance(model_output, dict):
            if "token_vector" in model_output:
                return self.calculate_reward_from_tokens(
                    model_output["token_vector"]
                )
            return -0.8

        if isinstance(model_output, list):
            return self.calculate_reward_from_tokens(model_output)

        return -1.0

    def _parse_token_list_string(self, text: str) -> List[int]:
        token_ids = []
        for token in text.strip().split():
            if token.isdigit():
                token_ids.append(int(token))
        return token_ids

    def calculate_reward_from_tokens(self, generated_token_ids: Any) -> float:
        if not isinstance(generated_token_ids, list):
            return -0.5

        if self.valid_vocab:
            fake_tokens = []
            if all(isinstance(t, int) for t in self.valid_vocab):
                fake_tokens = [
                    t for t in generated_token_ids if t not in self.valid_vocab
                ]
            elif self.vocab is not None:
                fake_tokens = []
                for t in generated_token_ids:
                    decoded = self.vocab.decode(t)
                    if decoded == "<UNK>" or decoded not in self.valid_vocab:
                        fake_tokens.append(t)

            if fake_tokens:
                return -0.9

        reward = 0.0
        if generated_token_ids:
            reward += 0.8
        if self.valid_vocab:
            reward += 0.2
        return min(max(reward, -1.0), 1.0)
