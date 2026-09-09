import json
import unittest
from src.llm.rlhf_reward import RLHFRewardModel
from src.llm.tokenizer import Vocabulary


class TestRLHFRewardModel(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.valid_vocab = set(self.vocab.stoi.keys())
        self.reward_model = RLHFRewardModel(
            valid_vocab=self.valid_vocab,
            vocab=self.vocab,
        )

    def test_reward_from_valid_token_ids(self):
        token_ids = [self.vocab.encode("<BOS>"), self.vocab.encode("<EOS>")]
        reward = self.reward_model.calculate_reward(token_ids)
        self.assertGreaterEqual(reward, 0.8)

    def test_reward_from_invalid_token_ids(self):
        reward = self.reward_model.calculate_reward([9999])
        self.assertEqual(reward, -0.9)

    def test_reward_from_json_string(self):
        json_output = json.dumps(
            {
                "input": "test",
                "analyses": [],
                "best_surface": "kitap",
                "token_vector": [2, 3],
            }
        )
        reward = self.reward_model.calculate_reward(json_output)
        self.assertGreaterEqual(reward, 0.8)


if __name__ == '__main__':
    unittest.main()
