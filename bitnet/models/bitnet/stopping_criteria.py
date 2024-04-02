
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stop_token, tokenizer):
        super().__init__()
        stop_token_ids = tokenizer(stop_token, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze().tolist()
        
        self.stop_token_ids = stop_token_ids[1:] # remote the <s> token
        self.tokenizer = tokenizer
        self.last_n_tokens = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1].item()
        # print(f"Stops {self.stop_token_ids}")

        # add last token to last_n_tokens, and limit to len(stop_token_ids) - 1
        if len(self.last_n_tokens) >= len(self.stop_token_ids):
            self.last_n_tokens.pop(0)
        self.last_n_tokens.append(last_token)
        
        # print(f"Last tokens: {self.last_n_tokens}")

        # check if they are equal
        if self.last_n_tokens == self.stop_token_ids:
            return True

        return False

class StoppingTokenCriteria(StoppingCriteriaList):
    def __init__(self, stop_token, tokenizer):
        self.stop_token = stop_token
        self.tokenizer = tokenizer
        stopping_criteria = StoppingCriteriaSub(tokenizer=self.tokenizer, stop_token=stop_token)
        super().__init__([stopping_criteria])
