
from bitnet.models.model import Model
from transformers import LlamaTokenizer
from transformers import TextStreamer

from .stopping_criteria import StoppingTokenCriteria
from .causal_lm import BitnetForCausalLM

class BitNetLLM(Model):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__()
        
    def _build(self):
        print(f"Loading model {self.model_name}...")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = BitnetForCausalLM.from_pretrained(self.model_name).to("cuda")

    # Function to run the model on a single example
    def _predict(self, data):
        if not 'prompt' in data:
            raise ValueError("Prompt is required to run the model.")

        prompt = data["prompt"]

        # Stop token
        stopping_criteria = StoppingTokenCriteria(stop_token="🐂", tokenizer=self.tokenizer)

        # Tokenize the data
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Stream the results to the terminal so we can see it generating
        streamer = TextStreamer(self.tokenizer)

        generated_ids = self.model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=50,
            stopping_criteria=stopping_criteria
        )

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0][:-1]
        answer = answer.replace(prompt, "").strip()
        
        is_correct = False
        if 'answers' in data:
            is_correct = answer.lower() in [d.lower() for d in data["answers"]]
            if data['answers'] == []:
                if "not in context" in answer.lower():
                    is_correct = True

        return {
            "prompt": prompt,
            "guess": answer,
            "is_correct": is_correct,
            "model": self.model_name
        }
