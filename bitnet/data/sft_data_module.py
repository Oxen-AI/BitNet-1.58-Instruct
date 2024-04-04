import torch
import json
import random
import transformers

from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import Dataset

from bitnet.prompts.assistant_prompt import AssistantPrompt

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, num_samples=-1, max_seq_len=-1):
        super(SFTDataset, self).__init__()
        data = []
        print(f"Reading in data from file: {data_path}")
        with open(data_path, "r") as file:
            for line in file:  
                try:
                    data.append(json.loads(line))
                    
                    if num_samples > 0 and len(data) >= num_samples:
                        break
                except Exception as e:
                    print("json processing exception", e)
                    continue

        print(f"Got {len(data)} examples, preprocess...")
        self.max_seq_len = max_seq_len
        data_dict = self.preprocess(data, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def preprocess(self, examples, tokenizer):
        """
        Preprocess the data by creating a text prompt for each example
        """
        all_input_ids = []
        print(f"Tokenizing {len(examples)} examples...")
        token_lens = []
        for ex in tqdm(examples):
            # Add a positive example
            text = AssistantPrompt(ex, should_add_answer=True).render()
            tokenized = tokenizer.encode(text)
            
            if self.max_seq_len > 0:
                tokenized = tokenized[:self.max_seq_len]
            
            all_input_ids.append(torch.LongTensor(tokenized))
            token_lens.append(len(tokenized))
            
        # calc the max and average token length
        max_len = max(token_lens)
        sum_len = sum(token_lens)
        avg_len = sum_len / len(token_lens)
        print(f"Max token length: {max_len}")
        print(f"Average token length: {avg_len} = {sum_len} / {len(token_lens)}")

        random.shuffle(all_input_ids)
        return dict(input_ids=all_input_ids, labels=all_input_ids)
    
@dataclass
class DataCollatorForSFTDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

class SFTDataModule():
    def __init__(self, tokenizer, data_path: str, num_samples=-1, max_seq_len=-1):
        self.dataset = SFTDataset(data_path=data_path, tokenizer=tokenizer, num_samples=num_samples, max_seq_len=max_seq_len)
        self.data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)