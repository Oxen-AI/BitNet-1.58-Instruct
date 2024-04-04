

from bitnet.training.bitnet_trainer import BitNetTrainer
from bitnet.data.sft_data_module import SFTDataModule
from bitnet.models.bitnet import BitNetLLM

import argparse
import os
import json

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train a BitNet')
    parser.add_argument('-m', '--model', required=True, type=str, help='base model to start with')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to train model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-n', '--num_samples', default=-1, type=int, help='how many examples to train on')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size for training')
    parser.add_argument('-e', '--epochs', default=2, type=int, help='Number of epochs to train for')
    parser.add_argument('-l', '--learning_rate', default=2e-4, type=float, help='Learning rate of the model')
    parser.add_argument('--max_seq_len', default=512, type=int, help='max sequence length for model')
    args = parser.parse_args()

    # mkdir output if not exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # instantiate model / tokenizer
    model = BitNetLLM(args.model)
    
    # load dataset
    dataset = SFTDataModule(
        tokenizer=model.tokenizer,
        data_path=args.dataset,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len
    )
    print(model.tokenizer.decode(dataset.dataset[0]['input_ids']))
    
    # save some of the training data for debugging
    with open(os.path.join(args.output, "debug_data.jsonl"), "w") as f:
        for i in range(5):
            data = {
                "text": model.tokenizer.decode(dataset.dataset[i]['input_ids']),
            }
            f.write(json.dumps(data) + "\n")

    # kick off the train
    trainer = BitNetTrainer(args.output, batch_size=args.batch_size, epochs=args.epochs)
    trainer.train(model, dataset)

if __name__ == '__main__':
    main()