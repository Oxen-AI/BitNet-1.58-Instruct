
import argparse
from bitnet.models.bitnet import BitNetLLM

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Prompt our BitNet model in a loop.')
    parser.add_argument('-m', '--model', required=True, type=str, help='dataset to run model on')
    args = parser.parse_args()

    # create the model
    model = BitNetLLM(args.model)
    
    # prompt the model in a loop
    while True:
        prompt = input("🐂 >")
        data = {"prompt": prompt, "answers": []}
        output = model.predict(data)
        print(f"Model: {output['model']}")
        print(f"Guess: {output['guess']}")
        print(f"Correct: {output['is_correct']}")
        print(f"Time: {output['time']}s")
        print()
    

if __name__ == '__main__':
    main()