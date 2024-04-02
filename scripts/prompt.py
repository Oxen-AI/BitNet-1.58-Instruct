
import argparse
from bitnet.models.bitnet import BitNetLLM
from bitnet.prompts.assistant_prompt import AssistantPrompt

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Prompt our BitNet model in a loop.')
    parser.add_argument('-m', '--model', required=True, type=str, help='dataset to run model on')
    args = parser.parse_args()

    # create the model
    model = BitNetLLM(args.model)
    
    # prompt the model in a loop
    while True:
        prompt = input("🐂 > ")
        prompt = AssistantPrompt({"prompt": prompt}).render()
        data = {"prompt": prompt, "answers": []}
        output = model.predict(data)
        print(f"Time: {output['time']}s")
        print()
    

if __name__ == '__main__':
    main()