
import pandas as pd
import json

class Saver:
    def __init__(
        self,
        output_file,
        output_keys=None,
        format=None,
        save_every=None
    ):
        self.output_file = output_file
        self.output_keys = output_keys
        self.format = format
        self.save_every = save_every

        if format is None:
            # take the format from the file extension
            self.format = self.output_file.split(".")[-1]

    def save(self, results):
        match self.format:
            case "csv":
                self.save_csv(results)
            case "jsonl":
                self.save_jsonl(results)
            case _:
                raise ValueError(f"Unsupported format: {self.format}")

    def save_jsonl(self, results):
        print(f"Saving {len(results)} results to {self.output_file}")
        with open(self.output_file, "w") as f:
            for result in results:
                # sort by keys for consistent output
                result = {k: result[k] for k in self.output_keys if k in result}
                f.write(json.dumps(result) + "\n")

    def save_csv(self, results):
        print(f"Saving {len(results)} results to {self.output_file}")
        df = pd.DataFrame(results)

        # Select only the output keys specified in their order
        if self.output_keys is not None:
            df = df[self.output_keys]

        df.to_csv(self.output_file, index=False)

