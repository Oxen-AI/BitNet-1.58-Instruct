# Iterates over a dataset and runs a model
from tqdm import tqdm

class Diver:
    def __init__(self, model, dataset, num_items=-1, saver=None):
        self.model = model
        self.dataset = dataset
        self.saver = saver
        self.num_items = num_items

    def run(self):
        # Setup the dataset
        self.dataset.build()

        results = []
        num_items = len(self.dataset)
        if self.num_items > 0:
            num_items = self.num_items
        for i in tqdm(range(num_items)):
            # try:
            row = self.dataset[i]
            row = self.model.predict(row)
            results.append(row)
            # except Exception as e:
            #     print(f"Error processing item {i}: {e}")
            #     continue

            if self.saver is not None and self.saver.save_every is not None:
                if i % self.saver.save_every == 0:
                    self.saver.save(results)

        # Save at the end
        if self.saver is not None:
            self.saver.save(results)

        return results