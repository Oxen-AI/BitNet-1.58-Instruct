import time

# Base class for all models
class Model:
    def __init__(self):
        self._build()

    def _build(self):
        raise NotImplementedError("Model must implement _build()")

    def _predict(self, data):
        raise NotImplementedError("Model must implement _predict()")

    # Wrapper around _predict() that adds timing etc.
    def predict(self, data):
        start_time = time.time()
        output = self._predict(data)
        end_time = time.time()
        if type(data) == dict:
            # join data with output
            for key in data:
                output[key] = data[key]
        output['time'] = end_time - start_time
        return output

