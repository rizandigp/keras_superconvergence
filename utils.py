from keras.callbacks import Callback
import keras.backend as K

class PerBatchMetrics(Callback):
    def __init__(self):
        self.logs = {}
        self.lr = []
        self.momentum = []
        self.iteration = []
        self.current_iteration = 0

    def on_batch_end(self, batch, logs):
        self.iteration.append(self.current_iteration)
        self.lr.append(K.eval(self.model.optimizer.lr))
        if hasattr(self.model.optimizer, 'momentum'):
            self.momentum.append(K.eval(self.model.optimizer.momentum))
        if not self.logs:
            for key in logs.keys():
                self.logs[key] = []
        for key, value in logs.items():
            self.logs[key].append(value)
        self.current_iteration += 1

class PerEpochMetrics(Callback):
    def __init__(self):
        self.logs = {}
        self.iteration = []

        self.current_iteration = 0

    def on_batch_end(self, batch, logs):
        self.current_iteration += 1

    def on_epoch_end(self, batch, logs):
        if not self.logs:
            for key in logs.keys():
                self.logs[key] = []
        for key, value in logs.items():
            self.logs[key].append(value)
        self.iteration.append(self.current_iteration)