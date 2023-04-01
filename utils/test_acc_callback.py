import tensorflow as tf
from tensorflow import keras


class StopAtTrainAcc(keras.callbacks.Callback):
    def __init__(self, threshold, patience):
        super(StopAtTrainAcc, self).__init__()
        self.threshold = threshold
        self.patience = patience
        self.best_acc = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('accuracy')
        if acc is not None:
            if acc >= self.threshold:
                print(f'\nReached {self.threshold:.2f} accuracy, stopping training')
                self.model.stop_training = True
            elif acc > self.best_acc:
                self.best_acc = acc
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f'\nAccuracy has plateaued for {self.patience} epochs, stopping training')
                    self.model.stop_training = True

class StopAtValAcc(keras.callbacks.Callback):
    def __init__(self, threshold, patience):
        super(StopAtValAcc, self).__init__()
        self.threshold = threshold
        self.patience = patience
        self.best_acc = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        if val_acc is not None:
            if val_acc >= self.threshold:
                print(f'\nReached {self.threshold:.2f} validation accuracy, stopping training')
                self.model.stop_training = True
            elif val_acc > self.best_acc:
                self.best_acc = val_acc
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f'\nValidation accuracy has plateaued for {self.patience} epochs, stopping training')
                    self.model.stop_training = True
