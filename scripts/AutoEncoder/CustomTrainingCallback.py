import tensorflow as tf
import matplotlib.pyplot as plt

class CustomTrainingCallback(tf.keras.callbacks.Callback):

#   def on_train_begin(self, logs={}):
#     # Initialize the lists for holding the logs, losses and accuracies
#     self.losses = []
#     self.acc = []
#     self.val_losses = []
#     self.val_acc = []
#     self.logs = []

  def on_epoch_end(self, epoch, logs={}):
    curr_loss = logs.get('loss')

    plt.plot(epoch, curr_loss, 'ro')
    
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.xlim([0, epoch])
    plt.ylim([0, 0.15])

    plt.draw()
    plt.pause(0.05)

#   def on_test_batch_begin(self, batch, logs=None):
#     return

#   def on_test_batch_end(self, batch, logs=None):
#     return 