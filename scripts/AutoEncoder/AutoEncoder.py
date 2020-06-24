import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
import datetime

from CustomTrainingCallback import *


class AutoEncoder():

    def __init__(self, code_size=320, img_dim=[16, 90, 1]):

        self.img_dim = img_dim

        self.data_file_dir = '/home/satchel/ctrl_ws/src/ctrl_planner/training_data/TrainingDataFVs.txt'

        FlattenedTrainingData, self.TrainingData = self.load_data(self.data_file_dir)
        self.TrainingData = np.expand_dims(self.TrainingData, -1)

        self.RI_train, self.RI_val = train_test_split(self.TrainingData, test_size=0.1, random_state=42)

        self.code_size = code_size
        self.encoder, self.decoder = self.build_autoencoder()

        inp = layers.Input(self.img_dim)
        code = self.encoder(inp)

        code = tf.expand_dims(code, -1)
        code = tf.expand_dims(code, 1)

        reconstruction = self.decoder(code)

        self.autoencoder = models.Model(inp, reconstruction)

        self.autoencoder.compile(optimizer='adam', loss='mse') # Mean squared error loss
        # decoder.compile(optimizer='adam', loss='mse', metrics=['accuracy']) # Mean squared error loss

        print(self.autoencoder.summary())

    def load_data(self, file_dir):

        FlattenedTrainingData = np.loadtxt(file_dir).astype(float)
        TrainingData = []
        for i in range (len(FlattenedTrainingData)):
            TrainingData.append(np.reshape(FlattenedTrainingData[i], (16, 90)))

        return FlattenedTrainingData, TrainingData

    def show_image(self, range_image):
        if (len(range_image.shape) >= 2):
            plt.imshow(range_image, cmap='gray')

    def build_autoencoder(self):
        # The encoder
        encoder = models.Sequential()
        input_layer_encoder = layers.InputLayer(self.img_dim)
        encoder.add(input_layer_encoder) # Just a layer that acts as a placeholder for the input

        conv_1 = layers.Conv2D(
            filters=8,
            kernel_size=(5, 5),
            strides=[1,1],
            padding='same',
            activation=layers.LeakyReLU(alpha=0.3), ## Note that leaky ReLu just means that instead of being 0 when the input is < 0,
            # there is a small negative slope of alpha.
        )

        encoder.add(conv_1)

        conv_2 = layers.Conv2D(
            filters=16, 
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            activation=layers.LeakyReLU(alpha=0.3),
        )

        encoder.add(conv_2)

        conv_3 = layers.Conv2D(
            filters=24,
            kernel_size=[3,5],
            strides=[2,2],
            padding='same',
            activation=layers.LeakyReLU(alpha=0.3),
        )

        encoder.add(conv_3)

        # conv_4 = layers.Conv2D(
        #     filters=32, 
        #     kernel_size=[3,3],
        #     strides=[2,2],
        #     padding='same',
        #     activation=layers.LeakyReLU(alpha=0.3)
        # )

        # encoder.add(conv_4)

        # conv_5 = layers.Conv2D(
        #     filters=8,
        #     kernel_size=[3,3],
        #     strides=[1,1],
        #     padding='same',
        #     activation=layers.LeakyReLU(alpha=0.3),
        #     # activation='sigmoid',
        # )

        # encoder.add(conv_5)

        bn = layers.BatchNormalization(
            axis=-1,
        )

        encoder.add(bn)

        encoder.add(layers.Flatten())
        encoder.add(layers.Dense(self.code_size))

        ############################
        ############################
        ############################
        ############################
        ############################
        ############################

        # The decoder
        decoder = models.Sequential()
        input_layer_decoder = layers.InputLayer((1, self.code_size, 1))
        decoder.add(input_layer_decoder)


        upconv_0 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=[1,1],
            strides=[1,1],
            padding='same',
            activation='relu',
            # input_shape=(1, self.code_size, 1)
        )

        decoder.add(upconv_0)

        upconv_1 = layers.Conv2DTranspose(
            filters=24,
            kernel_size=[2,3],
            strides=[2,2],
            padding='same',
            activation='relu',
        )

        decoder.add(upconv_1)

        upconv_2 = layers.Conv2DTranspose(
            filters=16,
            kernel_size=[3,3],
            strides=[2,2],
            padding='same',
            activation=layers.LeakyReLU(alpha=0.3)
            # activation=tf.nn.relu,
        )

        decoder.add(upconv_2)

        upconv_3 = layers.Conv2DTranspose(
            filters=8,
            kernel_size=[3,5],
            strides=[1,1],
            padding='same',
            activation=layers.LeakyReLU(alpha=0.3)
            # activation=tf.nn.relu,
        )

        decoder.add(upconv_3)

        upconv_4 = layers.Conv2DTranspose(
            filters=1,
            kernel_size=[3,3],
            strides=[1,1],
            padding='same',
            # activation=tf.nn.relu,
            activation='sigmoid',
        )

        decoder.add(upconv_4)


        decoder.add(layers.Flatten())
        decoder.add(layers.Dense(np.prod(self.img_dim))) # np.prod(img_shape) is the same as 16*90 = 1440, it's more generic than saying 1440
        decoder.add(layers.Reshape(self.img_dim))

        return encoder, decoder

    def find_best_code_size(self, Epochs=100, code_size_start=10, code_size_finish=300):
        val_loss_list = []

        for code in range (code_size_start, code_size_finish):

            self.code_size = code
            history = self.train(Epochs)
            curr_val_loss = history.history['val_loss'].pop()
            val_loss_list.append(curr_val_loss)
        

            plt.plot(code, curr_val_loss, 'ro')
            plt.title('Auto Encoder Validation Loss for Varying Code Sizes (150 Epcohs)')
            plt.xlabel('Code Size')
            plt.ylabel('Validation Loss')
            plt.xlim([code_size_start, code_size_finish])
            plt.ylim([0, 0.2])
            plt.draw()
            plt.pause(0.5)


        best_code_size = np.argmax(val_loss_list, axis=0) + code_size_start
        self.code_size = best_code_size

        return best_code_size

    def train(self, Epochs):

        print("\n\nNum GPUs Available: ")
        print(len(tf.config.experimental.list_physical_devices('GPU')))
        print('\n')

        history = self.autoencoder.fit(x=self.RI_train, y=self.RI_train, epochs=Epochs, 
            validation_data=[self.RI_val, self.RI_val], callbacks=[CustomTrainingCallback()])

        self.save_model()

        plt.plot(history.history['loss'], label='loss')
        # plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Auto Encoder Loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.ylim([0, 0.15])
        # plt.legend(['train'], loc='upper right')
        plt.savefig('AutoEncoderTrainingLoss.png')

        return history

    def visualize(self, range_img):
        """Draws original, encoded and decoded images"""
        # range_img[None] will have shape of (1, 16, 90, 1) which is the same as the model input
        code = self.encoder.predict(range_img[None])[0]
        code_input = np.expand_dims(code, -1)
        code_input = np.expand_dims(code_input, 0)
        reco = self.decoder.predict(code_input[None])[0]

        plt.subplot(1,3,1)
        plt.title("Original")
        self.show_image(np.reshape(range_img, (16, 90)))

        plt.subplot(1,3,2)
        plt.title("Code")
        plt.imshow(code.reshape([code.shape[-1]//2,-1]))

        plt.subplot(1,3,3)
        plt.title("Reconstructed")
        self.show_image(np.reshape(reco, (16, 90)))
        plt.show()

    def save_model(self, model_dir='/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder/weights/'):
        encoder_fn = model_dir + "encoder_weights"
        encoder_decoder_fn = model_dir + "encoder_DECODER_weights"

        print('\nWEIGHTS BEFORE SAVING:\n', self.encoder.get_weights()[0][0][0])
        self.encoder.save_weights(encoder_fn)
        self.autoencoder.save_weights(encoder_decoder_fn)
        return encoder_fn, encoder_decoder_fn

    def load_model(self,
            encoder_fn='/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder/weights/encoder_weights',
            encoder_decoder_fn='/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder/weights/encoder_DECODER_weights'
            ):
        print('\nWEIGHTS BEFORE RESTORE:\n', self.encoder.get_weights()[0][0][0])
        self.encoder.load_weights(encoder_fn)
        self.autoencoder.load_weights(encoder_decoder_fn)
        print('\nWEIGHTS AFTER RESTORE:\n', self.encoder.get_weights()[0][0][0])
        return



if __name__ == "__main__":

    autoencoder = AutoEncoder(code_size=320)
    # autoencoder.find_best_code_size(Epochs=100)
    history = autoencoder.train(150)

    # autoencoder.save_model()

    for i in range(500, 520):
        img = autoencoder.RI_train[i]
        autoencoder.visualize(img)
