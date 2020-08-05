
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, optimizers
from tensorflow.keras.layers import Add, Multiply
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque
import random

import time
import sys

sys.path.insert(1, '/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder')

from CustomTrainingCallback import *

tf.compat.v1.disable_eager_execution()

# EncodedTrainingData = np.loadtxt('/home/satchel/ctrl_ws/src/ctrl_planner/training_data/TrainingDataEncodedRIs.txt').astype(float)

X = np.arange(-0.3, 0.3, 0.1).tolist()
Y = np.arange(-0.2, 0.2, 0.1).tolist()
Z = np.arange(-0.1, 0.2, 0.05).tolist()

PossibleActions = []


for x in X:
    for y in Y:
        for z in Z:
            PossibleActions.append([x, y, z])


# PossibleActions.insert(0, [0, 0, 0])

class ActorCritic():

    def __init__(self, sess):

        self.PossibleActions = PossibleActions
        self.StateSize = 326
        self.sess = sess        
        self.learning_rate = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = .94
        self.gamma = 0.9 # .90
        self.tau   = .125
        self.memory = deque(maxlen=2000)

        self.DoneTraining = False

        ##########################################################################
        #                                                                        #
        #                                                                        #
        ##########################################################################
        # Set up ACTOR netowork and it's gradients:

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32, 
            [None, 3]) 

        actor_model_weights = self.actor_model.trainable_weights

        # with tf.GradientTape() as tape:
        #     self.actor_grads = tape.gradient(self.actor_model.output, actor_model_weights)
        self.actor_grads = tf.gradients(ys=self.actor_model.output, # Take the gradient of actor_model.ouput (action) with respect to it's weights actor_model_weights.
            xs=actor_model_weights, grad_ys=-self.actor_critic_grad) # Here the -self.actor_critic_grad has the same length as actor_model.output (i.e. number of possible 
        # actions). Essentially this term will 'weight' the gradients for each output differently, which means each output's gradient will be multiplied by the negation (see
        # - sign) of its respective 'initial gradient' value in this self.actor_critic_grad term. The reason we use a minus sign here is because we want to use gradient ASCENT
        # rather than descent for the purpose of our actor crtic structure. To explain this further:
        #
        # Since we are defining the actor gradients here, and the actor network inputs a state
        # and must ouput the 'best' action to take, in order to find the 'best' action we must use the critic network, which ouputs Q values (rewards) given the state and action. 
        # Therefore, we wish for the actor net to ouput the action that will give us the highest reward value when paired with the current state and plugged in to the critic net.
        # So essentially we are working our way all the way backwards from the output Q value of the critic net to the weights of our actor net, and therefore we want to update 
        # the weights of our actor net in such a way that the output of the CRITIC net will be highest. Sounds like gradient ASCENT to me. Note that the output of the critic model 
        # can be expressed in terms of the weights of the actor model, if we were to fully write it out (actor weights -> actor output which is critic input (don't need to consider 
        # state here, it stays the same) -> crtiic weights -> critic output).

        grads = zip(self.actor_grads, actor_model_weights)
        self.optimizer = optimizers.Adam(self.learning_rate).apply_gradients(grads) # tf.train.AdamOptimizer(self.learning_rate)

        ##########################################################################
        #                                                                        #
        #                                                                        #
        ##########################################################################
        # Set up CRITIC netowork and it's gradients:

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()


        # with tf.GradientTape() as tape:
        #     self.critic_grads = tape.gradient(self.critic_model.output, self.critic_action_input)

        self.critic_grads = tf.gradients(ys=self.critic_model.output, 
            xs=self.critic_action_input) # Same as before, we take the gradient of critic_model.output (i.e. q value) w.r.t. critic_action_input (view state as constant).

        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.global_variables_initializer())
        # print('\n\n\n\n\n\n\n')
        # test_state = np.random.uniform(-100.0, 100.0, self.StateSize)
        # test_state = np.expand_dims(test_state, axis=0)
        # act = self.actor_model.predict(test_state)
        # print(act[0])
        # print(self.critic_model.predict([test_state, act]))
        # print('\n\n\n\n\n\n\n')
        ############################################

    def create_actor_model(self):

        k_initializer = tf.keras.initializers.RandomUniform(minval=-0.08, maxval=0.08)
        b_initializer = tf.keras.initializers.Zeros()

        input_dim   = self.StateSize
        state_input = layers.Input(input_dim)

        layer_0     = layers.Dense(128, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(state_input)
        layer_1     = layers.Dense(64, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(layer_0)
        layer_2     = layers.Dense(32, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(layer_1)
        layer_3     = layers.Dense(12, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(layer_2)
        layer_4     = layers.Dense(3, activation='tanh', kernel_initializer=k_initializer, bias_initializer=b_initializer)(layer_3) # layers.Dense(len(self.PossibleActions), activation='relu')(layer_3)
        # final_layer = layers.Activation('softmax')(layer_4)

        model = Model(
            inputs = state_input, outputs=layer_4
        )
        # model = models.Sequential()
        # input_layer_model = layers.InputLayer(input_dim)
        # model.add(input_layer_model) # Just a layer that acts as a placeholder for the input
        
        # conv1 = layers.Conv1D( 
        #     filters=32, kernel_size=3, strides=1, activation='relu'
        # )

        # model.add( conv1 ) 

        # conv2 = layers.Conv1D( 
        #     filters=64, kernel_size=4, strides=1, activation='relu'
        # )

        # model.add( conv2 ) 

        # # maxpool1 = layers.MaxPool1D(
        # #     pool_size=2
        # # )

        # # model.add(maxpool1)

        # conv3 = layers.Conv1D( 
        #     filters=16, kernel_size=3, strides=1, activation='relu'
        # )

        # model.add( conv3 ) 

        # conv4 = layers.Conv1D( 
        #     filters=8, kernel_size=2, strides=1, activation='sigmoid'
        # )

        # model.add( conv4 ) 
        
        # model.add(layers.Flatten())
        # model.add(layers.Dense(len(PossibleActions))) # Here the parameter is the dimensionality of the output space, i.e. possible outputs. This means
        # that since we have, say 18 possible actions for our agent (acceleration vectors), we should put 18 here. We'll then use softmax on these 18 outputs like in the neural 
        # network built in ML class. Softmax's purpose here would be to give us 18 probablites, one for each action (label):

        adam = Adam(lr=self.learning_rate)

        model.compile(optimizer=adam,
            loss='mse',
            metrics=['accuracy'])
        
        # print('\nACTOR STRUCTURE:')
        # model.summary()

        return state_input, model

    
    def create_critic_model(self):

        k_initializer = tf.keras.initializers.RandomUniform(minval=-0.08, maxval=0.08)
        b_initializer = tf.keras.initializers.Zeros()

        state_input_dim    = self.StateSize
        state_input        = layers.Input(shape=state_input_dim)

        action_input_dim   = 3
        action_input       = layers.Input(shape=action_input_dim)

        state_layer_1      = layers.Dense(24, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(state_input)
        state_layer_2      = layers.Dense(12, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(state_layer_1)
        # state_layer_2 = layers.Dense(48, activation='relu')(state_layer_1)
        # state_layer_3 = layers.Dense(24, activation='relu')(state_layer_2)


        action_layer_1     = layers.Dense(24, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(action_input)
        action_layer_2     = layers.Dense(12, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(action_layer_1)

        merged = Add()(
            [state_layer_2, action_layer_2] # Here we are simly element-wise adding these seperate layers together. Note then that the
            # layers must have the save length/size.
        )

        merged_layer_1     = layers.Dense(12, activation='relu', kernel_initializer=k_initializer, bias_initializer=b_initializer)(merged)
        final_output_layer = layers.Dense(1, kernel_initializer=k_initializer, bias_initializer=b_initializer)(merged_layer_1)

        model = Model(
            inputs=[state_input, action_input], outputs=final_output_layer
        )

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])
        # print('\nCRITIC STRUCTURE:')
        # model.summary()

        return state_input, action_input, model


    def train(self, ac_input_lists, iterations): # Here ac_input_lists will contain element of the form [old_state, action, reward, new_state]

        self.StateSize = len(ac_input_lists[0][0])
        for iteration in range (iterations):

            self.train_critic(ac_input_lists)
            self.train_actor(ac_input_lists)

        self.DoneTraining=True

        return


    def train_critic(self, ac_input_lists): # Here ac_input_lists is of the same form as before.
    
        for ac_input_list in ac_input_lists:

            old_state = ac_input_list[0]
            action = ac_input_list[1]
            reward = ac_input_list[2]
            new_state = ac_input_list[3]

            # target_action_index = np.argmax(self.actor_model.predict(new_state))

            # target_action = self.PossibleActions[target_action_index]
            start_time = time.time()
            target_action = self.actor_model.predict(new_state)
            actor_inference_time = time.time() - start_time
            # print('\n\n\n\n\n' + str(actor_inference_time) + '\n\n\n\n')
            # print('\n\n\nTarget Action: ', target_action, '\n\n\n\n\n')
            target_action = np.array(target_action)
            # target_action = np.expand_dims(target_action, 0)

            future_reward = self.critic_model.predict([new_state, target_action])#[0][0]
            # print(future_reward)

            print('Original Reward: ' + str(reward))
            reward += self.gamma * future_reward
            print('Reward with future reward: ' + str(reward))
            
            action = np.array(action)

            # print(new_state.shape)
            self.critic_model.fit([old_state, action], 
                reward, verbose=1)

        return


    def train_actor(self, ac_input_lists): # Here ac_input_lists is of the same form as before.

        for ac_input_list in ac_input_lists:

            old_state = ac_input_list[0]
            action = ac_input_list[1]
            reward = ac_input_list[2]
            new_state = ac_input_list[3]

            # predicted_action_index = np.argmax(self.actor_model.predict(old_state))
            # predicted_action = self.PossibleActions[predicted_action_index]

            predicted_action = self.actor_model.predict(old_state)

            predicted_action = np.array(predicted_action)
            # predicted_action = np.expand_dims(predicted_action, 0)

            # print(predicted_action)

            grads = self.sess.run(self.critic_grads, feed_dict={ 
                self.critic_state_input:  old_state,
                self.critic_action_input: predicted_action
            })[0]

            # grads = self.critic_grads(old_state, predicted_action)[0]
            
            # print(grads, '\n\n\n\n\n\n')
            
            self.sess.run(self.optimizer, feed_dict={
                self.actor_state_input: old_state,
                self.actor_critic_grad: grads
            })
            
            # self.optimizer.apply_gradients(zip(grads, old_state))

        return

    def save_model(self, model_dir='/home/satchel/ctrl_ws/src/ctrl_planner/scripts/ActorCritic/weights/'):
        actor_fn = model_dir + "actor_weights"
        critic_fn = model_dir + "critic_weights"

        print('\nWEIGHTS BEFORE SAVING:\n', self.actor_model.get_weights()[0][0][0])
        self.actor_model.save_weights(actor_fn)
        self.critic_model.save_weights(critic_fn)
        return actor_fn, critic_fn
    
    def load_model(self,
            actor_fn='/home/satchel/ctrl_ws/src/ctrl_planner/scripts/ActorCritic/weights/actor_weights',
            critic_fn='/home/satchel/ctrl_ws/src/ctrl_planner/scripts/ActorCritic/weights/critic_weights'
        ):
        print('\nWEIGHTS BEFORE RESTORE:\n', self.actor_model.get_weights()[0][0][0])
        self.actor_model.load_weights(actor_fn)
        self.critic_model.load_weights(critic_fn)
        print('\nWEIGHTS AFTER RESTORE:\n', self.actor_model.get_weights()[0][0][0])
        return

    def act(self, cur_state, epsilon):

        print('Epsilon: ' + str(epsilon))

        if np.random.random() < epsilon:
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-0.4, 0.4)
            z = 0 # random.uniform(-0.1, 0.1)
            # rand_selected_action = random.choice(self.PossibleActions)
            rand_selected_action = [x, y, z]
            rand_selected_action = np.expand_dims(rand_selected_action, 0)
            return rand_selected_action

        # action_index = np.argmax(self.actor_model.predict(cur_state))
        # return self.PossibleActions[action_index]
        return self.actor_model.predict(cur_state)

        
        




