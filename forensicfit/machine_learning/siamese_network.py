from .. import HAS_PYMONGO
if HAS_PYMONGO:
    import pymongo
import tensorflow as tf
# from tensorflow.keras import layers
# import keras
# from keras.layers import  Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, Average, Subtract
# from keras.models import Model, Sequential
# from keras.regularizers import l2
# from keras import backend as K
# from keras.optimizers import SGD,Adam
# from keras.losses import binary_crossentropy
# from keras.utils import plot_model

from abc import ABCMeta, abstractmethod


class SiameseModel2(tf.keras.models.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the loss using 

    The loss is defined as:
       L(I1, I2) = max(‖f(I1) - f(I2)‖²) + margin, 0)
    """

    def __init__(self, 
                 input_shape,
                 color_channels=1,
                 normalize=True,
                 margin=0.5):
        super(SiameseModel2, self).__init__()

        self._input_shape = input_shape


        t1f_input = tf.keras.Input(name="tape1_front", shape=input_shape + (color_channels,))
        t2f_input = tf.keras.Input(name="tape2_front", shape=input_shape + (color_channels,))
        t1b_input = tf.keras.Input(name="tape1_back", shape=input_shape + (color_channels,))
        t2b_input = tf.keras.Input(name="tape2_back", shape=input_shape + (color_channels,))


        self.get_sub_net()

        t1f_encoded = self.sub_net(t1f_input)
        t2f_encoded = self.sub_net(t2f_input)
        t1b_encoded = self.sub_net(t1b_input)
        t2b_encoded = self.sub_net(t2b_input)

        distance_f = tf.keras.layers.subtract([t1f_encoded, t2f_encoded])
        distance_b = tf.keras.layers.subtract([t1b_encoded, t2b_encoded])

        average = tf.keras.layers.Average()([distance_f, distance_b])
        out = tf.keras.layers.Dense(1, activation='sigmoid')(average)
        self.siamese_network = tf.keras.models.Model(
            inputs=[t1f_input, t2f_input, t1b_input, t2b_input],
            outputs=out,
        )
        self.margin = margin


    def call(self, inputs):
        return self.siamese_network(inputs)

    @abstractmethod
    def get_sub_net(self):
        """This is an abstract method. Do not forget to create it.
        This method represents the sub network that is going to be applied to each image 
        before comparison.
        """
        pass

    
    
    # def train_step(self, data):
    #     # GradientTape is a context manager that records every operation that
    #     # you do inside. We are using it here to compute the loss so we can get
    #     # the gradients and apply them using the optimizer specified in
    #     # `compile()`.
    #     inputs, targets = data
    #     with tf.GradientTape() as tape:
    #         preds = self.siamese_network(inputs)
    #         loss = self.compiled_loss(targets, preds)

    #     # Storing the gradients of the loss function with respect to the
    #     # weights/parameters.
    #     gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

    #     # Applying the gradients on the model using the specified optimizer
    #     self.optimizer.apply_gradients(
    #         zip(gradients, self.siamese_network.trainable_weights)
    #     )
    #     # Let's update and return the training metric.
    #     self.compiled_metrics.update_state(targets, preds)
    #     return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data):
    #     inputs, targets = data
    #     preds = self.siamese_network(inputs)
    #     loss = self.compiled_loss(targets, preds)

    #     self.compiled_metrics.update_state(targets, preds)
    #     return {m.name: m.result() for m in self.metrics}

    def plot(self, filename='network_model.png'):
        tf.keras.utils.plot_model(self.siamese_network, to_file=filename)

    # def _compute_loss(self, data):
    #     # The output of the network is a array containing the distances
    #     # between the fronts and backs
    #     inputs, targets = data
    #     distance_f, distance_b = self.siamese_network(inputs)

    #     # Computing the Loss by adding both distances and
    #     # making sure we don't get a negative value.
    #     loss = (distance_f + distance_b)/2 - targets
        
    #     # loss = tf.maximum(loss + self.margin, 0.0)
    #     return loss


    
# class DistanceLayer3(layers.Layer):
#     """
#     This layer is responsible for computing the distance between the anchor
#     embedding and the positive embedding, and the anchor embedding and the
#     negative embedding.
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def call(self, anchor, positive, negative):
#         ap_distance = tf.math.reduce_sum(tf.square(anchor - positive), -1)
#         an_distance = tf.math.reduce_sum(tf.square(anchor - negative), -1)
#         return (ap_distance, an_distance)

# class DistanceLayer2(layers.Layer):
#     """
#     This layer calculates the euclidean distance of two images.
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def call(self, inp1, inp2):
#         distance = tf.reduce_sum(tf.square(inp1 - inp2), -1)
#         return distance

