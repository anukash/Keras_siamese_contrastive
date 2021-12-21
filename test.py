"""
Created by Anurag at 20-12-2021
"""
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.layers import Layer
import random

(x_train_val, y_train_val), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")


# Change the data type to a floating point format
# x_train_val = x_train_val.astype("float32")
x_test = x_test.astype("float32")


class L1_dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embed, val_embed):
        sum_square = tf.math.reduce_sum(tf.math.square(input_embed - val_embed), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def contrastive_loss(y_true, y_pred, margin=1):
    """Calculates the constrastive loss.

  Arguments:
      y_true: List of labels, each label is of type float32.
      y_pred: List of predictions of same length as of y_true,
              each label is of type float32.

  Returns:
      A tensor containing constrastive loss as floating point value.
  """

    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )


print("[INFO] loading siamese model...")
model = tf.keras.models.load_model('best_model_new.h5',
                                   custom_objects={'L1_dist': L1_dist, 'contrastive_loss': contrastive_loss})

for i in range(2):
    for j in range(3, 5):
        input_img = x_test[i] / 255.0
        validation_img = x_test[j] / 255.0

        predictions = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        print(np.round(predictions[0][0], 7))
        print(predictions)
        # visualize(pairs_test, labels_test, to_show=1, predictions=predictions, test=True)
        print('shape 0 : ', x_test[i].shape)
        print('shape 1 : ', x_test[j].shape)
        print('label 0 : ', y_test[i])
        print('label 1 : ', y_test[j])
        cv2.namedWindow('image1', cv2.WINDOW_FREERATIO)
        cv2.namedWindow('image2', cv2.WINDOW_FREERATIO)
        cv2.imshow('image1', x_test[i])
        cv2.imshow('image2', x_test[j])
        cv2.waitKey(0)

cv2.destroyAllWindows()
