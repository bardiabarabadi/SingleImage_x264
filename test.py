#!/usr/bin/env python
# title           :test.py
# description     :to test the model
# author          :Deepak Birla
# date            :2018/10/30
# usage           :python test.py --options
# python_version  :3.5.4

from keras.models import Model
from skimage import io
import numpy as np
from keras.models import load_model
from keras.layers import Input
import argparse
from keras.backend.tensorflow_backend import set_session

import config
from Utils_model import VGG_LOSS
import tensorflow as tf
from Utils import denormalize, denormalize_

from DataGen import TestDataGenerator
import cv2
_config = tf.ConfigProto()
_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=_config)
set_session(sess)

# Better to use downscale factor as 4
downscale_factor = 2
# Remember to change image shape if you are having different size of images
image_shape = tuple(config.window_size)


def test_model_for_lr_images(NR_dir, LR_dir, stride, model, output_dir):
    test_data = TestDataGenerator(input_dir=NR_dir, output_dir=LR_dir, batch_size=1, stride=stride)
    # predictions = model.predict_generator(test_data)
    samples = len(test_data)
    for ex in range(samples):
        ex_name = test_data.list_IDs[ex]
        test_image = test_data.__getitem__(ex)[0]
        prediction = model.predict(test_image, batch_size=1, verbose=1)[0]
        #print (prediction)
        generated_image = denormalize(prediction)
        #print (generated_image)
        cv2.imwrite(output_dir + ex_name, generated_image)
        print('Generating predictions, ' + str(100 * ex / samples) + '% ' + str(generated_image.shape) + '...\r')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-inr', '--input_noisy', action='store', dest='input_NR',
                        default='./QP42/',
                        help='Path for input images N resolution')

    parser.add_argument('-ilr', '--input_noNoise', action='store', dest='input_LR',
                        default=None,
                        help='Path for input images L resolution')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./SR/',
                        help='Path for Output images')

    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/model_best_weights.h5',
                        help='Path for model')

    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=120,
                        help='Number of Images', type=int)

    parser.add_argument('-t', '--stride', action='store', dest='stride', default=3,
                        help='Stride between frames', type=int)

    values = parser.parse_args()

    loss = VGG_LOSS(image_shape)
    model = load_model(values.model_dir, custom_objects={'vgg_loss': loss.vgg_loss})

    model.summary()

    model.layers.pop(0)

    model.summary()
    newInput = Input(batch_shape=(None, 540, 960, 3))  # let us say this new InputLayer
    newOutputs = model(newInput)
    newModel = Model(newInput, newOutputs)

    newModel.summary()

    if values.input_LR is None:
        values.input_LR = values.input_NR

    test_model_for_lr_images(NR_dir=values.input_NR, LR_dir=values.input_LR, model=newModel,
                             output_dir=values.output_dir, stride=3)
