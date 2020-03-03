from Network import Generator
import Utils_model
from Utils_model import VGG_LOSS
from DataGen import DataGenerator

import numpy as np
import argparse
import config
from scipy import io as sio

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

_config = tf.ConfigProto()
_config.gpu_options.allow_growth = True  # dynamically grows the memory used on the GPU
sess = tf.Session(config=_config)
set_session(sess)

image_shape = tuple(config.window_size)


# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(epochs, batch_size, LR_dir, NR_dir, model_save_dir, stride, restore):
    loss = VGG_LOSS(image_shape)

    if restore == -1:
        print('Creating new model for training...')
        generator = Generator([image_shape[0], image_shape[1], 3]).generator()
    else:
        print('loading from model ' + model_save_dir + 'model_best_weights.h5')
        generator = load_model(model_save_dir + 'model_best_weights.h5', custom_objects={'vgg_loss': loss.vgg_loss})

    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    generator.summary()

    training_data = DataGenerator(input_dir=NR_dir, output_dir=LR_dir, batch_size=batch_size,
                                  dim=image_shape[0:2], stride=stride)

    model_json = generator.to_json()
    with open(model_save_dir + "model.json", "w") as json_file:
        json_file.write(model_json)

    checkpoint = ModelCheckpoint(model_save_dir + 'model_best_weights.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min',
                                 period=1)
    hist = generator.fit_generator(generator=training_data, use_multiprocessing=False, callbacks=[checkpoint],
                                   epochs=epochs, shuffle=True)

    to_save_dic = hist.history
    to_save_dic.update(hist.params)
    sio.savemat(model_save_dir + 'history.mat', to_save_dic)
    print('History saved at ' + model_save_dir + 'history.mat')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--LR_dir', action='store', dest='LR_dir',
                        default='./RAW/', help='Path for input images')

    parser.add_argument('-o', '--NR_dir', action='store', dest='NR_dir',
                        default='./QP42/',
                        help='Path for Output images')

    parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/',
                        help='Path for model')

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=6,
                        help='Batch Size', type=int)

    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=10,
                        help='number of iterations for training', type=int)

    parser.add_argument('-s', '--restore', action='store', dest='restore', default=-1,
                        help='Restore model number, -1 for no restore', type=int)

    parser.add_argument('-t', '--stride', action='store', dest='stride', default=0,
                        help='Number of Images', type=int)

    values = parser.parse_args()
    print(values.batch_size)
    train(epochs=values.epochs, batch_size=values.batch_size, LR_dir=values.LR_dir, NR_dir=values.NR_dir,
          model_save_dir=values.model_save_dir, stride=values.stride, restore=values.restore)
