#!/usr/bin/env python
# title           :Utils.py
# description     :Have helper functions to process images and plot images
# author          :Deepak Birla
# date            :2018/10/30
# usage           :imported in other files
# python_version  :3.5.4

from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from scipy.misc import imresize
import os
import sys
import config
from skimage.util.shape import view_as_windows
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.switch_backend('agg')


# import matplotlib.pyplot as plt

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale, int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)


# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images)
    return images_hr


# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real, downscale):
    images = []
    for img in range(len(images_real)):
        images.append(
            imresize(images_real[img], [images_real[img].shape[0] // downscale, images_real[img].shape[1] // downscale],
                     interp='bicubic', mode=None))
    images_lr = array(images)
    return images_lr


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def normalize_(input_data):
    return (input_data.astype(np.float32)) / 127.5


def denormalize_(input_data):
    input_data = (input_data) * 127.5
    return input_data.astype(np.uint8)


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                fpath = os.path.join(d, f)
                image = io.imread(fpath)
                if image.shape[2] > 3:
                    files.append(image[:, :, 0:3])
                else:
                    files.append(image)
                file_names.append(os.path.basename(os.path.join(d, f)))
                count = count + 1
    return files, file_names


def load_data(directory, ext):
    files, _ = load_data_from_dirs(load_path(directory), ext)
    return files


def load_training_data(directory, ext, number_of_images=1000, train_test_ratio=0.8, isTest=False,output_dir=''):
    number_of_train_images = int(number_of_images * train_test_ratio)

    hr_files, hr_names = load_data_from_dirs(load_path(os.path.join(directory)), ext)
    nr_files, nr_names = load_data_from_dirs(load_path(os.path.join(output_dir)), ext)

    sorted_indices = sorted(range(len(hr_names)), key=lambda k: ord(hr_names[k][0])*10000+int(hr_names[k][-8:-4]))

    hr_names_ = map(lambda x, y: hr_names[y], hr_names, sorted_indices)
    hr_names = list(hr_names_)

    hr_files_ = map(lambda x, y: hr_files[y], hr_files, sorted_indices)
    hr_files = list(hr_files_)

    nr_names_ = map(lambda x, y: nr_names[y], nr_names, sorted_indices)
    nr_names = list(nr_names_)

    nr_files_ = map(lambda x, y: nr_files[y], nr_files, sorted_indices)
    nr_files = list(nr_files_)

    x_hr_train = hr_files
    x_hr_test = hr_files

    # Overlapping augmentation (patching)

    x_train_hr = array(x_hr_train)
    x_test_hr = array(x_hr_test)

    x_train_hr = normalize(x_train_hr)
    x_test_hr = normalize(x_test_hr)

    window_size_test_hr = array(x_test_hr.shape)
    window_size_train_hr = array(x_train_hr.shape)

    window_size_test_hr[1:4] = config.window_size
    window_size_train_hr[1:4] = config.window_size

    x_test_hr = view_as_windows(x_test_hr, window_size_test_hr, config.step_size_test)

    x_train_hr = view_as_windows(x_train_hr, window_size_train_hr, config.step_size)

    print(np.swapaxes(x_test_hr, 0, 4).shape)

    x_test_hr = np.reshape(np.swapaxes(x_test_hr, 0, 4), np.concatenate(([-1], array(x_test_hr.shape[-3:]))))
    x_train_hr = np.reshape(np.swapaxes(x_train_hr, 0, 4), np.concatenate(([-1], array(x_train_hr.shape[-3:]))))

    # plt.figure()
    # fig, axises = plt.subplots(15, 28)
    # axises = np.array(axises)
    # for y in range(0, 15):
    #     for x in range(0, 28):
    #         print(x, y)
    #         # plt.subplot(14,28,x+y+1)
    #         axises[y, x].imshow(denormalize(x_test_hr[x + y * 28, :, :, :]))
    #         axises[y, x].set_axis_off()
    #
    # plt.show(block=True)

    return [], x_train_hr, [], x_test_hr


def load_test_data_for_model(directory, ext, number_of_images=100):
    hr_files, _ = load_data_from_dirs(load_path(os.path.join(directory, 'HR')), ext)
    nr_files, _ = load_data_from_dirs(load_path(os.path.join(directory, 'NR')), ext)

    x_test_hr = array(hr_files)
    x_test_hr = normalize(x_test_hr)

    x_test_nr = array(nr_files)
    x_test_nr = normalize(x_test_nr)

    return x_test_nr, x_test_hr


def load_test_data(directory, ext, number_of_images=100):
    files, file_names = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    x_test_lr = array(files)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr, file_names


# While training save generated image(in form LR, SR, HR)
# Save only one image as sample  
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    value = 103
    image_batch_lr = denormalize(x_test_lr[value])

    image_batch_hr = denormalize(x_test_hr[value])
    if generator is not None:
        raw_generated = generator.predict(x_test_lr[value:value + 1, :, :, :])
        gen_img = denormalize(raw_generated[0, :, :, :])  # + image_batch_lr
    else:
        gen_img = image_batch_hr

    generated_image = gen_img

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr, interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image, interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr, interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)

    plt.close('all')


# Plots and save generated images(in form LR, SR, HR) from model to test the model 
# Save output for all images given for testing  
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]

    for index in range(examples):

        image_batch_hr = denormalize(np.expand_dims(x_test_hr[index], axis=0))
        image_batch_lr = x_test_lr
        gen_img = generator.predict(np.expand_dims(image_batch_lr[index], axis=0))
        generated_image = denormalize(gen_img)
        image_batch_lr = denormalize(np.expand_dims(image_batch_lr[index], axis=0))

        plt.figure(figsize=figsize)

        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[0], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[0], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[0], interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.png' % index)

        plt.show(block=True)

        sec = input('Next test image? (y/n)')
        if sec != 'y':
            break


# Takes LR images and save respective HR images
def plot_test_generated_images(output_dir, generator, x_test_lr, file_names, figsize=(5, 5)):
    examples = x_test_lr.shape[0]

    for index in range(examples):
        image_batch_lr = x_test_lr
        gen_img = generator.predict(np.expand_dims(image_batch_lr[index], axis=0))
        generated_image = denormalize(gen_img)
        io.imsave(output_dir + file_names[index], generated_image[0])
        print('Generating predictions, ' + str(index / examples) + '% ...\r')
