import numpy as np
import keras
import cv2
from os import walk
from config import window_size, step_size
from skimage.util.shape import view_as_windows
from Utils import normalize


class DataGenerator(keras.utils.Sequence):

    def __init__(self, input_dir=None, output_dir=None, batch_size=32, dim=(28, 28),
                 stride=3):
        self.stride = stride
        self.dim = dim
        self.batch_size = batch_size
        f = []
        for (_, _, file_names) in walk(input_dir):
            f.extend(file_names)
            break

        #f = [x for x in f if int(x[-8:-4]) > self.stride]

        self.list_IDs = f
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.indexes = np.arange(len(self.list_IDs))
        print ('Len= ' + str(self.indexes))
        x0 = cv2.imread(self.input_dir + self.list_IDs[0], cv2.IMREAD_COLOR)
        x_train_hr = view_as_windows(x0, window_size, step_size)

        self.patches_per_image = x_train_hr.shape[0] * x_train_hr.shape[1]
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size * self.patches_per_image, *self.dim, 3))
        Y = np.empty((self.batch_size * self.patches_per_image, *self.dim, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ID_num = int(ID[-8:-4])
            # We are sure that ID_num is larger than self.stride
            ID_num_new_string = f'{(ID_num - self.stride):04}'
            ID_new = ID[:-8] + ID_num_new_string + ID[-4:]

            #x1 = cv2.imread(self.input_dir + ID_new, cv2.IMREAD_COLOR)
            x0 = cv2.imread(self.input_dir + ID, cv2.IMREAD_COLOR)
            #x1_patched = view_as_windows(x1, window_size, step_size)
            x0_patched = view_as_windows(x0, window_size, step_size)
            y0 = cv2.imread(self.output_dir + ID, cv2.IMREAD_COLOR)
            y0_patched = view_as_windows(y0, window_size, step_size)

            X[i * self.patches_per_image:(i + 1) * self.patches_per_image, :, :, 0:3] = normalize(
                np.reshape(x0_patched, [self.patches_per_image, 192, 192, 3]))
            #X[i * self.patches_per_image:(i + 1) * self.patches_per_image, :, :, 3:6] = normalize(
            #    np.reshape(x1_patched, [self.patches_per_image, 192, 192, 3]))
            Y[i * self.patches_per_image:(i + 1) * self.patches_per_image, :, :, 0:3] = normalize(
                np.reshape(y0_patched, [self.patches_per_image, 192, 192, 3]))

        return X, Y


class TestDataGenerator(keras.utils.Sequence):

    def __init__(self, input_dir=None, output_dir=None, batch_size=1,
                 stride=3):
        self.stride = stride
        self.batch_size = batch_size
        f = []
        for (_, _, file_names) in walk(input_dir):
            f.extend(file_names)
            break

        f = [x for x in f if int(x[-8:-4]) > self.stride]

        self.list_IDs = f
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 540, 960, 3))
        Y = np.empty((self.batch_size, 540, 960, 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ID_num = int(ID[-8:-4])
            # We are sure that ID_num is larger than self.stride
            ID_num_new_string = f'{(ID_num - self.stride):04}'
            #ID_new = ID[:-8] + ID_num_new_string + ID[-4:]

            #x1 = cv2.imread(self.input_dir + ID_new, cv2.IMREAD_COLOR)
            x0 = cv2.imread(self.input_dir + ID, cv2.IMREAD_COLOR)
            y0 = cv2.imread(self.output_dir + ID, cv2.IMREAD_COLOR)

            X[i, :, :, 0:3] = normalize(x0)
            #X[i, :, :, 3:6] = normalize(x1)
            Y[i, :, :, 0:3] = normalize(y0)

        return X, Y
