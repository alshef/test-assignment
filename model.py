from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.layers import Input
from keras.layers import Flatten, Dropout, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import numpy as np
import pandas as pd


class Model:

    def __init__(self, type_cnn=0, opt=Adam(), input_shape=(660,512,16), nb_epoch=20, patience=3,
                 batch_size=8, channels=16, name='default_name'):
        """

        :param type_cnn:
        :param opt:
        :param input_shape:
        :param nb_epoch:
        :param patience:
        :param batch_size:
        :param channels:
        :param name:
        """
        self.model = None
        self.type_cnn = type_cnn
        self.opt = opt
        self.input_shape = input_shape
        self.nb_epoch = nb_epoch
        self.patience = patience
        self.batch_size = batch_size
        self.channels = channels
        self.name = name

    def train(self, train_dataset, val_dataset, dir_path):
        """
        Train function to train model on train_dataset; val_dataset - for validation of model performance

        :param train_dataset: dataset.Dataset
            train dataset
        :param val_dataset: dataset.Dataset
            validation dataset
        :param dir_path: str
            path to the directory of images
        :return:
            trained model
        """

        if self.type_cnn == 0:
            print('Configure CNN for set of 2D projections')
            self.prog_set_2d_CNN()
        elif self.type_cnn == 1:
            print('Configure Multi-View CNN')
            self.multi_view_CNN()
        elif self.type_cnn == 2:
            print('Configure 3D CNN')
            self.CNN_3D()
        else:
            raise ValueError('Error: wrong type_cnn, should be in (0,1,2)')

        self.model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])
        chkp_name = self.name + "-{epoch:02d}-{val_loss:.2f}.hdf5"

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.patience, verbose=0),
            ModelCheckpoint(filepath=chkp_name, monitor='val_loss', verbose=0, save_best_only=True)
        ]

        self.model.fit_generator(generator=train_dataset.get_train_batch_generator(dir_path, self.batch_size,
                                                                                   channels=self.channels),
                                 epochs=self.nb_epoch,
                                 steps_per_epoch=train_dataset.nb_files // self.batch_size,
                                 validation_data=val_dataset.get_predict_batch_generator(dir_path, self.batch_size,
                                                                                         channels=self.channels),
                                 validation_steps=np.ceil(val_dataset.nb_files / self.batch_size),
                                 callbacks=callbacks,
                                 verbose=2)

    def save_model(self, path):
        """

        :param path:
        :return:
        """
        print('Save model')
        self.model.save(filepath=path)

    def make_submission(self, test_dataset, path_submit, dir_path, after_training=True, path_to_model=None):
        """

        :param test_dataset:
        :param path_submit:
        :param dir_path:
        :param after_training:
        :param path_to_model:
        :return:
        """
        print('Make submission')
        if not after_training:
            self.model = load_model(path_to_model)

        predictions = self.model.predict(x=test_dataset.get_list_images(dir_path, channels=self.channels),
                                         batch_size=self.batch_size,
                                         verbose=1)
        z_list = test_dataset.zone_list
        pred_df = pd.DataFrame(index=test_dataset.X_file_names,
                               columns=z_list,
                               data=predictions)
        submit = pd.DataFrame(columns=['Id', 'Probability'])
        for index, row in pred_df.iterrows():
            for zone in z_list:
                obj_id = str(index) + '_' + zone
                submit.loc[submit.shape[0]] = [obj_id, row[zone]]

        submit.to_csv(path_submit, index=False)

    def prog_set_2d_CNN(self):
        """

        :return:
        """
        inputs = Input((self.input_shape[0], self.input_shape[1], 4))

        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        batch2 = BatchNormalization()(conv2)
        pool1 = MaxPooling2D(pool_size=(2, 2))(batch2)

        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        batch3 = BatchNormalization()(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(batch3)

        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        batch4 = BatchNormalization()(conv4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(batch4)

        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        batch5 = BatchNormalization()(conv5)
        pool4 = MaxPooling2D(pool_size=(2, 2))(batch5)

        flat1 = Flatten()(pool4)
        fc1 = Dense(17, activation='relu', kernel_initializer='he_normal')(flat1)
        out = Dense(17, activation='sigmoid')(fc1)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()

    def multi_view_CNN(self):
        pass

    def CNN_3D(self):
        """
        Not implemented
        :return: model
        """
        pass



