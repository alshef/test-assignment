import os

import pandas as pd
import numpy as np


class Dataset:
    """
    Dataset class contains names of image files, labels. Implements batch_generator, image reading from file

    """
    def __init__(self, path_to_labels_csv, image_format):
        """
        :param path_to_labels_csv: str
            path to the csv file with labels
        :param image_format: str
            type of the image (aps, a3daps, ahi)
        """
        data_frame = pd.read_csv(path_to_labels_csv, index_col=0)
        self.X_file_names = data_frame.index.values
        self.Y_labels = data_frame.values
        self.zone_list = data_frame.columns.values
        self.image_format = image_format
        self.nb_files = len(self.X_file_names)

    def get_train_batch_generator(self, dir_path, batch_size, channels):
        """
        Return a batch generator for training which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier

        :param dir_path: str
            path to the directory with certain type images
        :param batch_size: int
            size of the generated batches
        :param channels: int
            number of channels in the image
        :return: generator
            batch generator
        """
        number_of_batches = self.nb_files // batch_size
        slices = np.arange(0, channels, channels // 4)
        counter = 0

        while True:
            batch_file_names = self.X_file_names[batch_size*counter:batch_size*(counter+1)]
            batch_labels = self.Y_labels[batch_size*counter:batch_size*(counter+1)]
            image_list = []

            for file in batch_file_names:
                image = self.read_extract_slices_normalize(os.path.join(dir_path, file + '.' + self.image_format),
                                                           slices)
                image_list.append(image)

            counter += 1

            yield np.array(image_list), batch_labels

            if counter == number_of_batches:
                idx = np.arange(self.nb_files)
                np.random.shuffle(idx)
                self.X_file_names = self.X_file_names[idx]
                self.Y_labels = self.Y_labels[idx]
                counter = 0

    def get_predict_batch_generator(self, dir_path, batch_size, channels):
        """
        Return a batch generator for prediction which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier

        :param dir_path: str
            path to the directory with certain type images
        :param batch_size: int
            size of the generated batches
        :param channels: int
            number of channels in the image
        :return: generator
            batch generator
        """
        number_of_batches = np.ceil(self.nb_files / batch_size)
        counter = 0
        slices = np.arange(0, channels, channels // 4)

        while True:
            batch_file_names = self.X_file_names[batch_size * counter:batch_size * (counter + 1)]
            batch_labels = self.Y_labels[batch_size * counter:batch_size * (counter + 1)]
            image_list = []

            for file in batch_file_names:
                image = self.read_extract_slices_normalize(os.path.join(dir_path, file + '.' + self.image_format),
                                                           slices)
                image_list.append(image)

            counter += 1
            image_list = np.array(image_list)
            yield image_list, batch_labels

            if counter == number_of_batches:
                counter = 0

    def get_list_images(self, dir_path, channels):
        """
        Return array of all images with names in X_file_names

        :param dir_path: str
            path to the directory with certain type images
        :param channels: int
            number of channels in the image
        :return: numpy.ndarray
            array of images
        """
        slices = np.arange(0, channels, channels // 4)
        image_list = []
        for file in self.X_file_names:
            image = self.read_extract_slices_normalize(os.path.join(dir_path, file + '.' + self.image_format), slices)
            image_list.append(image)
        return np.array(image_list)

    def read_extract_slices_normalize(self, image_path, slices):
        """
        Read an image, extract special slices (front, left, back, right) and normalize them

        :param image_path: str
            path to the image
        :param slices: numpy.ndarray
            numbers of slices
        :return: numpy.ndarray
            image
        """
        source_image = self.read_data(image_path)
        image = source_image[:, :, slices]
        for i in range(4):
            imin = image[:, :, i].min()
            imax = image[:, :, i].max()
            if (imax - imin) != 0:
                image[:, :, i] = (image[:, :, i] - imin) / (imax - imin)
            else:
                image[:, :, i] = 0
        return image

    def read_header(self, infile):
        """
        Read image header (first 512 bytes)

        :param infile: str
            path to the file with image
        :return: dict
            header of file with image
        """
        h = dict()
        fid = open(infile, 'r+b')
        h['filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
        h['energy_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['config_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['file_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['trans_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scan_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['data_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype='S1', count=16))
        h['frequency'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['mat_velocity'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['num_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['spare00'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['band_width'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['spare01'] = np.fromfile(fid, dtype=np.int16, count=5)
        h['polarization_type'] = np.fromfile(fid, dtype=np.int16, count=4)
        h['record_header_size'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['word_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['word_precision'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['min_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['max_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['avg_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['data_scale_factor'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['data_units'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['surf_removal'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['edge_weighting'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['x_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['y_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['z_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['t_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
        h['spare02'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['x_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['scan_orientation'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scan_direction'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['data_storage_order'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scanner_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['x_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['t_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['num_x_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_y_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_z_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['num_t_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
        h['x_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
        h['depth_recon'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['adc_type'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['spare06'] = np.fromfile(fid, dtype=np.int16, count=1)
        h['scanner_radius'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['x_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['y_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['z_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['t_delay'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['range_gate_start'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['range_gate_end'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['ahis_software_version'] = np.fromfile(fid, dtype=np.float32, count=1)
        h['spare_end'] = np.fromfile(fid, dtype=np.float32, count=10)
        return h

    def read_data(self, infile):
        """
        Read any of the 4 types of image files, returns a numpy array of the image contents

        :param infile: str
            path to the file with image
        :return: numpy.ndarray
            image
        """
        extension = os.path.splitext(infile)[1]
        h = self.read_header(infile)
        nx = int(h['num_x_pts'])
        ny = int(h['num_y_pts'])
        nt = int(h['num_t_pts'])
        fid = open(infile, 'rb')
        fid.seek(512)  # skip header
        if extension == '.aps' or extension == '.a3daps':
            if h['word_type'] == 7:  # float32
                data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
            elif h['word_type'] == 4:  # uint16
                data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)
            data = data * h['data_scale_factor']  # scaling factor
            data = data.reshape(nx, ny, nt, order='F').copy()  # make N-d image
        elif extension == '.a3d':
            if h['word_type'] == 7:  # float32
                data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
            elif h['word_type'] == 4:  # uint16
                data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)
            data = data * h['data_scale_factor']  # scaling factor
            data = data.reshape(nx, nt, ny, order='F').copy()  # make N-d image
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype=np.float32, count=2 * nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0, :, :, :].copy()
            imag = data[1, :, :, :].copy()
        fid.close()
        if extension != '.ahi':
            return np.flipud(np.moveaxis(data.T, 0, -1))
        else:
            return real, imag
