import os
import datetime

from model import Model
from dataset import Dataset
from keras.optimizers import Adam


def get_config():
    conf = dict()
    conf['path_train_labels'] = '/home/alex/Projects/KagglePassenger/data/train_labels.csv'
    conf['path_val_labels'] = '/home/alex/Projects/KagglePassenger/data/val_labels.csv'
    conf['path_test_labels'] = '/home/alex/Projects/KagglePassenger/data/test_labels.csv'
    conf['path_raw_data'] = '/home/alex/Projects/KagglePassenger/data/raw_data/'
    conf['submit_path'] = '/home/alex/Projects/KagglePassenger/submission/'
    conf['image_shape'] = (660, 512)
    conf['batch_size'] = 4
    conf['nb_epoch'] = 15
    conf['model_name'] = 'cus4aps_batchnorm.h5'
    conf['patience'] = 3
    conf['opt'] = Adam(lr=0.001)
    conf['image_format'] = 'aps'
    conf['type_cnn'] = 0  # 0 - for set of 2D projections; 1 - multi-view; 2 - 3D(not implemented)
    if conf['image_format'] == 'aps':
        conf['n_channels'] = 16
    elif conf['image_format'] == 'a3daps':
        conf['n_channels'] = 64
    conf['train'] = True
    return conf


def main(config):
    path_to_script = os.path.dirname(os.path.abspath(__file__))
    input_shape = config['image_shape'][0], config['image_shape'][1], config['n_channels']
    image_dir_path = os.path.join(config['path_raw_data'], config['image_format'])
    if config['train']:
        train = Dataset(config['path_train_labels'], image_format=config['image_format'])
        valid = Dataset(config['path_val_labels'], image_format=config['image_format'])
        model = Model(type_cnn=config['type_cnn'],
                      opt=config['opt'],
                      batch_size=config['batch_size'],
                      input_shape=input_shape,
                      patience=config['patience'],
                      nb_epoch=config['nb_epoch'],
                      channels=config['n_channels'],
                      name=config['model_name'])
        model.train(train_dataset=train, val_dataset=valid, dir_path=image_dir_path)

        model.save_model(os.path.join(path_to_script, 'models', config['model_name']))
        test = Dataset(config['path_test_labels'], image_format=config['image_format'])
        name_submit = 'submit' + datetime.datetime.now().strftime("%d_%H_%M_%S") + '.csv'
        model.make_submission(test_dataset=test, path_submit=os.path.join(config['submit_path'], name_submit),
                              dir_path=image_dir_path)
    else:
        model_path = os.path.join(path_to_script, 'models', config['model_name'])
        test = Dataset(config['path_test_labels'], image_format=config['image_format'])
        name_submit = 'submit' + datetime.datetime.now().strftime("%d_%H_%M_%S") + '.csv'
        model = Model()
        model.make_submission(test_dataset=test, path_submit=os.path.join(config['submit_path'], name_submit),
                              dir_path=image_dir_path, path_to_model=model_path, after_training=False)


if __name__ == '__main__':
    conf = get_config()
    main(conf)
