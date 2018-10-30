import numpy as np
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import os
import cv2
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--train_dir', '-t')
argparser.add_argument('--log_dir', '-l', default='logs')
argparser.add_argument('--epochs', '-e', default=10)


def build_net():
    input_images = Input(shape=(None, None, 3))
    x = input_images
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)
    model = Model(inputs=input_images, outputs=x)
    return model


def mosaic_bayer(image_bgr):
    """
    # // 0  1  2  3  4  5
    /////////////////////
    0 // B G B G B G B
    1 // G R G R G R G
    2 // B G B G B G B
    ...
    :param image_bgr:
    :return:
    """
    mosaiced_image = np.zeros_like(image_bgr)
    # B
    mosaiced_image[::2, ::2, 0] = image_bgr[::2, ::2, 0]
    # G
    mosaiced_image[1::2, ::2, 1] = image_bgr[1::2, ::2, 1]
    mosaiced_image[::2, 1::2, 1] = image_bgr[::2, 1::2, 1]
    # R
    mosaiced_image[1::2, 1::2, 2] = image_bgr[1::2, 1::2, 2]
    return mosaiced_image


def psnr(img_true, img_pred):
    img_pred_clipped = K.clip(img_pred, 0, 255)
    return tf.image.psnr(img_true, img_pred_clipped, 255)


def gen(images_dir):
    while True:
        for fn in os.listdir(images_dir):
            image = cv2.imread(os.path.join(images_dir, fn))
            mosaided = mosaic_bayer(image)
            yield np.expand_dims(mosaided, 0), np.expand_dims(image, 0)


def get_all_callbacks(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    last_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"))
    best_checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir, "model_best.h5"), save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir)
    return [
        last_checkpoint_callback,
        best_checkpoint_callback,
        tensorboard_callback
    ]


def train(images_dir, log_dir, epochs):
    model = build_net()
    model.compile('adam', loss='mse', metrics=[psnr])
    model.fit_generator(gen(images_dir), steps_per_epoch=1000, epochs=epochs, verbose=1,
                        callbacks=get_all_callbacks(log_dir), max_queue_size=1000, workers=2)


if __name__ == '__main__':
    args = argparser.parse_args()

    train(images_dir=args.train_dir, log_dir=args.log_dir, epochs=args.epochs)
