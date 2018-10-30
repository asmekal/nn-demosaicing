import argparse
import logging
import os

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input
from keras.models import load_model, Model
from train import psnr as psnr_score

argparser = argparse.ArgumentParser()
argparser.add_argument('--src_image_dir', '-src')
argparser.add_argument('--gt_image_dir', '-gt')
argparser.add_argument('--save_results_dir', '-sv')
argparser.add_argument('--log_dir', '-l', default='logs')


def get_model(log_dir):
    trained_model = load_model(
        os.path.join(log_dir, 'model.h5'),
        custom_objects={'psnr':psnr_score}
    )
    input_images = trained_model.inputs[0]
    predicted_image = trained_model.outputs[0]
    clipped_image = Lambda(lambda x: K.clip(x, 0, 255))(predicted_image)
    model = Model(inputs=input_images, outputs=clipped_image)
    return model


def get_psnr_model():
    predicted_images = Input(shape=(None, None, 3))
    gt_images = Input(shape=(None, None, 3))
    psnr = Lambda(lambda x: tf.image.psnr(x[0], x[1], 255))([gt_images, predicted_images])
    return Model(inputs=[predicted_images, gt_images], outputs=psnr)


def predict_one_image(model, image):
    return model.predict(np.expand_dims(image, 0))[0]


def calc_psnr(psnr_eval_model, img1, img2):
    return psnr_eval_model.predict([np.expand_dims(img1, 0), np.expand_dims(img2, 0)])[0]


def predict(log_dir, mosaic_images_folder, demosaic_images_folder, save_images_folder):
    model = get_model(log_dir)
    psnr_eval_model = get_psnr_model()
    os.makedirs(save_images_folder, exist_ok=True)
    for fn in os.listdir(mosaic_images_folder):
        logging.info(f"processing {fn}")
        image_mosaic = cv2.imread(os.path.join(mosaic_images_folder, fn))
        gt_image = cv2.imread(os.path.join(demosaic_images_folder, fn))
        assert image_mosaic is not None and gt_image is not None

        out_image = predict_one_image(model, image_mosaic)
        psnr = calc_psnr(psnr_eval_model, out_image, gt_image)

        logging.info(f"PSNR: {psnr}")
        out_fn = os.path.join(save_images_folder, fn)
        logging.info(f"saving to: {out_fn}")
        cv2.imwrite(out_fn, out_image)


if __name__ == '__main__':
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)
    predict(args.log_dir, args.src_image_dir, args.gt_image_dir, args.save_results_dir)
