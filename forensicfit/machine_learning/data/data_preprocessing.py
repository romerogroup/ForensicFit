import os
import shutil
import multiprocessing
from functools import partial
import numpy as np
from PIL import Image
import imageio
import re

from forensicfit.utils import ROOT

Image.MAX_IMAGE_PIXELS = 933120000

def resize_images(file, raw_tape_type_dir="", resized_tape_type_dir=""  ):

    try:
        im = Image.open(f"{raw_tape_type_dir}{os.sep}{file}")

        dpi=im.info['dpi']
        if dpi[0] < 1200:
            print(im.info['dpi'])
            print(file)

        im = im.convert('L')
        # im = im.resize((6289,2472))
        im.save(f"{resized_tape_type_dir}{os.sep}{file}", dpi=dpi)
    except Exception as e:
        print("----------")
        print(e)
        print(file)
        print("----------")


def preprocess_images(raw_shared_dir, preprocessed_images_dir, ncores=16):


    raw_tape_type_dirs = [f"{raw_shared_dir}{os.sep}High Quality Hand Torn", 
                            f"{raw_shared_dir}{os.sep}High Quality Cut", 
                            f"{raw_shared_dir}{os.sep}High Quality Hand Torn Stretched", 
                            f"{raw_shared_dir}{os.sep}High Quality Scissor Cut",
                            f"{raw_shared_dir}{os.sep}Low Quality Hand Torn", 
                            f"{raw_shared_dir}{os.sep}Low Quality Hand Torn Stretched", 
                            f"{raw_shared_dir}{os.sep}Low Quality Scissor Cut",
                            f"{raw_shared_dir}{os.sep}Medium Quality Hand Torn", 
                            f"{raw_shared_dir}{os.sep}Medium Quality Scissor Cut" ]
    preprocessed_tape_type_dirs = [f"{preprocessed_images_dir}{os.sep}HQHT", 
                                f"{preprocessed_images_dir}{os.sep}HQC",
                                f"{preprocessed_images_dir}{os.sep}HQHTS", 
                                f"{preprocessed_images_dir}{os.sep}HQSC", 
                                f"{preprocessed_images_dir}{os.sep}LQHT", 
                                f"{preprocessed_images_dir}{os.sep}LQHTS", 
                                f"{preprocessed_images_dir}{os.sep}LQSC", 
                                f"{preprocessed_images_dir}{os.sep}MQHT", 
                                f"{preprocessed_images_dir}{os.sep}MQSC"]
                                
    for raw_tape_type_dir, preprocessed_tape_type_dir in zip(raw_tape_type_dirs, preprocessed_tape_type_dirs):
        print(f'Processing : {raw_tape_type_dir} - Saving to : {preprocessed_tape_type_dir}')

        if os.path.exists(preprocessed_tape_type_dir):
            shutil.rmtree(preprocessed_tape_type_dir)
        os.makedirs(preprocessed_tape_type_dir)

        image_files = os.listdir(raw_tape_type_dir)

        ignore_pattern = re.compile('.*tif.*')
        filtered_files = [f for f in image_files if ignore_pattern.match(f)]

        with multiprocessing.Pool(ncores) as pool:
            pool.map(partial(resize_images, raw_tape_type_dir=raw_tape_type_dir, resized_tape_type_dir=preprocessed_tape_type_dir ), filtered_files)


if __name__ == '__main__':

    dataset_dir = f"{ROOT}{os.sep}datasets{os.sep}raw"
    raw_shared_dir=f"{dataset_dir}{os.sep}shared"
    preprocessed_images_dir = f"{dataset_dir}{os.sep}preprocessed_images"
    ncores = 20

    preprocess_images(raw_shared_dir=raw_shared_dir, preprocessed_images_dir=preprocessed_images_dir, ncores=ncores)