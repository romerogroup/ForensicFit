import os
import sys
import shutil
import re
import multiprocessing
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

import forensicfit as ff
from forensicfit.utils import ROOT

Image.MAX_IMAGE_PIXELS = 933120000


DATA_DIR = f"{ROOT}{os.sep}data"
SCRATCH_DIR= os.path.dirname(ROOT)

@dataclass
class DataIngestionConfig:
    """
    DataIngestionConfig is a dataclass that holds the configuration for data ingestion.
    """

    original_data_dir: str= f"{SCRATCH_DIR}{os.sep}shared_forensics{os.sep}Tape{os.sep}Automated_Algorithm_Photos{os.sep}Scans"

    interim_dir: str= f"{DATA_DIR}{os.sep}interim"
    raw_dir: str= f"{DATA_DIR}{os.sep}raw"

    shared_dir: str= f"{raw_dir}{os.sep}shared"

    preprocess_dir: str= f"{interim_dir}{os.sep}preprocess"
    final_image_dir: str = f"{interim_dir}{os.sep}final_images"
    split_image_dir: str = f"{interim_dir}{os.sep}split_images"

    bad_tape_pairs: Tuple[str] = ('HQSC_5777A','HQSC_4393A','HQHTSA_2812')

    def __init__(self,from_scratch: bool=False):
        if from_scratch:

            if os.path.exists(self.preprocess_dir):
                shutil.rmtree(self.preprocess_dir)

            if os.path.exists(self.final_image_dir):
                shutil.rmtree(self.final_image_dir)

            if os.path.exists(self.split_image_dir):
                shutil.rmtree(self.split_image_dir)


class DataIngestion:
    def __init__(self,from_scratch: bool=False, ncores:int=20):
        self.config = DataIngestionConfig(from_scratch=from_scratch)
        self.ncores=ncores

    def initiate_data_ingestion(self,
                                split_image_dir_name='default'
        ):
        print("Initiating data ingestion")
        # try:
        if not os.path.exists(self.config.shared_dir):
            print("Downloading data")
            self.download_data_to_dir(original_scan_dir=self.config.original_data_dir, copy_dir=self.config.shared_dir)
            self.switch_mod_images(shared_dir=self.config.shared_dir)

        if not os.path.exists(self.config.preprocess_dir):
            print("Preprocessing Images")
            os.makedirs(self.config.preprocess_dir,exist_ok=True)
            self.preprocess_images(shared_dir=self.config.shared_dir, preprocess_dir=self.config.preprocess_dir)

        if not os.path.exists(self.config.final_image_dir):
            print("Creating final images directory")
            os.makedirs(self.config.final_image_dir,exist_ok=True)
            self.copy_images_to_dir(preprocess_dir=self.config.preprocess_dir, final_image_dir=self.config.final_image_dir)

        if not os.path.exists(f'{self.config.raw_dir}{os.sep}match.xlsx'):
            print("Preprocessing xlsx file and Combining match and nonmatches")
        self.copy_master_inventory_to_dir(shared_dir=self.config.shared_dir,raw_data_dir=self.config.raw_dir)
        df_match,df_nonmatch = self.preprocess_xlsx()
        self.combine_match_nonmatch(df_match, df_nonmatch)
        if not os.path.exists(f'{self.config.split_image_dir}{os.sep}{split_image_dir_name}'):
            print("Splitting Images")
            os.makedirs(self.config.split_image_dir,exist_ok=True)
            self.split_images(split_image_dir_name=split_image_dir_name)
            self.correct_rotated_images(raw_data_dir=self.config.raw_dir, tape_dir=f'{self.config.split_image_dir}{os.sep}{split_image_dir_name}')


    def download_data_to_dir(self,original_scan_dir, copy_dir):
        
        
        dirs_to_copy = [f"{original_scan_dir}{os.sep}High Quality Hand Torn{os.sep}Set 1",
                        f"{original_scan_dir}{os.sep}High Quality Hand Torn{os.sep}Set C",
                        f"{original_scan_dir}{os.sep}High Quality Hand Torn Stretched",
                        f"{original_scan_dir}{os.sep}High Quality Scissor Cut",
                        f"{original_scan_dir}{os.sep}Low Quality Hand Torn",
                        f"{original_scan_dir}{os.sep}Low Quality Hand Torn Stretched",
                        f"{original_scan_dir}{os.sep}Low Quality Scissor Cut",
                        f"{original_scan_dir}{os.sep}Medium Quality Hand Torn",
                        f"{original_scan_dir}{os.sep}Medium Quality Scissor Cut",
                        f"{original_scan_dir}{os.sep}Corrected Scans",
                        f"{original_scan_dir}{os.sep}Master Inventory"]

        dirs_to_copy_to = [f"{copy_dir}{os.sep}High Quality Hand Torn", 
                                f"{copy_dir}{os.sep}High Quality Cut", 
                                f"{copy_dir}{os.sep}High Quality Hand Torn Stretched", 
                                f"{copy_dir}{os.sep}High Quality Scissor Cut",
                                f"{copy_dir}{os.sep}Low Quality Hand Torn", 
                                f"{copy_dir}{os.sep}Low Quality Hand Torn Stretched", 
                                f"{copy_dir}{os.sep}Low Quality Scissor Cut",
                                f"{copy_dir}{os.sep}Medium Quality Hand Torn", 
                                f"{copy_dir}{os.sep}Medium Quality Scissor Cut",
                                f"{copy_dir}{os.sep}Corrected Scans", 
                                f"{copy_dir}{os.sep}Master Inventory"]
        try:
            if self.ncores==1:
                for dir_to_copy, dir_to_copy_to in zip(dirs_to_copy,dirs_to_copy_to):
                    print(f'Copying {dir_to_copy} to {dir_to_copy_to}')
                    shutil.copytree(dir_to_copy, dir_to_copy_to, ignore =shutil.ignore_patterns("Old{slash}Replaced Scans") )
            else:

                with multiprocessing.Pool(self.ncores) as pool:
                    pool.map(mp_copy, list(zip(dirs_to_copy,dirs_to_copy_to)))

        except Exception as e:
            logging.info('Failed to download data')
            raise CustomException(e,sys)

    def preprocess_images(self, shared_dir, preprocess_dir):


        shared_tape_type_dirs = [f"{shared_dir}{os.sep}High Quality Hand Torn", 
                            f"{shared_dir}{os.sep}High Quality Cut", 
                            f"{shared_dir}{os.sep}High Quality Hand Torn Stretched", 
                            f"{shared_dir}{os.sep}High Quality Scissor Cut",
                            f"{shared_dir}{os.sep}Low Quality Hand Torn", 
                            f"{shared_dir}{os.sep}Low Quality Hand Torn Stretched", 
                            f"{shared_dir}{os.sep}Low Quality Scissor Cut",
                            f"{shared_dir}{os.sep}Medium Quality Hand Torn", 
                            f"{shared_dir}{os.sep}Medium Quality Scissor Cut" ]
        preprocessed_tape_type_dirs = [f"{preprocess_dir}{os.sep}HQHT", 
                                    f"{preprocess_dir}{os.sep}HQC",
                                    f"{preprocess_dir}{os.sep}HQHTS", 
                                    f"{preprocess_dir}{os.sep}HQSC", 
                                    f"{preprocess_dir}{os.sep}LQHT", 
                                    f"{preprocess_dir}{os.sep}LQHTS", 
                                    f"{preprocess_dir}{os.sep}LQSC", 
                                    f"{preprocess_dir}{os.sep}MQHT", 
                                    f"{preprocess_dir}{os.sep}MQSC"]
        
        
        try:             
            for raw_tape_type_dir, preprocessed_tape_type_dir in zip(shared_tape_type_dirs, preprocessed_tape_type_dirs):
                print(f'Processing : {raw_tape_type_dir} - Saving to : {preprocessed_tape_type_dir}')

                if os.path.exists(preprocessed_tape_type_dir):
                    shutil.rmtree(preprocessed_tape_type_dir)
                os.makedirs(preprocessed_tape_type_dir)

                image_files = os.listdir(raw_tape_type_dir)

                ignore_pattern = re.compile('.*tif.*')
                filtered_files = [f for f in image_files if ignore_pattern.match(f)]

                with multiprocessing.Pool(self.ncores) as pool:
                    pool.map(partial(mp_preprocess, raw_tape_type_dir=raw_tape_type_dir, resized_tape_type_dir=preprocessed_tape_type_dir ), filtered_files)
        except Exception as e:
            logging.info('Failed to Preprocess IMages')
            raise CustomException(e,sys)
        
    def switch_mod_images(self,shared_dir):
        logging.info("Switching Out Modified Images")
        try:
            ####################################################################
            # Following will check the directories for modified files, if so replace the files with the modified, then delete the extra modfied files
            ####################################################################
            raw_tape_type_dirs = [f"{shared_dir}{os.sep}High Quality Hand Torn", 
                                    f"{shared_dir}{os.sep}High Quality Cut", 
                                    f"{shared_dir}{os.sep}High Quality Hand Torn Stretched", 
                                    f"{shared_dir}{os.sep}High Quality Scissor Cut",
                                    f"{shared_dir}{os.sep}Low Quality Hand Torn", 
                                    f"{shared_dir}{os.sep}Low Quality Hand Torn Stretched", 
                                    f"{shared_dir}{os.sep}Low Quality Scissor Cut",
                                    f"{shared_dir}{os.sep}Medium Quality Hand Torn", 
                                    f"{shared_dir}{os.sep}Medium Quality Scissor Cut" ]

            for raw_tape_type_dir in raw_tape_type_dirs:

                for file in os.listdir(raw_tape_type_dir):
                    is_tape_modified = 'mod' in file

                    if is_tape_modified:
                        tape_name = file.rsplit('_',1)[0]
                        im = Image.open(f"{raw_tape_type_dir}{os.sep}{file}")
                        
                
                        # Open the file the save the modified file in the raw directories
                        im.save(f"{raw_tape_type_dir}{os.sep}{tape_name}.tif")

                        if os.path.isfile(f"{raw_tape_type_dir}{os.sep}{file}"):
                            os.remove(f"{raw_tape_type_dir}{os.sep}{file}")
            logging.info("Switching Out Modified Images in Tape Type Directories Completed")
            ####################################################################
            # Following lines of code will check the modified folders, 
            # if there are modified files, they will replace the image in the raw directories, 
            # then delete the extra modfied files in the folder
            ####################################################################
            corrected_type_dirs = [f"{shared_dir}{os.sep}Corrected Scans{os.sep}Final HQHT", 
                                    f"{shared_dir}{os.sep}Corrected Scans{os.sep}Final HQC", 
                                    f"{shared_dir}{os.sep}Corrected Scans{os.sep}Final LQHT", 
                                    f"{shared_dir}{os.sep}Corrected Scans{os.sep}Final MQHT", 
                                    f"{shared_dir}{os.sep}Corrected Scans{os.sep}Final MQSC" ]

            raw_tape_type_dirs = [f"{shared_dir}{os.sep}High Quality Hand Torn", 
                                    f"{shared_dir}{os.sep}High Quality Cut", 
                                    f"{shared_dir}{os.sep}Low Quality Hand Torn", 
                                    f"{shared_dir}{os.sep}Medium Quality Hand Torn", 
                                    f"{shared_dir}{os.sep}Medium Quality Scissor Cut" ]

            # Loop through the modified folders
            for corrected_type_dir, raw_tape_type_dir in zip(corrected_type_dirs, raw_tape_type_dirs):
                for file in os.listdir(corrected_type_dir):

                    tape_name = file.rsplit('_',1)[0]
                    # Remove prexisting files from the raw directories
                    # if os.path.isfile(f"{raw_tape_type_dir}{os.sep}{file}"):
                    #     os.remove(f"{raw_tape_type_dir}{os.sep}{file}")
                    # if os.path.isfile(f"{raw_tape_type_dir}{os.sep}{tape_name}.tif"):
                    #     os.remove(f"{raw_tape_type_dir}{os.sep}{tape_name}.tif")


                    # Open the file the save the modified file in the raw directories
                    im = Image.open(f"{corrected_type_dir}{os.sep}{file}")
                    im.save(f"{raw_tape_type_dir}{os.sep}{tape_name}.tif")
            logging.info("Switching Out Modified Images in the corrected image directory")
        except Exception as e:
            logging.info('Failed to switch out modified images')
            CustomException(e,sys)

    def copy_images_to_dir(self,preprocess_dir, final_image_dir):
        resized_tape_type_dirs = [f"{preprocess_dir}{os.sep}HQHT", 
                                f"{preprocess_dir}{os.sep}HQC",
                                f"{preprocess_dir}{os.sep}HQHTS", 
                                f"{preprocess_dir}{os.sep}HQSC", 
                                f"{preprocess_dir}{os.sep}LQHT", 
                                f"{preprocess_dir}{os.sep}LQHTS", 
                                f"{preprocess_dir}{os.sep}LQSC", 
                                f"{preprocess_dir}{os.sep}MQHT", 
                                f"{preprocess_dir}{os.sep}MQSC"]


        if os.path.exists(final_image_dir):
                shutil.rmtree(final_image_dir)
        os.makedirs(final_image_dir)

        for resized_tape_type_dir in resized_tape_type_dirs:
            for file in os.listdir(resized_tape_type_dir):
                shutil.copy2(f"{resized_tape_type_dir}{os.sep}{file}", final_image_dir )

    def copy_master_inventory_to_dir(self, shared_dir, raw_data_dir):
        df_match = pd.read_excel(f"{shared_dir}{os.sep}Master Inventory{os.sep}MasterInventory_Match Files_Verified_17Aug2022.xlsx")
        df_nonmatch = pd.read_excel(f"{shared_dir}{os.sep}Master Inventory{os.sep}MasterInventory_Nonmatchfiles_Verified_18August2022.xlsx")

        df_match = df_match.drop(columns=['Sample 1 Unique ID','Sample 2 Unique ID' , 'Verified', 'Set ID', 'Unnamed: 10'])
        df_nonmatch = df_nonmatch.drop(columns=['Sample 1 Unique ID','Sample 2 Unique ID' , 'Verified ', 'Set ID', 'Verified by', 'Unnamed: 11'])

        df_match.to_excel(f"{raw_data_dir}{os.sep}match.xlsx" )
        df_nonmatch.to_excel(f"{raw_data_dir}{os.sep}nonmatch.xlsx" )
        return None
    
    def preprocess_xlsx(self):
        df_nonmatch = pd.read_excel(f"{self.config.raw_dir}{os.sep}nonmatch.xlsx").drop(columns = 'Unnamed: 0')
        df_match = pd.read_excel(f"{self.config.raw_dir}{os.sep}match.xlsx").drop(columns = 'Unnamed: 0')

        df_match, df_nonmatch = self.convert_HQSC_labels(df_match, df_nonmatch)
        df_match, df_nonmatch = self.convert_HQHTSA_HQHTSC_LQHTS_labels(df_match, df_nonmatch)


        df_match, df_nonmatch = self.remove_tape_pairs(df_match, df_nonmatch, tape_pairs=self.config.bad_tape_pairs)

        df_nonmatch.to_excel(f"{self.config.raw_dir}{os.sep}nonmatch.xlsx")
        df_match.to_excel(f"{self.config.raw_dir}{os.sep}match.xlsx")

        return  df_match, df_nonmatch 
    
    def remove_tape_pairs(self, df_match, df_nonmatch, tape_pairs):

        match_index=[]
        nonmatch_index=[]
        for tape_pair in tape_pairs:
            for i,row in df_nonmatch.iterrows():

                tmp_tape_pairs=[]
                for label in ['Tape 1 (Backing)','Tape 1 (Scrim)','Tape 2 (Backing)','Tape 2 (Scrim)']:
                    tmp_tape_pair = '_'.join(row[label].split('_')[:-1])
                    tmp_tape_pairs.append(tmp_tape_pair)
                if tape_pair in tmp_tape_pairs:
                    nonmatch_index.append(i)

            for i,row in df_match.iterrows():

                tmp_tape_pairs=[]
                for label in ['Tape 1 (Backing)','Tape 1 (Scrim)','Tape 2 (Backing)','Tape 2 (Scrim)']:
                    tmp_tape_pair = '_'.join(row[label].split('_')[:-1])
                    tmp_tape_pairs.append(tmp_tape_pair)

                if tape_pair in tmp_tape_pairs:
                    match_index.append(i)

        df_match = df_match.drop( index = match_index )
        df_nonmatch = df_nonmatch.drop( index = nonmatch_index)
        return df_match, df_nonmatch

    def convert_HQHTSA_HQHTSC_LQHTS_labels(self,df_match,df_nonmatch):

        for i, row in df_match.iterrows():
            pair_type = row['Tape 1 (Backing)'].split('_')[0]

            if pair_type in ['HQHTSA','HQHTSC','LQHTS']:
                for label in ['Tape 1 (Backing)','Tape 1 (Scrim)','Tape 2 (Backing)','Tape 2 (Scrim)']:
                    tape_num = row[label].split('_')[1]
                    tape_type = row[label].split('_')[0]
                    side = row[label].split('_')[-1]
                    
                    privious_name = '_'.join(row[label].split('_')[:-1])
                    new_name = f'{tape_type}_{tape_num}'

                    img_path = f"{self.config.final_image_dir}{os.sep}{privious_name}.tif"
                    new_img_path = f"{self.config.final_image_dir}{os.sep}{new_name}.tif"

                    df_match.at[i,label] = f"{new_name}_{side}"
                    if os.path.exists(img_path):
                        os.rename(img_path, new_img_path)     
   
        for i, row in df_nonmatch.iterrows():
            pair_type = row['Tape 1 (Backing)'].split('_')[0]

            if pair_type in ['HQHTSA','HQHTSC','LQHTS']:
                for label in ['Tape 1 (Backing)','Tape 1 (Scrim)','Tape 2 (Backing)','Tape 2 (Scrim)']:
                    tape_num = row[label].split('_')[1]
                    tape_type = row[label].split('_')[0]
                    side = row[label].split('_')[-1]
                    
                    privious_name = '_'.join(row[label].split('_')[:-1])
                    new_name = f'{tape_type}_{tape_num}'
                    
                    img_path = f"{self.config.final_image_dir}{os.sep}{privious_name}.tif"
                    new_img_path = f"{self.config.final_image_dir}{os.sep}{new_name}.tif"

                    df_nonmatch.at[i,label] = f"{new_name}_{side}"
                    if os.path.exists(img_path):
                        os.rename(img_path, new_img_path)     

        return df_match, df_nonmatch 
    
    def convert_HQSC_labels(self,df_match,df_nonmatch):
        for i, row in df_match.iterrows():
            if re.match('\d', row['Tape 1 (Backing)'].split('_')[0]) is not None:
                for label in ['Tape 1 (Backing)','Tape 1 (Scrim)','Tape 2 (Backing)','Tape 2 (Scrim)']:
                    tape_num = row[label].split('_')[0]
                    img_path = f"{self.config.final_image_dir}{os.sep}{row[label].split('_')[0]}.tif"
                    new_img_path = f"{self.config.final_image_dir}{os.sep}HQSC_{tape_num}.tif"
                    # print(img_path)
                    df_match.at[i,label] = f"HQSC_{tape_num}_{row[label].split('_')[-1]}"
                    if os.path.exists(img_path):
                        os.rename(img_path, new_img_path)     
   
        for i, row in df_nonmatch.iterrows():
            if re.match('\d', row['Tape 1 (Backing)'].split('_')[0]) is not None:

                for label in ['Tape 1 (Backing)','Tape 1 (Scrim)','Tape 2 (Backing)','Tape 2 (Scrim)']:
                    tape_num = row[label].split('_')[0]
                    img_path = f"{self.config.final_image_dir}{os.sep}{row[label].split('_')[0]}.tif"
                    new_img_path = f"{self.config.final_image_dir}{os.sep}HQSC_{tape_num}.tif"

                    df_nonmatch.at[i,label] = f"HQSC_{tape_num}_{row[label].split('_')[-1]}"
                    if os.path.exists(img_path):
                        os.rename(img_path, new_img_path)     

        return df_match, df_nonmatch

    def combine_match_nonmatch(self, df_match, df_nonmatch):
        tape_f1 = []
        side_f1 = []
        tape_f2 = []
        side_f2 = []
        flip_f = []
        tape_b1 = []
        side_b1 = []
        tape_b2 = []
        side_b2 = []
        flip_b = []
        match = []
        for i, row in df_match.iterrows():

            tape_f1.append('_'.join(row['Tape 1 (Backing)'].split('_')[:-1]))
            side_f1.append(row['Tape 1 (Backing)'].split('_')[-1])

            tape_f2.append('_'.join(row['Tape 2 (Backing)'].split('_')[:-1]))
            side_f2.append(row['Tape 2 (Backing)'].split('_')[-1])
            flip_f.append(row['Rotation?'])

            tape_b1.append('_'.join(row['Tape 1 (Scrim)'].split('_')[:-1]))
            side_b1.append(row['Tape 1 (Scrim)'].split('_')[-1])

            tape_b2.append('_'.join(row['Tape 2 (Scrim)'].split('_')[:-1]))
            side_b2.append(row['Tape 2 (Scrim)'].split('_')[-1])
            flip_b.append(row['Rotation?.1'])

            match.append(1)

        for i, row in df_nonmatch.iterrows():
            tape_f1.append('_'.join(row['Tape 1 (Backing)'].split('_')[:-1]))
            side_f1.append(row['Tape 1 (Backing)'].split('_')[-1])

            tape_f2.append('_'.join(row['Tape 2 (Backing)'].split('_')[:-1]))
            side_f2.append(row['Tape 2 (Backing)'].split('_')[-1])
            flip_f.append(row['Rotation?'])

            tape_b1.append('_'.join(row['Tape 1 (Scrim)'].split('_')[:-1]))
            side_b1.append(row['Tape 1 (Scrim)'].split('_')[-1])

            tape_b2.append('_'.join(row['Tape 2 (Scrim)'].split('_')[:-1]))
            side_b2.append(row['Tape 2 (Scrim)'].split('_')[-1])
            flip_b.append(row['Rotation?.1'])

            match.append(0)

        tmp_dict = {'tape_f1': tape_f1,
                    'side_f1':side_f1,
                    'tape_f2':tape_f2,
                    'side_f2':side_f2,
                    'flip_f':flip_f,
                    'tape_b1': tape_b1,
                    'side_b1':side_b1,
                    'tape_b2':tape_b2,
                    'side_b2':side_b2,
                    'flip_b':flip_b,
                    'match':match}

        df = pd.DataFrame(tmp_dict)
        df.to_excel(f"{self.config.raw_dir}{os.sep}match-nonmatch.xlsx")
        return None

    def split_images(self,split_image_dir_name):
        save_dir=f'{self.config.split_image_dir}{os.sep}{split_image_dir_name}'

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        image_files = os.listdir(self.config.final_image_dir)

        with multiprocessing.Pool(processes=self.ncores) as pool:
            pool.map(partial(mp_edge_split, 
                original_images_dir=self.config.final_image_dir, 
                save_dir=save_dir,
                resize_factor=2), 
                image_files)

    def correct_rotated_images(self,raw_data_dir:str, tape_dir:str):
        df = pd.read_excel(f"{raw_data_dir}{os.sep}match-nonmatch.xlsx").drop(columns=['Unnamed: 0'])

        tapes_to_flip = []
        for i, row in df.iterrows():
            if row['flip_f'] == 1 and row['flip_b'] == 1 and row['match'] == 1:
                # print(row)
                tape_f2_path = f"{tape_dir}{os.sep}{row['tape_f2']}_{row['side_f2']}.jpg"
                tape_b2_path = f"{tape_dir}{os.sep}{row['tape_b2']}_{row['side_b2']}.jpg"
                tape_tuple = (tape_f2_path,tape_b2_path)
                if tape_tuple not in tapes_to_flip:
                    tapes_to_flip.append(tape_tuple)
                # im = Image.open(f"{tape_dir}{os.sep}{file}")
            else:
                pass

        for tape_to_flip in tapes_to_flip:

            tape_name_f = tape_to_flip[0].split(os.sep)[-1]
            tape_name_b = tape_to_flip[1].split(os.sep)[-1]

            imf = Image.open(tape_to_flip[0])
            imb = Image.open(tape_to_flip[1])


            imf = imf.transpose(Image.FLIP_TOP_BOTTOM)
            imb = imb.transpose(Image.FLIP_TOP_BOTTOM)

            # imf.save(f"{dataset_dir}{os.sep}split_images{os.sep}train_flip{os.sep}{tape_name_f}")
            # imb.save(f"{dataset_dir}{os.sep}split_images{os.sep}train_flip{os.sep}{tape_name_b}")

            imf.save(f"{tape_dir}{os.sep}{tape_name_f}")
            imb.save(f"{tape_dir}{os.sep}{tape_name_b}")
            

def mp_edge_split(image_file: str , 
                original_images_dir: str, 
                save_dir: str,
                resize_factor=2):
        
        try:
            split_styles = ['R','L']

            for split_style in split_styles:

                im_file = f'{original_images_dir}{os.sep}{image_file}'
                # print(im_file)
                label = im_file.split(os.sep)[-1].split('.')[0]

                tape = ff.core.Tape.from_file(im_file)
                tape.resize(dpi=(600, 600))

                tape.split_v(side=split_style)

                analyzer = ff.core.TapeAnalyzer(tape=tape,
                                        correct_tilt = True, 
                                        mask_threshold=60,
                                        auto_crop=True, 
                                        padding='black')

                analyzer.get_bin_based(window_background=10,
                                            window_tape=400,
                                            dynamic_window=True,
                                            size=None,
                                            n_bins=1,
                                            overlap=10,
                                            border='min')[0]
 
                image = analyzer['bin_based'][0,:,:]

                im = Image.fromarray(image)
                im = im.resize((image.shape[1],2400))
                image = np.array(im)

                # Make dimensions even
                if image.shape[0] % 2 != 0:
                    image = np.pad(image, pad_width=((1,0), (0,0), (0,0)))
                if image.shape[1] % 2 != 0:
                    image = np.pad(image, pad_width=((0,0), (1,0), (0,0)))
                
                image_resolution = (image.shape[1]//(resize_factor),image.shape[0]//(resize_factor))
                im = im.resize(image_resolution)

                new_image_name= image_file.split(".")[0] + "_" + split_style
                im.save(f"{save_dir}{os.sep}{new_image_name}.jpg")

                del im 
                del image
                del tape
                del analyzer

        except Exception as e:
            raise e
        
def mp_copy(directory_pair):
    dir_to_copy, dir_to_copy_to =  directory_pair[0],directory_pair[1]
    print(f'Copying {dir_to_copy} to {dir_to_copy_to}')
    shutil.copytree(dir_to_copy, dir_to_copy_to, ignore =shutil.ignore_patterns("Old{slash}Replaced Scans") )
    return None

def mp_preprocess(file, raw_tape_type_dir="", resized_tape_type_dir=""  ):
    try:
        im = Image.open(f"{raw_tape_type_dir}{os.sep}{file}")

        dpi=im.info['dpi']
        if dpi[0] < 1200:
            print(im.info['dpi'])
            print(file)

        im = im.convert('L')
        im.save(f"{resized_tape_type_dir}{os.sep}{file}", dpi=dpi)
    except Exception as e:
        print("----------")
        print(e)
        print(file)
        print("----------")


if __name__ == '__main__':

    ################################################
    # Starting data injestion
    #################################################
    obj=DataIngestion(from_scratch=False)
    obj.initiate_data_ingestion()