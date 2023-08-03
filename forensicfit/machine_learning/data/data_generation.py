import os
import sys
import shutil
import re
import random
from typing import List
import multiprocessing
import itertools
from dataclasses import dataclass
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from sklearn.model_selection import KFold

from forensicfit.utils import ROOT

Image.MAX_IMAGE_PIXELS = 933120000

DATA_DIR = f"{ROOT}{os.sep}data"

@dataclass
class DataGenerationConfig:
    """
    DataGenerationConfig is a dataclass that holds the configuration for data generation.
    """

    raw_dir: str= f"{DATA_DIR}{os.sep}raw"
    interim_dir: str= f"{DATA_DIR}{os.sep}interim"
    processed_dir: str = f"{DATA_DIR}{os.sep}processed"

    split_images_dir: str= f"{interim_dir}{os.sep}split_images{os.sep}default"
    match_nonmatch_file: str = f"{raw_dir}{os.sep}match-nonmatch.xlsx"
    bad_tapes_file: str = f"{raw_dir}{os.sep}bad_tapes.xlsx"

    quality_to_remove = ["HQHTSA","HQHTSC","LQHTS"]

    def __init__(self,from_scratch: bool=False):
        pass
        # if from_scratch:
        #     if os.path.exists(self.processed_dir):
        #         shutil.rmtree(self.processed_dir)
        #     os.makedirs(self.processed_dir)

def get_random_permutation_without_replacement(permuation_list ):
    rand_i = random.randint(0,len(permuation_list)-1)
    random_tape_permutation = permuation_list.pop(rand_i)
    return random_tape_permutation

class DataGenerator:
    def __init__(self,from_scratch: bool=False):
        self.config = DataGenerationConfig(from_scratch=from_scratch)
    
    def initiate_normal_split_generation(self,generated_dir, match_nonmatch_ratio=0.3):
        print("Initiating data genetation")
        self.generated_dir = f"{self.config.processed_dir}{os.sep}normal_split{os.sep}{generated_dir}"
        if os.path.exists(self.generated_dir):
            shutil.rmtree(self.generated_dir)
        os.makedirs(self.generated_dir)
        
        df = pd.read_excel(self.config.match_nonmatch_file).drop(columns=['Unnamed: 0','flip_f', 'flip_b'])

        if self.config.quality_to_remove is not None:
            df = self.remove_quality_types(df=df, quality_list=self.config.quality_to_remove)

        bad_tape_df = pd.read_excel(self.config.bad_tapes_file).drop(columns=['Unnamed: 0','flip_f', 'flip_b'])
        df = self.remove_bad_tapes( df=df, bad_tape_df=bad_tape_df)

        try:
            df_train, df_test = self.generate_train_test(df, match_nonmatch_ratio=match_nonmatch_ratio )

            self.save_generated_data(df_train, split="train",save_dir=self.generated_dir)
            self.save_generated_data(df_test, split="test",save_dir=self.generated_dir)
        except Exception as e:
            raise e
        
    def initiate_cross_validation_generation(self,generated_dir, match_nonmatch_ratio=0.3):
        print("Initiating data genetation")
        self.generated_dir = f"{self.config.processed_dir}{os.sep}cross_validation{os.sep}{generated_dir}"
        if os.path.exists(self.generated_dir):
            shutil.rmtree(self.generated_dir)
        os.makedirs(self.generated_dir)
        # Randomize

        df = pd.read_excel(self.config.match_nonmatch_file).drop(columns=['Unnamed: 0','flip_f', 'flip_b'])
        if self.config.quality_to_remove is not None:
            df = self.remove_quality_types(df=df, quality_list=self.config.quality_to_remove)

        bad_tape_df = pd.read_excel(self.config.bad_tapes_file).drop(columns=['Unnamed: 0','flip_f', 'flip_b'])
        df = self.remove_bad_tapes( df=df, bad_tape_df=bad_tape_df)

        df['tape_f_1'] = df['tape_f1'] + '_' + df['side_f1']
        df['tape_f_2'] = df['tape_f2'] + '_' + df['side_f2']

        df['tape_b_1'] = df['tape_b1'] + '_' + df['side_b1']
        df['tape_b_2'] = df['tape_b2'] + '_' + df['side_b2']

        df = df[['tape_f_1','tape_f_2','tape_b_1','tape_b_2','match']]
        df['quality'] = df['tape_f_1'].apply(lambda x: x.split('_')[0])

        df_nonmatch = df[ df['match'] == 0]
        df_match = df[ df['match'] == 1]

        df_match=df_match.sample(frac=1.0,random_state=200) 

        # creating fk folds
        kf = KFold()
        splits = list(kf.split(df_match))


        for isplit,split in enumerate(splits):
            df_train = df_match.iloc[split[0]]
            df_test= df_match.iloc[split[1]].sample(frac=1.0,random_state=200) 
            
            split_dir = f"{self.generated_dir}{os.sep}{isplit}"
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir)

            df_train = self.generate_nonmatches(df_train, match_nonmatch_ratio=match_nonmatch_ratio)
            df_test = self.generate_nonmatches(df_test, match_nonmatch_ratio=match_nonmatch_ratio)

            self.save_generated_data(df_train, split="train",save_dir=split_dir)
            self.save_generated_data(df_test, split="test",save_dir=split_dir)
            
        # for isplit, split_dir in enumerate(os.listdir(cross_val_dir)):
        #     df_train, df_test = self.generate_train_test(match_nonmatch_ratio=match_nonmatch_ratio )
        #     self.save_generated_data(df_train, split="train")
        #     self.save_generated_data(df_test, split="test")
        
    def generate_train_test(self,df, match_nonmatch_ratio=0.3, test_size=0.2):
        
        # df = pd.read_excel(self.config.match_nonmatch_file).drop(columns=['Unnamed: 0','flip_f', 'flip_b'])

        df['tape_f_1'] = df['tape_f1'] + '_' + df['side_f1']
        df['tape_f_2'] = df['tape_f2'] + '_' + df['side_f2']

        df['tape_b_1'] = df['tape_b1'] + '_' + df['side_b1']
        df['tape_b_2'] = df['tape_b2'] + '_' + df['side_b2']

        df = df[['tape_f_1','tape_f_2','tape_b_1','tape_b_2','match']]
        df['quality'] = df['tape_f_1'].apply(lambda x: x.split('_')[0])

        df_nonmatch = df[ df['match'] == 0]
        df_match = df[ df['match'] == 1]


        df_train, df_test = train_test_split(df_match, stratify=df_match[ ['quality'] ], test_size=test_size,random_state=0)

        df_train = self.generate_nonmatches(df_train, match_nonmatch_ratio=match_nonmatch_ratio)
        df_test = self.generate_nonmatches(df_test, match_nonmatch_ratio=match_nonmatch_ratio)

        print('Number of Matches : {0}'.format( df_match.shape[0] ))
        print('Number of NonMatches : {0}'.format( df_nonmatch.shape[0] ))

        print('Number of Train Nonmatches : {0}'.format( df_train[ df_train['match'] == 0].shape[0] ))
        print('Number of Train Matches : {0}'.format( df_train[ df_train['match'] == 1].shape[0] ))

        print('Number of Test Nonmatches : {0}'.format( df_test[ df_test['match'] == 0].shape[0] ))
        print('Number of Test Matches : {0}'.format(df_test[ df_test['match'] == 1].shape[0] ))
        
        for i,x in enumerate([df_train, df_test]):
            if i==0:
                split = 'train'
            else:
                split = 'test'
            print('______________________________________________________________')
            print("Logging tape type ratio for {split} split".format(split=split))
            print('______________________________________________________________')
            for quality in df['quality'].unique():
                print('{0} qualities : {1}'.format( quality, x[ x['quality'] == quality].shape[0]/len(x) ))
        return df_train, df_test

    def generate_nonmatches(self, df, match_nonmatch_ratio=0.3):
        # Finding all match names and there permutations
        match_pairs = [[row['tape_f_1'], row['tape_f_2'], row['tape_b_1'], row['tape_b_2'], int(row['match']), row['quality']] for i, row in df.iterrows()]
        reversed_match_pairs = [[row['tape_f_2'], row['tape_f_1'],row['tape_b_2'], row['tape_b_1'], int(row['match']), row['quality']] for i, row in df.iterrows()]
        match_pairs.extend(reversed_match_pairs)
        
        # Finding tape names per quality type
        tape_names = {}
        tape_f_b_mapping = {}
        for quality in df['quality'].unique():
            tmp_list = df[ df['quality'] == quality]['tape_f_1'].values.tolist()
            tmp_list.extend(df[ df['quality'] == quality]['tape_f_2'].values.tolist())
            tape_names[quality] = tmp_list

            tmp_list_2 = df[ df['quality'] == quality]['tape_b_1'].values.tolist()
            tmp_list_2.extend(df[ df['quality'] == quality]['tape_b_2'].values.tolist())

            for tape_f,tape_b in zip(tmp_list, tmp_list_2):
                tape_f_b_mapping[tape_f] = tape_b

        # Finding permutations for each tape type
        tape_names_permutations={}

        for quality in tape_names.keys():

            if quality == 'HQSC':
                tmp_tape_names = tape_names[quality]
                tmp_tape_names.extend(tape_names['HQC'])
                quality = 'HQSC'
            elif quality =='HQC':
                continue
            elif quality == 'LQ':
                tmp_tape_names = tape_names[quality]
                quality = 'LQHT'
            else:
                tmp_tape_names = tape_names[quality]

            if quality in ["HQHT","HQSC","MQHT","MQSC","LQHT","LQSC"]:
                permutations =  list(itertools.permutations(tmp_tape_names, 2))
                random.shuffle(permutations)
                tape_names_permutations[quality] = permutations
            

        # # non_match_names for not including double nonmatches
        non_match_pairs = [] 
        isListFull = False
        while isListFull == False:

            rand_quality = random.choice(list(tape_names_permutations.keys()))

            # random permutation without replacement makes sure pairs are unique
            random_tape_permutation = get_random_permutation_without_replacement(tape_names_permutations[rand_quality])
            
            tape_f_1 = random_tape_permutation[0]
            tape_f_2 = random_tape_permutation[1]
            tape_b_1 = tape_f_b_mapping[tape_f_1]
            tape_b_2 = tape_f_b_mapping[tape_f_2]

            # nonmatch is 0, match is 1
            match = int(0)
            random_match_pair = [ tape_f_1, tape_f_2, tape_b_1, tape_b_2, match, rand_quality]

            # Make sure random_match_pair permutation is not one of the match pairs
            if random_match_pair not in match_pairs:
                non_match_pairs.append( random_match_pair)

            try:
                if (len(match_pairs)/2)/len(non_match_pairs) <= match_nonmatch_ratio:
                    isListFull = True
            except ZeroDivisionError:
                raise 'zero division error'


        final_list = df.values.tolist()
        final_list.extend(non_match_pairs)
  
        df = pd.DataFrame(final_list, columns=['tape_f_1','tape_f_2','tape_b_1','tape_b_2','match','quality'])
        return df

    def save_generated_data(self, df, split:str, save_dir:str,):

        split_front_dir = f'{save_dir}{os.sep}front{os.sep}{split}'
        split_back_dir = f'{save_dir}{os.sep}back{os.sep}{split}'


        split_match_front_dir = f'{split_front_dir}{os.sep}match'
        split_nonmatch_front_dir = f'{split_front_dir}{os.sep}nonmatch'
        

        split_match_back_dir = f'{split_back_dir}{os.sep}match'
        split_nonmatch_back_dir = f'{split_back_dir}{os.sep}nonmatch'

        # Creating directories
        if os.path.exists(split_nonmatch_front_dir):
            shutil.rmtree(split_nonmatch_front_dir)
        os.makedirs(split_nonmatch_front_dir)

        if os.path.exists(split_nonmatch_back_dir):
            shutil.rmtree(split_nonmatch_back_dir)
        os.makedirs(split_nonmatch_back_dir)

        if os.path.exists(split_match_front_dir):
            shutil.rmtree(split_match_front_dir)
        os.makedirs(split_match_front_dir)

        if os.path.exists(split_match_back_dir):
            shutil.rmtree(split_match_back_dir)
        os.makedirs(split_match_back_dir)

        for i,row in df.iterrows():
            image_f1_path = f'{self.config.split_images_dir}{os.sep}{row["tape_f_1"]}.jpg'
            image_f2_path = f'{self.config.split_images_dir}{os.sep}{row["tape_f_2"]}.jpg'

            image_b1_path = f'{self.config.split_images_dir}{os.sep}{row["tape_b_1"]}.jpg'
            image_b2_path = f'{self.config.split_images_dir}{os.sep}{row["tape_b_2"]}.jpg'

            
            if row['match'] == 1:
                name_f = f'{row["tape_f_1"]}-{row["tape_f_2"]}'
                img_f = self.concatenate_images(image_f1_path, image_f2_path)
                img_f.save(f'{split_match_front_dir}{os.sep}{name_f}.jpg')

                name_b = f'{row["tape_b_1"]}-{row["tape_b_2"]}'
                img_b = self.concatenate_images(image_b1_path, image_b2_path)
                img_b.save(f'{split_match_back_dir}{os.sep}{name_b}.jpg')

            elif row['match'] == 0:
                name_f = f'{row["tape_f_1"]}-{row["tape_f_2"]}'
                img_f = self.concatenate_images(image_f1_path, image_f2_path)
                img_f.save(f'{split_nonmatch_front_dir}{os.sep}{name_f}.jpg')

                name_b = f'{row["tape_b_1"]}-{row["tape_b_2"]}'
                img_b = self.concatenate_images(image_b1_path, image_b2_path)
                img_b.save(f'{split_nonmatch_back_dir}{os.sep}{name_b}.jpg')

    def concatenate_images(self, image_a_path, image_b_path):
        img_a = Image.open(image_a_path)
        img_b = Image.open(image_b_path)

        # img_a = img_a.rotate(180)
        # img_a = ImageOps.flip(img_a)
        img_a = ImageOps.mirror(img_a)

        # concatenate the images horizontally
        img = Image.new('L', (img_a.width + img_b.width, img_a.height))
        img.paste(im=img_a, box=(0, 0))
        img.paste(im=img_b, box=(img_a.width, 0))
        return img
    
    def remove_bad_tapes(self, df, bad_tape_df):
        idx_to_remove = []
        for j,row in bad_tape_df.iterrows():
            tape_f1 = row["tape_f1"]
            tape_f2 = row["tape_f2"]
            tape_b1 = row["tape_b1"]
            tape_b2 = row["tape_b2"]

            for i,df_row in df.iterrows():
                if tape_f1 == df_row["tape_f1"] and tape_f2 == df_row["tape_f2"] and tape_b1 == df_row["tape_b1"] and tape_b2 == df_row["tape_b2"]:
                    idx_to_remove.append(i)
        df = df.drop( index = idx_to_remove )
        return df

    def remove_quality_types(self, df, quality_list ):
        files_do_not_exists = []
        rows_to_remove = []
        for i, row in df.iterrows():

            tape_f1_num = row['tape_f1'].split('_')[0]
            tape_s1_num = row['tape_b1'].split('_')[0]
            tape_f2_num = row['tape_f2'].split('_')[0]
            tape_s2_num = row['tape_b2'].split('_')[0]

            if tape_f1_num in quality_list:
                rows_to_remove.append(i)
    
        df = df.drop(index = rows_to_remove)
        return df



if __name__=='__main__':

    ################################################
    # Starting data injestion
    #################################################
    obj=DataGenerator(from_scratch=False)
    obj.initiate_data_generation(generated_dir='match_nonmatch_ratio_0.3', match_nonmatch_ratio=0.3)