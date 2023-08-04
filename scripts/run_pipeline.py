import os
from glob import glob
from forensicfit.machine_learning.download import download
from forensicfit.machine_learning.download_models import download_models
from forensicfit.machine_learning.data_pipeline import data_pipeline
from forensicfit.machine_learning.trainer_tensorflow import train
from forensicfit.machine_learning.predict import predict
from forensicfit.machine_learning.evaluate import evaluate

from forensicfit.machine_learning.data.data_ingestion import DataIngestion
from forensicfit.machine_learning.data.data_generation import DataGenerator
from forensicfit.utils import ROOT

def run_pipeline():


    # This will copy the files from the external directory to the shared directory
    # obj=DataIngestion(from_scratch=False,ncores=40)
    # obj.initiate_data_ingestion()

    # This will generate fit and non-fits pair from from the pre-process data in the interim directory
    # This will save the processed data into the processed directory
    # obj=DataGenerator(from_scratch=True)
    # obj.initiate_normal_split_generation(generated_dir='match_nonmatch_ratio_0.3', match_nonmatch_ratio=0.3)
    # obj.initiate_cross_validation_generation(generated_dir='match_nonmatch_ratio_0.3', match_nonmatch_ratio=0.3)

    # # This download the trained models to ForensicFit/models
    # download_models()

    # This will train a model from scratch
    # train()

    # # This will train a model from scratch
    # evaluate()

    processed_dir=os.path.join(ROOT, "data", "processed", "normal_split", "match_nonmatch_ratio_0.3")
    processed_back_dir = os.path.join(processed_dir,'back','test','match')
    processed_front_dir = os.path.join(processed_dir,'front','test','match')

    filenames_back=glob(processed_back_dir + '/*.jpg')
    filenames_front=glob(processed_front_dir + '/*.jpg')

    image_front_path=os.path.join(processed_front_dir,'HQC_001_R-HQC_003_L.jpg')
    image_back_path=os.path.join(processed_back_dir,'HQC_002_L-HQC_004_R.jpg')
    prediction=predict(image_front_path=image_front_path,image_back_path=image_back_path)
    print(prediction)
    
    
if __name__ == '__main__':
    run_pipeline()