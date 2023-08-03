from forensicfit.machine_learning.download import download
from forensicfit.machine_learning.download_models import download_models
from forensicfit.machine_learning.data_pipeline import data_pipeline
from forensicfit.machine_learning.trainer_tensorflow import train
from forensicfit.machine_learning.predict import predict


def run_pipeline():
    
    download()
    download_models()
    data_pipeline()
    train()
    predict()
    
if __name__ == '__main__':
    run_pipeline()