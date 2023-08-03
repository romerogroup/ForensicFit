# This directory contains everything needed to set up a training pipeline for determining if Tapes are a fit or non fit

Install the necessary pyhon libraries by doing the following:

pip install -r requirements.txt

Run the code in the following order:

1. python download.py
    - This step can be skipped if you have already downloaded the data into ForensicFit/data/raw and created empty folders ForensicFit/data/raw/interim and ForensicFit/data/processed

The expected files directories in data at this point should be:

data/interim
data/processed
data/raw/shared/High Quality Hand Torn 
data/raw/shared/High Quality Cut
data/raw/shared/High Quality Hand Torn Stretched 
data/raw/shared/High Quality Scissor Cut
data/raw/shared/Low Quality Hand Torn
data/raw/shared/Low Quality Hand Torn Stretched
data/raw/shared/Low Quality Scissor Cut
data/raw/shared/Medium Quality Hand Torn
data/raw/shared/Medium Quality Scissor Cut
data/raw/shared/Corrected Scans
data/raw/shared/Master Inventory
data/raw/bad_tapes.xlsx

2. python download_models.py
    - This step can be skipped if you have already downloaded the data into a
        ForensicFit/models/back_model_tensorflow and ForensicFit/models/front_model_tensorflow
3. python data_pipeline.py
    - This step only needs to be done if you are training the model from scratch
4. python tainer_tensorflow.py
   - This step only needs to be done if you are training the model from scratch
5. python predict.py
   - This step should only be used after training the model

# Trinaned models
You can download the trained models from the following link.
https://drive.google.com/drive/folders/1viez7q5TmwkdbcwceS4dWxeBsM8bDg0X?usp=sharing
