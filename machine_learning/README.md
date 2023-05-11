# This directory contains everything needed to set up a training pipeline for determining if Tapes are a fit or non fit

Install the necessary pyhon libraries by doing the following:

pip install -r requirements.txt

Run the code in the following order:

1. python download.py
2. python download_models.py
3. python data_pipeline.py
4. python tainer_tensorflow.py
5. python predict.py