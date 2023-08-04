# ForensicFit
ForensicFit is created to preprocess scanned images from different tears and generate a database that will be used in different machine-learning approaches. This package will prepare the data in four different approaches, which will be shown in the tutorial sections.
For more detailed documentation please visit https://romerogroup.github.io/ForensicFit/.
ForensicFit can be installed from PyPI https://pypi.org/project/forensicfit/.

## Machine Learning Model
To access the trined models please follow the following steps:

1. Create a virtual envronment
    python -m venv venv

    # Activate the environment
    # For ubunutu
    source venv/bin/activate 
    # For windows
    venv\Scripts\activate.bat

2. Clone repo
    git clone git@github.com:romerogroup/ForensicFit.git

3. Go into repo
    cd ForensicFit

3. After cloning the repo. Install the ML Requirements
    pip install -r requirements_ml.txt
    pip install .

4. Download the processed data from into ForensicFit/data/processed. 

5. Download the the pretrained models: (This must be executed from the ForensicFit directory)
```python
from forensicfit.machine_learning.download_models import download_models
download_models()
```

6. If training from scratch. Call the following function (This must be executed from the ForensicFit directory)
```python
from forensicfit.machine_learning.trainer_tensorflow import train
train()

```

7. Evaluate the dataset. (This must be executed from the ForensicFit directory)
```python
from forensicfit.machine_learning.evaluate import evaluate

evaluate()
```

8. If you want to predict on a single image do the following: (This must be executed from the ForensicFit directory)
```python
import os
from forensicfit.machine_learning.predict import predict
processed_dir=os.path.join("data", "processed", "normal_split", "match_nonmatch_ratio_0.3")
processed_back_dir = os.path.join(processed_dir,'back','test','match')
processed_front_dir = os.path.join(processed_dir,'front','test','match')
image_front_path=os.path.join(processed_front_dir,'HQC_001_R-HQC_003_L.jpg')
image_back_path=os.path.join(processed_back_dir,'HQC_002_L-HQC_004_R.jpg')
prediction=ml.predict(image_front_path=image_front_path,image_back_path=image_back_path)
print(prediction)
```