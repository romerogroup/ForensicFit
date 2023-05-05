import os
import shutil
import gdown

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

def download_models():

    machine_learning_dir = os.path.join(PROJECT_DIR,'machine_learning')

    output = f"{machine_learning_dir}{os.sep}models.zip"
    to = f"{machine_learning_dir}{os.sep}"
    url  = f'https://drive.google.com/drive/folders/1viez7q5TmwkdbcwceS4dWxeBsM8bDg0X'


    gdown.download_folder(url=url, output=to,use_cookies=False,remaining_ok=True)


def main():
    download_models()

if __name__ == '__main__':
    main()