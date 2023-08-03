import os
import shutil
import gdown

from forensicfit.utils import ROOT

def download_models():

    output = f"{ROOT}{os.sep}models.zip"
    to = f"{ROOT}{os.sep}"
    url  = f'https://drive.google.com/drive/folders/1viez7q5TmwkdbcwceS4dWxeBsM8bDg0X'


    gdown.download_folder(url=url, output=to,use_cookies=False,remaining_ok=True)


def main():
    download_models()

if __name__ == '__main__':
    main()