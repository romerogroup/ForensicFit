import os
import shutil
import gdown


def download_models(project_dir='.'):

    output = f"{project_dir}{os.sep}models.zip"
    to = f"{project_dir}{os.sep}"
    url  = f'https://drive.google.com/drive/folders/1viez7q5TmwkdbcwceS4dWxeBsM8bDg0X'


    gdown.download_folder(url=url, output=to,use_cookies=False,remaining_ok=True)


def main():
    download_models()

if __name__ == '__main__':
    main()