import os
import shutil
import gdown


from forensicfit.utils import ROOT


def download(project_dir='.'):
    data_dir = os.path.join(project_dir,'data')

    interim_dir = os.path.join(data_dir,'interim')
    processed_dir = os.path.join(data_dir,'processed')

    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)



    output = f"{data_dir}{os.sep}raw.zip"
    to = f"{data_dir}{os.sep}"
    url  = f'https://drive.google.com/drive/folders/1XpIo8l0ZMFmmSuIsEbiMC2ZXNR5d3KVt'


    gdown.download_folder(url=url, output=to,use_cookies=False,remaining_ok=True)




def main():
    download()

if __name__ == '__main__':
    main()