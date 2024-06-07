import os
import tarfile
import urllib.request


def download_and_extract_voc2007(dest_path):
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    tar_path = os.path.join(dest_path, 'VOCtrainval_06-Nov-2007.tar')

    if not os.path.exists(tar_path):
        print(f'Downloading {url}...')
        urllib.request.urlretrieve(url, tar_path)
        print('Download complete.')

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=dest_path)
        print('Extraction complete.')


if __name__ == '__main__':
    dest_path = './data/voc2007'
    download_and_extract_voc2007(dest_path)
