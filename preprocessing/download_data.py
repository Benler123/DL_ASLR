

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

def download_data():
    CHUNK_SIZE = 40960
    DATA_SOURCE_MAPPING = 'asl-signs:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F46105%2F5087314%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240423%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240423T005456Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D05de445137be291fccc16e31846b960ae442369be6b9cf76c85d5982dc27fd1c5dc4f9cfd2ded022a600781a52b00cfee68e542bb41dfda85082af356e8d746d5841f6d5f759649c931aebe0935a25026cdc36fd4352c8e285ff10ebca3f422b5928426de9bccd0205651e6553eb9c19caab051ff92390653248efe87c53bd4d13047dddce5e7e904c60b047de829dd928ea3b7008ac7bf5453ec669f5808d85d12eb32f86a61b4ee98b056fcb8ff80bdae6039103aa2fef7289547b942d309dd9e1f14c346a0d8700e6f7988911a44dbc63ea1c8a3a52d12ef8835f3946e6732bab3c4e861835a15178d6886cabbbdcf45222311e8fe7a50bb41ffb832b29ba'

    KAGGLE_INPUT_PATH='kaggle/input'
    KAGGLE_WORKING_PATH='kaggle/working'
    KAGGLE_SYMLINK='kaggle'

    '''
    Uncomment these lines to download the data from Kaggle. If already downloaded, commemt these lines.
    '''

    os.makedirs(KAGGLE_SYMLINK)
    os.makedirs(KAGGLE_INPUT_PATH, 0o777)
    os.makedirs(KAGGLE_WORKING_PATH, 0o777)

    for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
        directory, download_url_encoded = data_source_mapping.split(':')
        download_url = unquote(download_url_encoded)
        filename = urlparse(download_url).path
        destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
        try:
            with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                total_length = fileres.headers['content-length']
                print(f'Downloading {directory}, {total_length} bytes compressed')
                dl = 0
                data = fileres.read(CHUNK_SIZE)
                while len(data) > 0:
                    dl += len(data)
                    tfile.write(data)
                    done = int(50 * dl / int(total_length))
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                    sys.stdout.flush()
                    data = fileres.read(CHUNK_SIZE)
                if filename.endswith('.zip'):
                    with ZipFile(tfile) as zfile:
                        zfile.extractall(destination_path)
                else:
                    with tarfile.open(tfile.name) as tarfile:
                        tarfile.extractall(destination_path)
                print(f'\nDownloaded and uncompressed: {directory}')
        except HTTPError as e:
            print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
            continue
        except OSError as e:
            print(f'Failed to load {download_url} to path {destination_path}')
            continue

    print('Data source import complete.')


if __name__=='__main__':
    download_data()    