import os
import sys
import requests
from tqdm import tqdm

def get_data_path_or_download(dataset, data_root):
    """Finds a dataset locally and downloads if needed.

    Args:
        dataset (str): dataset name. For instance 'camouflage_n=100000_2020-Oct-19.h5py'.
            See https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated for the complete list. (please ignore .a[a-z] extensions)
        data_root (str): path where the dataset will be or is stored. If empty string, it defaults to $TMPDIR 

    Raises:
        ValueError: dataset name does not exist in local path nor in remote

    Returns:
        str: dataset final path 
    """
    url_prefix = 'https://github.com/ElementAI/synbols-resources/raw/master/datasets/generated/'
    if data_root == "":
        data_root = os.environ.get("TMPDIR", "/tmp")
    full_path = os.path.join(data_root, dataset)

    if os.path.isfile(full_path):
        print("%s found." %full_path)
        return full_path
    else:
        print("Downloading %s..." %full_path)

    r = requests.head(os.path.join(url_prefix, dataset))
    is_big = not r.ok

    if is_big:
        r = requests.head(os.path.join(url_prefix, dataset + ".aa"))
        if not r.ok:
            raise ValueError("Dataset %s" %dataset, "not found in remote.") 
        response = input("Download more than 3GB (Y/N)?: ").lower()
        while response not in ["y", "n"]:
            response = input("Download more than 3GB (Y/N)?: ").lower()
        if response == "n":
            print("Aborted")
            sys.exit(0)
        parts = []
        current_part = "a"
        while r.ok: 
            r = requests.head(os.path.join(url_prefix, dataset + ".a%s" %current_part))
            parts.append(".a" + current_part)
            current_part = chr(ord(current_part) + 1)
    else:
        parts = [""]

    if not os.path.isfile(full_path):
        with open(full_path, 'wb') as file:
            for i, part in enumerate(parts):
                print("Downloading part %d/%d" %(i + 1, len(parts)))
                url = os.path.join(url_prefix, "%s%s" %(dataset, part))
                
                # Streaming, so we can iterate over the response.
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kilobyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong downloading %s" %url)
    return full_path