"""
Utility functions for OpenVINO notebooks
"""
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url, filename=None, directory=".", show_progress=True, silent=False, timeout=10):
    """
    Download a file from a URL.

    :param url: URL of the file to download
    :param filename: Name to save the file as. If not provided, use the original filename
    :param directory: Directory to save the file in
    :param show_progress: Show a progress bar
    :param silent: Do not print any output
    :param timeout: Number of seconds before timeout
    :return: path to downloaded file
    """
    from urllib.parse import urlparse, unquote

    filename = filename or Path(unquote(urlparse(url).path)).name
    chunk_size = 16384
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True) 
    filepath = directory / filename

    if filepath.exists():
        if not silent:
            print(f"'{filepath}' already exists.")
        return filepath

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        #Download with progress bar
        with open(filepath, "wb") as file:
            if show_progress and total_size > 0:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as progress_bar:
                    for chunk in response.iter_content(chunk_size):
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size):
                    file.write(chunk)

        if not silent:
            print(f"Downloaded '{filepath}'")
        return filepath
    except requests.exceptions.RequestException as e:
        if not silent:
            print(f"Failed to download from {url}: {e}")
        raise
