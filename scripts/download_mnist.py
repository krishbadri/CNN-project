#!/usr/bin/env python3
"""
Download MNIST dataset files
"""

import urllib.request
import os
import gzip
import shutil

def download_file(url, filename):
    """Download a file from URL"""
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping download")
        return
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def extract_gz(gz_filename, out_filename):
    """Extract .gz file"""
    if os.path.exists(out_filename):
        print(f"{out_filename} already exists, skipping extraction")
        return
    
    print(f"Extracting {gz_filename}...")
    with gzip.open(gz_filename, 'rb') as f_in:
        with open(out_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted to {out_filename}")
    
    # Remove .gz file
    os.remove(gz_filename)

def main():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    data_dir = "data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    files = [
        ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"),
    ]
    
    for gz_name, final_name in files:
        url = base_url + gz_name
        gz_path = os.path.join(data_dir, gz_name)
        final_path = os.path.join(data_dir, final_name)
        
        download_file(url, gz_path)
        extract_gz(gz_path, final_path)
    
    print("\nAll MNIST files downloaded and extracted!")

if __name__ == "__main__":
    main()

