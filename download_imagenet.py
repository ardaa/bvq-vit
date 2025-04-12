import os
import argparse
import requests
import tarfile
from tqdm import tqdm
import subprocess

def download_file(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            pbar.update(size)

def extract_tar(tar_path, extract_path):
    """Extract a tar file with progress bar."""
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Extracting") as pbar:
           
            for member in members:
                if os.path.exists(extract_path + "/" + member.name):
                    print(f"File {extract_path + "/" + member.name} already exists, skipping extraction.")
                    continue
                tar.extract(member, extract_path)
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description='Download and setup ImageNet dataset')
    parser.add_argument('--output-dir', type=str, default='./drive/MyDrive/BT-ViT/data',
                      help='Directory to store the ImageNet dataset')
    parser.add_argument('--username', type=str, required=True,
                      help='ImageNet username')
    parser.add_argument('--access-key', type=str, required=True,
                      help='ImageNet access key')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # URLs for ImageNet dataset
    train_url = f"https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
    val_url = f"https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
    test_url = f"https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test.tar"

    # Define tar file paths
    train_tar = os.path.join(args.output_dir, "ILSVRC2012_img_train.tar")
    val_tar = os.path.join(args.output_dir, "ILSVRC2012_img_val.tar")
    test_tar = os.path.join(args.output_dir, "ILSVRC2012_img_test.tar")

    # Download training data if it doesn't exist
    if not os.path.exists(train_tar):
        print("Downloading training data...")
        download_file(train_url, train_tar)
    else:
        print("Training data tar file already exists, skipping download.")

    # Download validation data if it doesn't exist
    if not os.path.exists(val_tar):
        print("Downloading validation data...")
        download_file(val_url, val_tar)
    else:
        print("Validation data tar file already exists, skipping download.")
    
    # Download test data if it doesn't exist
    if not os.path.exists(test_tar):
        print("Downloading test data...")
        download_file(test_url, test_tar)
    else:
        print("Test data tar file already exists, skipping download.")

    # Extract training data
    print("Extracting training data...")
    train_dir = os.path.join(args.output_dir, "train")
    extract_tar(train_tar, train_dir)

    # Extract validation data
    print("Extracting validation data...")
    val_dir = os.path.join(args.output_dir, "val")
    extract_tar(val_tar, val_dir)
    
    # Extract test data
    print("Extracting test data...")
    test_dir = os.path.join(args.output_dir, "test")
    extract_tar(test_tar, test_dir)

    print("Dataset setup complete!")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")
    print(f"Test data: {test_dir}")

if __name__ == '__main__':
    main() 