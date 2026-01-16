import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. Directory to save downloaded datasets
DOWNLOAD_FOLDER = './my_kaggle_datasets'

# 2. Kaggle Authentication
os.environ['KAGGLE_USERNAME'] = "your_username" 
os.environ['KAGGLE_KEY'] = "your_key" 

# Input JSON file path
JSON_SOURCE = 'dsaeval.json'

def download_datasets():
    api = KaggleApi()
    try:
        api.authenticate()
        print(f"✅ Kaggle API Authenticated (User: {os.environ['KAGGLE_USERNAME']})")
    except Exception as e:
        print(f"❌ Authentication Failed: {e}")
        return

    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    try:
        with open(JSON_SOURCE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {JSON_SOURCE}")
        return

    print(f"🕵️  Scanning JSON file to extract unique dataset list...")
    
    unique_datasets = set()
    for item in data:
        slug = item.get('dataset')
        if slug:
            unique_datasets.add(slug)
    
    print(f"📊 Stats: {len(data)} records in JSON -> {len(unique_datasets)} unique datasets to download.\n")
    
    for i, dataset_slug in enumerate(sorted(unique_datasets)):
        dsname = dataset_slug.split('/')[-1]
        target_folder = f"{DOWNLOAD_FOLDER}/{dsname}/datasets"
        
        print(f"⬇️  [{i+1}/{len(unique_datasets)}] Processing: {dataset_slug}")
        
        if os.path.exists(target_folder) and os.listdir(target_folder):
             print(f"   ⏩ Folder exists and is not empty, skipping download: {target_folder}")
             continue

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        try:
            api.dataset_download_files(
                dataset_slug, 
                path=target_folder, 
                unzip=True,
                quiet=False
            )
            print(f"   ✅ Download successful")
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            if "403" in str(e):
                print("      -> Possible Token error or permission issue.")
            elif "404" in str(e):
                print("      -> Dataset does not exist.")

if __name__ == "__main__":
    download_datasets()