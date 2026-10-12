import pandas as pd
import numpy as np
import io
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from PIL import Image

DATASET = "Docty/Blood-Cells"

def load_file(filename = "data/train-00000-of-00001.parquet", size = 64):
    path = hf_hub_download(repo_id=DATASET, filename=filename, repo_type="dataset", local_dir=".")
    df = pd.read_parquet(path)
    df["image"] = df["image"].apply(img_to_matrix, size = size)
    
    X = np.stack(df["image"].to_list())
    y = df["label"].to_numpy()
    return X, y
    

def load_data(size):
    print("Loading data:")
    X, y = load_file(size = size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = 0.2,
        stratify = y,
        random_state = 42
    )
    
    return X_train, X_test, y_test

def img_to_matrix(cell, size):
    raw_data = cell["bytes"]
    img = Image.open(io.BytesIO(raw_data)).convert("RGB")
    img = img.resize((size, size))
    pixels = np.asarray(img)
    return pixels
    
