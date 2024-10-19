import numpy as np
import webdataset as wds
import PIL.Image
from glob import glob
import json
import os
import pandas as pd
from multiprocessing import Pool

encode_format = "jpg"

def read_csv_row(args):
    index, row = args
    image = PIL.Image.open(row["image"])
    image = image.convert("RGB")
    caption = row["caption"]
    return image, caption

def read_csv_dataset(df):
    num_processes = 16  # You can adjust this based on your system and dataset size
    with Pool(num_processes) as pool:
        results = pool.map(read_csv_row, df.iterrows())
    images, captions = zip(*results)
    return list(images), list(captions)

def WebDatasetSampleWriter(dataset, dir):
    sink = wds.TarWriter(f"{dir}.tar")

    for index, (img_str, caption) in enumerate(zip(*dataset)):
        str_index = str(index).zfill(5)
        sample = {"__key__":str(str_index), encode_format: img_str}
        sample["txt"] = str(caption)
        sink.write(sample)
    sink.close()

data_path="../data/cc12m/cc12m_train_normal_sample.csv"

output_path="../data/cc12m/cc12m_train_normal_sample"
os.makedirs(output_path, exist_ok=True)


# read csv by chunks
chunksize = 10000
reader = pd.read_csv(data_path, chunksize=chunksize, sep="\t")
for i, df in enumerate(reader):
    name = str(i).zfill(5)
    print(f"chunk {name}")
    dataset = read_csv_dataset(df)
    WebDatasetSampleWriter(dataset, os.path.join(output_path, f"cc12m-train-{name}"))
