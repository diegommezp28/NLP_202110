import json
import pickle as pk
import pandas as pd
from tqdm import tqdm
from functools import partial
import os

data_dir = '.\data'
subfolders = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

def pickleToJson(path):
    df = None
    with open(path, "rb") as output_file:
        df = pk.load(output_file)
    jsonFileName = path.replace(".pkl", ".json")
    df.to_json(jsonFileName, orient="records", lines=True)

for folder in subfolders:
    tagged = pd.DataFrame()
    for file in os.listdir(folder):
        input_path = os.path.join(folder, file)
        with open(input_path, 'rb') as tagged_file:
            tagged = tagged.append(pk.load(tagged_file))
    file_path = folder+'.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as original_file:
            original = pk.load(original_file)
            del original['Unnamed: 0']

        subtracted = pd.merge(original, pd.to_numeric(tagged['id']), on=['id'], how="outer", indicator=True ).query('_merge=="left_only"')
        del subtracted['_merge']

        print()
        print(file_path)
        print('original: ',original.shape)
        print('tagged: ',tagged.shape)
        print('subtracted: ',subtracted.shape)
        
        json_file_path = file_path.replace(".pkl", ".json")
        subtracted.to_json(json_file_path, orient="records", lines=True)