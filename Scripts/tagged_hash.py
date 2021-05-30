import json
import re
import sys
import os
from tqdm import tqdm
from functools import partial

data_dir = '.\data'
subfolders = [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

def rawincount(filename):
    with open(filename, 'rb') as f:
        bufgen = iter(partial(f.raw.read, 1024*1024), b'')
        return sum(buf.count(b'\n') for buf in bufgen)

for folder in subfolders:
    dict = {}
    total_size = 0
    with open(f"{folder}.jsonl", 'a') as output:
        for file in os.listdir(folder):
            input_path = os.path.join(folder, file)
            input_size = rawincount(input_path)
            total_size += input_size
            print(input_path)
            with open(input_path, 'r') as input:
                for i, line in enumerate(tqdm(input, total=input_size)):
                    json_line = json.loads(line)
                    key = re.sub(r'[^0-9a-zA-Z]','', json_line['text'].strip())
                    if key not in dict:
                        dict[key] = True
                        json.dump(json_line, output)
                        output.write('\n')