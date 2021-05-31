import threading
import requests
import time
import json
import pandas as pd
import time
import unidecode

spanish_path = "./datasets/es.json"
english_path = "./datasets/all_data_en.json"

file = open(english_path, "r")
data = []
for index, line in enumerate(file):
    data.append(json.loads(line))
    if index == 1000:
        break

text_raw_df = pd.json_normalize(data)

print(text_raw_df.shape)
text_raw_df["text"].head(10)


def worker(i, dfi, chunk_size):
    """thread worker function"""
    tags = []
    start_time = time.time()
    print("Running worker:", i)
    for num in range(dfi.shape[0]):
        text = dfi.iloc[num, :]["text"]
        text = unidecode.unidecode(text)
        headers = {"Content-Type": "text/plain"}
        r = requests.post("http://localhost:8080/", headers=headers, data=text)
        try:
            missinfo_class = r.json()["entities"]["MisinfoClass"][0]["class"]
            tags.append(missinfo_class)
        except:
            print(" Thread:", i)
            print("text:", text)
            print(r.json())
    index = list(range(i * chunk_size, (i + 1) * chunk_size))
    final = pd.concat(
        [pd.DataFrame(dfi), pd.Series(tags, name="gate_tags", index=index)], axis=1
    )
    final.to_json(f"./gateTagging/tagThread_{i}", orient="records", lines=True)
    end_time = time.time()
    print(f"Time worker_{i}:", end_time - start_time)


threads = []
for i in range(1):
    chunk_size = 100
    dfi = text_raw_df.iloc[i * chunk_size : (i + 1) * chunk_size, :]
    t = threading.Thread(target=worker, args=(i, dfi, chunk_size))
    threads.append(t)
    t.start()
