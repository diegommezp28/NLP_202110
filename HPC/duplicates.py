import json
import pandas as pd
from fuzzywuzzy import fuzz
data = []

inputFile = '../HW4/datasets/english/english100.json'
with open(inputFile, "r") as f:
    for line in f:
        'sdasd'
        json_line = json.loads(line)
        json_line['text'] = json_line['text'].strip()
        data.append(json_line)

df = pd.json_normalize(data)
arr = list(df['text'])
to_delete = []
dict_to_keep = {}
percentage = 0
comparisons = (len(arr)**2)/2
cur_comps = 0
prevEntero = 1
for i in range(len(arr)):
    text1 = arr[i]
    dict_to_keep[i] = True
    for j in range(i+1, len(arr)):
        text2 = arr[j]
        similarity = fuzz.token_sort_ratio(text1, text2)
        if similarity > 90:
            to_delete.append(j)
        else:
            dict_to_keep[j] = True
        cur_comps += 1
        percentage = (cur_comps)*100/comparisons
        if int(percentage) != prevEntero:
            print(str(int(percentage)) + '%')
            prevEntero = int(percentage)

arr_keep = list(dict_to_keep.keys())
print('Files to filter',len(to_delete))
df_filtered = df.iloc[df.index.isin(arr_keep)]
df_filtered.to_json('output1.json', orient='records')
