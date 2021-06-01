import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import json
import os
import time


def jsonToPd(path, n=3):
    file = open(path, "r")
    data = []
    for index, line in enumerate(file):
        if len(data) == n:
            break
        if index > -1:
            data.append(json.loads(line))

    df = pd.json_normalize(data)
    print("Number of parsed fake news:", df.shape[0])
    return df


def cls():
    os.system("cls" if os.name == "nt" else "clear")


tokenizer = GPT2Tokenizer.from_pretrained("mrm8488/GPT-2-finetuned-CORD19")
model = GPT2LMHeadModel.from_pretrained("mrm8488/GPT-2-finetuned-CORD19")


def generateFakeNew(input):
    start = time.time()
    inputs = tokenizer.encode(input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=80, do_sample=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    total = time.time() - start
    return text, total


def saveFakeNew(text, file):
    row = [{"text": text, "tag": "false"}]
    rowDf = pd.DataFrame(row)
    rowDf.to_json(file, orient="records", lines=True)
    file.write("\n")


def getAverageAndExpectedTime(timeList, lastTime, index, total):
    if len(timeList) == 4:
        del timeList[0]
    timeList.append(lastTime)
    average = sum(timeList) / len(timeList)
    expected = (total - index + 1) * average
    return timeList, round(average, 3), round(expected, 3)


actualFakePath = "./datasets/true/en_real.json"
newFakeFilePath = "./datasets/true/en_gpt2.json"
newFakefile = open(newFakeFilePath, "a")

actualFakes = jsonToPd(actualFakePath, 2000)

fakes = actualFakes["text"].values.tolist()

# fakes = [
#     "A chain lists recommendations to prevent and treat coronavirus.",
#     "CDC has released an update on how the novel coronavirus can be transmitted",
# ]
nTotal = len(fakes)
times = []
startTime = time.time()

for i, fake in enumerate(fakes):
    if i % 50 == 0:
        newFakefile.flush()
    generatedFake, totalTime = generateFakeNew(fake)
    saveFakeNew(generatedFake, newFakefile)
    cls()
    print(f"Fake news {i} saved onto file")
    times, average, expected = getAverageAndExpectedTime(times, totalTime, i, nTotal)
    progress = (i + 1) * 30 // nTotal
    passedTime = time.time() - startTime
    formatTimePassed = f"{(passedTime//3600)%3600}h {(passedTime//60)%60}min {round(passedTime%60, 3)}segs."
    print(
        f"[{'*'*progress}{'-'*(30 - progress)}] {(i+1) * 100/nTotal}%. {1/average} fake/s .  Expected time-> {(expected//3600)%3600}h {(expected//60)%60}min {expected%60}segs. Time passed: {formatTimePassed}"
    )
