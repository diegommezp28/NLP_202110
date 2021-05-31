
from deep_translator import GoogleTranslator
import json
import yaml
import time
from tqdm import tqdm
from functools import partial
import warnings

failed = 'failed.jsonl'

def logFail(index, target, source, file):
    with open(file, 'a') as f:
        json_line ={}
        json_line['index']=index
        json_line['target']=target
        json_line['source']=source
        json.dump(json_line, f)


def close(files):
    for file in files:
        file.close()


def translate(index, text, target, source='auto', sleep=1):
    try:
        time.sleep(sleep)
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        name = e.__class__.__name__
        if name == 'NotValidPayload':
            return text
        elif name == 'TranslationNotFound':
            logFail(index, target, source, failed)
            return
        print(e)
        print(f'Failed to translate, waiting {sleep+1} seconds to try again')
        return translate(index, text, target, source, sleep+1)


def translate_text(index, text, target, source='auto', n=5000):
    if len(text) < n:
        translated = translate(index, text, target, source)
        if not translated: return
        return translated
    else:
        #warnings.warn(f'The text to translate its larger than {n} characters, translation will be done by chunks.')
        chunks = [text[j:j + n-1] for j in range(0, len(text), n-1)]
        for j in range(len(chunks)):
            translated = translate(index, chunks[j], target, source)
            if not translated: return
            chunks[j] = translated
        return ' '.join(chunks)


def rawincount(filename):
    print("Calculating file size...")
    with open(filename, 'rb') as f:
        bufgen = iter(partial(f.raw.read, 1024*1024), b'')
        return sum(buf.count(b'\n') for buf in bufgen)


with open("./config.yaml", "r+") as file:
    config = yaml.safe_load(file)
    filename = config["filename"]
    inputFile = f"{filename}.jsonl"
    source_lang = config["source_lang"]
    langs = config["langs"]
    num_langs = len(langs)
    outputFiles = []

    num_lines = config["num_lines"] = config["num_lines"] if "num_lines" in config else rawincount(inputFile)
    file.seek(0)
    file.truncate()
    yaml.dump(config, file)

    try:
        for i in range(num_langs):
            outputFiles.append(open(f"{filename}_{langs[i]}.jsonl", 'a'))

        with open(inputFile, "r") as inf:
            for i, line in enumerate(tqdm(inf, total=num_lines)):
                if i <= config["current_index"]:
                    continue
                lang_i = i % num_langs
                json_line = json.loads(line)

                if langs[lang_i] != source_lang:
                    translated = translate_text(i, json_line["body"], langs[lang_i], source_lang)
                    if not translated: continue
                    json_line["body"] = translated
                json_line["language"] = langs[lang_i]

                json.dump(json_line, outputFiles[lang_i])
                outputFiles[lang_i].write('\n')
                config["current_index"] = i
                file.seek(0)
                file.truncate()
                yaml.dump(config, file)
        close(outputFiles)
    except Exception as e:
        print(e)

    finally:
        close(outputFiles)
