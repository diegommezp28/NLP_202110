
from deep_translator import GoogleTranslator
import json
import yaml
import time
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import pickle as pk

failed = 'failed.jsonl'
dataset = load_dataset("covid_qa_deepset")
dataset_completo = dataset["train"]

def read_dataframe(dataset):
    contexts = []
    questions = []
    answers = []
    dataset.map(lambda example: contexts.append(example['context']))
    dataset.map(lambda example: questions.append(example['question']))
    dataset.map(lambda example: answers.append(example['answers']))
    return contexts, questions, answers

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


contexts, questions, answers = read_dataframe(dataset_completo)

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
    source_lang = config["source_lang"]
    lang = config["lang"]

    num_files = len(contexts)

    count_usables = []
    count_no_usables = []
    contexts_tra = []
    questions_tra = []
    answers_tra = []
    try:
        for i in range(num_files):
            context = contexts[i]
            #print("contexto og",context)
            question = questions[i]
            answer = answers[i]["text"][0]
            #print("answer og",answer)
            context_tra = translate_text(i,context,lang)
            #print("contexto",context_traducido)
            answer_tra_texto = translate_text(i,answer,lang)
            #print("respuesta",answer_traducido)
            if answer_tra_texto in context_tra:
                contexts_tra.append(context_tra)
                start = context_tra.index(answer_tra_texto)
                answer_tra = {'answer_start':[start], 'text':[answer_tra_texto]}
                answers_tra.append(answer_tra)
                question_tra = translate_text(i,question,lang)
                questions_tra.append(question_tra)
                count_usables.append(i)
            else:
                count_no_usables.append(i)
            print(f'Traducidos: {str(len(count_usables))}, no traducido: {str(len(count_no_usables))}')

        with open("questions.pkl", "wb") as model_file:
            pk.dump(questions_tra, model_file)
        with open("answers.pkl", "wb") as model_file:
            pk.dump(answers_tra, model_file)
        with open("contexts.pkl", "wb") as model_file:
            pk.dump(contexts_tra, model_file)

    except Exception as e:
        print(e)
