from deep_translator import GoogleTranslator
import json
import yaml
import time
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import pickle as pk
import sys

failed = 'failed.jsonl'
dataset = load_dataset("covid_qa_deepset")
dataset_completo = dataset["train"]

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print('sorry not enough arguments')
        exit()
    lang = args[0]
    output_file = args[1]
    last_position = -1 if len(args) == 2 else args[2]


def read_dataframe(dataset):
    contexts = []
    questions = []
    answers = []
    dataset.map(lambda example: contexts.append(example['context']))
    dataset.map(lambda example: questions.append(example['question']))
    dataset.map(lambda example: answers.append(example['answers']))
    return contexts, questions, answers


contexts, questions, answers = read_dataframe(dataset_completo)

def translate_contexts(contexts):
    contextos_traducidos = {}
    for i in range(len(contexts)):
        contexto = contexts[i]
        titulo = contexto.partition("\n")[0];
        if not titulo in contextos_traducidos:
            context2 = translate_text(i, contexto, lang)
            contextos_traducidos[titulo] = context2
    return contextos_traducidos

def logFail(index, target, source, file):
    with open(file, 'a') as f:
        json_line = {}
        json_line['index'] = index
        json_line['target'] = target
        json_line['source'] = source
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
        print(f'Failed to translate, waiting {sleep + 1} seconds to try again')
        return translate(index, text, target, source, sleep + 1)


def translate_text(index, text, target, source='auto', n=5000):
    if len(text) < n:
        translated = translate(index, text, target, source)
        if not translated: return
        return translated
    else:
        # warnings.warn(f'The text to translate its larger than {n} characters, translation will be done by chunks.')
        chunks = [text[j:j + n - 1] for j in range(0, len(text), n - 1)]
        for j in range(len(chunks)):
            translated = translate(index, chunks[j], target, source)
            if not translated: return
            chunks[j] = translated
        return ' '.join(chunks)


def rawincount(filename):
    print("Calculating file size...")
    with open(filename, 'rb') as f:
        bufgen = iter(partial(f.raw.read, 1024 * 1024), b'')
        return sum(buf.count(b'\n') for buf in bufgen)


num_files = len(contexts)

usables = 0
no_usables = 0

with open(output_file, "a") as output:
    print("Empezando ... traduciendo contextos")
    contextos_traducidos = translate_contexts(contexts)
    print(f"{str(len(contextos_traducidos))} contextos traducidos")

    for i in tqdm(range(last_position+1, num_files)):
        context = contexts[i]
        titulo = context.partition("\n")[0];
        answer = answers[i]["text"][0]
        question = questions[i]

        context2 = contextos_traducidos[titulo]
        answer2_text = translate_text(i, answer, lang)

        if answer2_text in context2:
            start = context2.index(answer2_text)
            answer2 = {'answer_start': [start], 'text': [answer2_text]}
            question2 = translate_text(i, question, lang)
            json_line = {"context": context2, "answer": answer2, "question": question2, "id": i}
            json.dump(json_line, output)
            output.write('\n')
            usables += 1
        else:
            no_usables += 1

    print(f'Traducidos: {str(usables)}, no traducido: {str(no_usables)}')
