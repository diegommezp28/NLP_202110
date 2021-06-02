from transformers import BertForQuestionAnswering, AutoTokenizer
from transformers import pipeline
import sys
import pickle as pk

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 4:
        print('sorry not enough arguments')
        exit()
    model_folder_path = args[0]
    test_contexts_path = args[1]
    test_answers_path = args[2]
    test_questions_path = args[3]

with open(test_contexts_path, "rb") as file:
    test_contexts = pk.load(file)

with open(test_answers_path, "rb") as file:
    test_answers = pk.load(file)

with open(test_questions_path, "rb") as file:
    test_questions = pk.load(file)

model2 = BertForQuestionAnswering.from_pretrained(model_folder_path)
tokenizer2 = AutoTokenizer.from_pretrained(model_folder_path)


nlp = pipeline('question-answering', model=model2, tokenizer=tokenizer2)

#%%

#Basado en: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html

def get_prediction(qid, questions, contexts):
    specific_question = questions[qid]
    specific_context = contexts[qid]

    answer = nlp({
    'question': specific_question,
    'context': specific_context
    })
    return answer["answer"]

#%%

#Basado en: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


#%%

'''
Demorado
'''
def average_metrics(contexts, answers, questions):
    if len(contexts)!= len(answers) or len(contexts)!= len(questions):
        print("Hay un error en el tamaño de los arreglos")
        return
    test_size = len(contexts)
    em_average = 0
    f1_average = 0
    for n in range(test_size):
        #print(n)
        prediction = get_prediction(n, questions, contexts)
        #print("Prediction: ",prediction)
        answer = answers[n]["text"]
        #print("Answer: ",answer)
        em_score = compute_exact_match(prediction, answer)
        f1_score = 1 if em_score == 1 else compute_f1(prediction, answer)
        #print(f"EM: {em_score} \t F1: {f1_score}")
        em_average += em_score
        f1_average += f1_score
    em_average_tot = em_average/test_size
    f1_average_tot = f1_average/test_size
    return em_average_tot, f1_average_tot

#%%

em_average, f1_score = average_metrics( test_contexts, test_answers, test_questions)

#%%
print("Results calculated with "+ len(test_contexts) + " answer / question pairs" )
print("Average F1: " +f1_score+ "\tAverage EM: " +em_average)
