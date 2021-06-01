
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import pickle as pk
import torch
import sys
from transformers import BertForQuestionAnswering, AutoTokenizer

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print('sorry not enough arguments')
        exit()
    input_encodings_file = args[0]
    model_path = args[1]
    lang = "en" if len(args) == 2 else args[2]

if lang == "en":
    modelname = 'deepset/bert-base-cased-squad2'
elif lang == "es":
    modelname = ""
else:
    modelname = ""

model = BertForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)

with open(input_encodings_file, "rb") as file:
    train_encodings = pk.load(file)

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)

# setup GPU/CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device
model.to(device)
# activate training mode of model
model.train()
# initialize adam optimizer with weight decay (reduces chance of overfitting)
optim = AdamW(model.parameters(), lr=5e-5)

# initialize data loader for training data
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(3):
    # set model to train mode
    model.train()
    # setup loop (we use tqdm for the progress bar)
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all the tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        # train model on batch and return outputs (incl. loss)
        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        # extract loss
        loss = outputs[0]
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
