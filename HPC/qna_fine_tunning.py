
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import pickle as pk
import torch
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 3:
        print('sorry not enough arguments')
        exit()
    input_model_file = args[0]
    input_train_dataset_file = args[1]
    output_model_file = args[2]

with open(input_model_file, "rb") as file:
    model = pk.load(file)

with open(input_train_dataset_file, "rb") as file:
    train_dataset = pk.load(file)

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

with open(output_file, "wb") as model_file:
    pk.dump(model, model_file)
