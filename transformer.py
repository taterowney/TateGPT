import torch
import numpy as np
import torch.nn as nn
import math, tqdm, time, os, glob

from tokenizer import encode, decode, START_OF_TEXT, END_OF_TEXT, PADDING_TOKEN, VOCAB_SIZE, pad_end

from data_parser import TRAIN, TEST, VALIDATION

# Gcloud command:
# gcloud compute ssh instance-name

# Screen session
# screen -S python-session
# command; sudo shutdown -h now
# (crtl + a) then d to detach
# To reattach:
# screen -r python-session


# If this is True then the training loops will be cut short so it doesn't go through the entire process
EXPERIMENTING = True

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

# MAXIMUM POWER!!!!!
# MAX_TOKENS = 1000
# FEATURES = 2048
# BATCH_SIZE = 16

# Lame version
MAX_TOKENS = 1000
# FEATURES = 256
FEATURES = 128
BATCH_SIZE = 4

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=MAX_TOKENS):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])




class TransformerModel(nn.Module):
    def __init__(self, features=FEATURES, num_tokens=MAX_TOKENS, vocab_size=VOCAB_SIZE, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, features).to(DEVICE)
        self.positional_encoding = PositionalEncoding(features, 0.1, num_tokens).to(DEVICE)

        self.transformer = nn.Transformer(num_decoder_layers=8, nhead=16, batch_first=True, d_model=features, dropout=dropout).to(DEVICE)

        # self.activation1 = nn.ReLU().to(DEVICE)
        # self.flatten = nn.Flatten().to(DEVICE)
        # self.linear1 = nn.Linear(num_tokens*features, features).to(DEVICE)
        self.activation2 = nn.ReLU().to(DEVICE)
        self.linear2 = nn.Linear(features, VOCAB_SIZE).to(DEVICE)

    def forward(self, prompt, context):

        prompt = self.positional_encoding(self.embedding(prompt)).type(torch.float32)
        context = self.positional_encoding(self.embedding(context)).type(torch.float32)

        tgt_mask = self.transformer.generate_square_subsequent_mask(context.size(1)).to(DEVICE)

        output = self.transformer(prompt, context, tgt_mask=tgt_mask)

        # output = self.flatten(output)

        # output = self.linear1(output)
        output = self.linear2(self.activation2(output))

        # return nn.functional.softmax(output.type(torch.float32), dim=1)
        return output.type(torch.float32)



# DATA LOADING + HANDLING

class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, directory, num_tokens=MAX_TOKENS):
        self.num_tokens = num_tokens
        self.padding_token = PADDING_TOKEN
        from lxml import etree
        self.xml_roots = []
        for file in os.listdir(directory):
            if file.endswith(".xml"):
                self.xml_roots.append(etree.parse( os.path.join(directory, file) ).getroot())
        # keeps track of which n-grams are where
        self.indices = []
        for file_num, root in enumerate(self.xml_roots):
            for tag_num, child in enumerate(root):
                # this includes both the context, and the target (which is the context shifted right by 1)
                length = len(child.find('value').text.split(',')) // (num_tokens + 1)
                for chunk in range(length+1):
                    self.indices.append((file_num, tag_num, chunk))
        self.length = len(self.indices)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_num, tag_num, chunk_num = self.indices[idx]
        root = self.xml_roots[file_num]
        child = root[tag_num]
        body_tokens = list(map(int, child.find('value').text.split(',')))
        chunk = self.get_nth_chunk(body_tokens, chunk_num)
        title_tokens = list(map(int, child.find('key').text.split(',')))
        title_tokens = pad_end(title_tokens, self.num_tokens)
        return torch.tensor(title_tokens, dtype=torch.long), torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

    def get_nth_chunk(self, tokens, n):
        # returns the n-th set of num_tokens+1 tokens
        if len(tokens) >= (n+1)*(self.num_tokens+1):
            return tokens[n*(self.num_tokens+1):(n+1)*(self.num_tokens+1)]
        else:
            return tokens[n*(self.num_tokens+1):] + [self.padding_token] * ((self.num_tokens+1) - len(tokens[n*(self.num_tokens+1):]))

class DebuggingDataset(torch.utils.data.Dataset):
    def __init__(self, num_tokens=MAX_TOKENS):
        self.num_tokens = num_tokens
        self.padding_token = PADDING_TOKEN

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        title_tokens = pad_end([1], self.num_tokens)
        body_tokens = [1, 0]*(self.num_tokens//2) + [1]
        return torch.tensor(title_tokens, dtype=torch.long), torch.tensor(body_tokens[:-1], dtype=torch.long), torch.tensor(body_tokens[1:], dtype=torch.long)

def get_dataloaders(name="default", train=0.9, validation=0.05, test=0.05, batch_size=BATCH_SIZE, num_tokens=MAX_TOKENS):
    if name == "default":
        ds = XMLDataset('cleaned_data', num_tokens=num_tokens)
        train, validation, test = torch.utils.data.random_split(ds, [int(len(ds)*train), int(len(ds)*validation), len(ds) - int(len(ds)*train) - int(len(ds)*validation)])
        return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    elif name == "unittest":
        ds = DebuggingDataset(num_tokens=num_tokens)
        train, validation, test = torch.utils.data.random_split(ds, [90, 5, 5])
        return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER = get_dataloaders("unittest")


def train_loop(model, opt, loss_fn):
    model.train()
    total_loss = 0
    num_batches = 0

    i = 0
    for batch_x, batch_y, batch_target in tqdm.tqdm(TRAIN_LOADER, desc="Training model"):
        batch_x, batch_y, batch_target = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_target.to(DEVICE)

        output = model(batch_y, batch_x)

        output = output.view(-1, output.size(-1))  # Reshape to (batch_size * sequence_length, vocab_size)
        batch_target = batch_target.view(-1)  # Reshape to (batch_size * sequence_length)

        loss = loss_fn(output, batch_target.type(torch.long))

        opt.zero_grad()
        loss.backward()
        opt.step()

        current_loss = loss.detach().item()
        total_loss += current_loss
        num_batches += 1

        i += 1
        if i == 50 and EXPERIMENTING:
            break

    return total_loss/num_batches


def validation_loop(model, loss_fn):
    model.eval()
    total_loss = 0
    num_batches = 0

    i = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_target in tqdm.tqdm(VALIDATION_LOADER, desc="Validating model"):

            batch_x, batch_y, batch_target = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_target.to(DEVICE)

            output = model(batch_y, batch_x)

            output = output.view(-1, output.size(-1))
            batch_target = batch_target.view(-1)

            loss = loss_fn(output, batch_target.type(torch.long))

            total_loss += loss.detach().item()
            num_batches += 1

            i += 1
            if i == 5 and EXPERIMENTING:
                break

    return total_loss/num_batches


def test_model(model, loss_fn):
    # add stuff for token-by-token accuracy
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_target in tqdm.tqdm(TEST_LOADER, desc="Testing model"):
            batch_x, batch_y, batch_target = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_target.to(DEVICE)

            output = model(batch_y, batch_x)

            output = output.view(-1, output.size(-1))
            batch_target = batch_target.view(-1)

            loss = loss_fn(output, batch_target.type(torch.long))

            total_loss += loss.detach().item()
            num_batches += 1

    return total_loss/num_batches


def fit(model, opt, loss_fn, num_epochs=10):
    training_losses = []
    validation_losses = []
    for i in range(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}")
        validation_losses.append(validation_loop(model, loss_fn))
        training_losses.append(train_loop(model, opt, loss_fn))
        print(f"Training loss: {training_losses[-1]}")
        print(f"Validation loss: {validation_losses[-1]}")
        print("\n\n")


def save_model(model, path=f"./models/transformer_model_{int(time.time())}.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path=None):
    if not path:
        path = max(glob.glob("./models/*.pth"), key=os.path.getctime)
    model.load_state_dict(torch.load(path))
    return model

def predict(model, prompt, context, max_length=100):
    model.eval()
    with torch.no_grad():
        prompt = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
        context = torch.tensor(encode(context)[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
        for _ in range(max_length):
            pred = model(prompt, context)
            next_item = pred.topk(1)[1].view(-1)[-1].item()
            next_item = torch.tensor([[next_item]], device=DEVICE)
            context = torch.cat([context, next_item], dim=1)
            if next_item.view(-1).item() == END_OF_TEXT:
                break

        return decode(context.view(-1).tolist())


def test_loss_function(output, target):
    output_tokens = torch.nn.functional.one_hot(torch.tensor(output, dtype=torch.long, device=DEVICE), num_classes=VOCAB_SIZE).unsqueeze(0)
    target_tokens = torch.tensor(target, dtype=torch.long, device=DEVICE).unsqueeze(0)
    output_tokens = output_tokens.view(-1, output_tokens.size(-1))
    target_tokens = target_tokens.view(-1)
    print(output_tokens, target_tokens)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output_tokens.type(torch.float32), target_tokens.type(torch.float32))
    return loss.item()


def predict_tokens(model, prompt, context, max_length=10):
    model.eval()
    with torch.no_grad():
        prompt = torch.tensor(pad_end(prompt, MAX_TOKENS), dtype=torch.long, device=DEVICE).unsqueeze(0)
        context = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
        for _ in range(max_length):
            pred = model(prompt, context)
            next_item = pred.topk(1)[1].view(-1)[-1].item()
            next_item = torch.tensor([[next_item]], device=DEVICE)
            context = torch.cat([context, next_item], dim=1)
            if next_item.view(-1).item() == END_OF_TEXT:
                break

        return context.view(-1).tolist()


if __name__ == '__main__':
    transformer = TransformerModel()
    fit(transformer, torch.optim.Adam(transformer.parameters(), lr=0.0001), nn.CrossEntropyLoss().to(DEVICE), num_epochs=10)
    save_model(transformer)

    #
    # from data_parser import done_message
    # done_message()

    transformer = load_model(transformer)
    # print(predict(transformer, "Albedo", "'''Albedo''' is the science of", 10))

    print(predict_tokens(transformer, [1], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 10))
    # print(test_loss_function([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]))

# TODO: Checkpointing, reverting if model overfits, token accuracy, increased speed of data processing, ACTIVATION FUNCTIONS LOL, tweak learning rate
