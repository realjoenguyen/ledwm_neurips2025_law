# %%
# for s in ["s1", "s2", "s3"]:
# for task in ["train", "eval", "test"]:
# start = time()
# env = MessengerSent(s)

import os
import pickle
import sys
from typing import Counter

from messenger.envs import manual
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from ledwm.nets.EncoderSentHist import NUM_ENTITIES
from ledwm.embodied.envs.MessengerSent import MIN_HIST_LEN, MessengerSent

# set gpu = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

env = MessengerSent("s2", mode="train", entity_track=True)
NUM_EPS = 1000
LEN_EPS = 32
NUM_ENTITIES = 3
MIN_HIST_LEN = 6
dataset = []

# debug
# if self._step >= HIST_LEN and self.mode == "train":
#     # assert res['dp'] is not full of 0
#     if np.all(res["dp"] == 0):
#         print(f"{self._step=}, {res['dp']=}")
#     assert not np.all(res["dp"] == 0), f"{self._step=}, {res['dp']=}"
# debug
# min_res = np.min(res["dp"], -1)
# max_res = np.max(res["dp"], -1)
# zeros = away = closer = 0
# for i in range(self.num_entities_task):
#     zeros += min_res[i] == 0 and max_res[i] == 0
#     away += min_res[i] < 0
#     closer += max_res[i] > 0
# if self._step >= MIN_HIST_LEN and self.mode == "train":
#     assert zeros == 1, f"{zeros=}"
#     assert away == 1, f"{away=}"
#     assert closer == 1, f"{closer=}"


# def detect_movement(dp, obs=None, t=None):
#     min_res = np.min(dp, -1)
#     max_res = np.max(dp, -1)
#     MIN_THRESHOLD = 0

#     if np.all(min_res == 0) and np.all(max_res == 0):
#         return "stay"

#     # elif majority of res < 0
#     elif np.sum(dp < 0) > np.sum(dp > 0) + MIN_THRESHOLD and np.sum(dp < 0) > 0:
#         return "closer"

#     elif np.sum(dp < 0) < np.sum(dp > 0) + MIN_THRESHOLD and np.sum(dp > 0) > 0:
#         return "away"

#     # else:
#     #     return "notsure"

#     # raise ValueError(
#     #     f"{min_res=}, {max_res=}, {dp=}, {obs=}, {t=}, \n{np.sum(min_res < 0)=}, {np.sum(max_res > 0)=}"
#     # )

mode = "train_eval_test"
task = "s2"
model_sent = "mpnet"
fname = f"ledwm/embodied/envs/data/messenger/{mode}_{task}_{model_sent}.pkl"
with open(fname, "rb") as f:
    sent2id, id2sentemb = pickle.load(f)
    # swap
    sent2id, id2sentemb = id2sentemb, sent2id
    print(
        f"Loading {len(id2sentemb)}, {id2sentemb.shape} mean and ids sent embs from {fname}",
    )

id2send = {v: k for k, v in sent2id.items()}
# %%

raw_dataset = []

for t in tqdm(range(NUM_EPS)):
    obs = env.reset()
    entity2move = {}
    error_cnt = 0
    # print(env._env.cur_env.manual2role)
    manual2role = env._env.cur_env.manual2role
    class2id = {"immovable": 0, "fleeing": 1, "chaser": 2}
    # print(env._env.cur_env.shuffled_manual_ids)
    # print(env._env.cur_env.shuffled_manual)
    # print("len of dataset:", len(dataset))
    # print(obs)
    # print([id2send[i] for i in obs["sent_ids"]])
    # break

    for l in range(LEN_EPS):
        obs, reward, done, info = env.step(env.action_space.sample())
        # print(f'{obs["entity_pos"]=}')
        # print(obs["dp"].shape)

        for i in range(NUM_ENTITIES):
            # print(obs["dp"][i])
            # print(detect_movement(obs["dp"][i]))
            # movement = detect_movement(obs["dp"][i], obs, l)
            # if movement == "notsure":
            #     error_cnt += 1
            #     cprint("notsure", "red")
            # else:
            #     if i not in entity2move:
            #         entity2move[i] = movement
            #     else:
            #         if l > MIN_HIST_LEN:
            #             assert (
            #                 entity2move[i] == movement
            #             ), f"{entity2move[i]=}, {movement=}, {l=}, {obs=}, {i=}"

            #     entity2move[i] = movement
            # print("")
            # print(obs['entity_pos_hist'].shape)

            if obs["manual_ids"][i] in manual2role:
                # feature = np.concatenate(
                #     [obs["entity_pos_hist"][i], obs["avatar_pos_hist"][0]], -1
                # )
                label = manual2role[obs["manual_ids"][i]]
                # feature = np.concatenate([obs["avatar_pos_hist"][0], obs["dp"][i]], -1)
                feature = obs["dp"][i]
                # replace feature: keep 0 and -1, replace others with label
                # feature = np.where(
                #     (feature != 0) & (feature != -1), class2id[label], feature
                # )
                # feature = np.where(feature == 0, class2id[label], feature)
                # print(feature, label)
                # if l >= 2:
                # add l to feature

                # if l <= 3 and feature are all zeros, skip
                if l <= 3 and np.all(feature == 0):
                    continue
                # feature = np.concatenate([feature, [l]])
                # add sent emb to feature
                # sent_embs = id2sentemb[obs["sent_ids"][i]]
                # print(sent_embs.shape)
                # break
                # add sent ids
                feature = (feature, obs["sent_ids"])
                # print(feature)
                # break
                # raw_dataset.append((feature, manual2role[obs["manual_ids"][i]]))
                raw_dataset.append((feature, obs["manual_ids"][i]))

        if done:
            break
    # break

# %%
# len(dataset)
# print(dataset[:5])
print(len(raw_dataset))
for k, v in raw_dataset[:20]:
    print(k, v)

# take random 20 samples from a list
import random

samples = random.sample(raw_dataset, 20)
for k, v in samples:
    print(k, v)


# %%

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


class CustomDataset(Dataset):
    def __init__(self, data, id2send):
        self.data = data
        self.id2send = id2send  # Mapping of sentence ID to embeddings
        self.classes = {
            "immovable": 0,
            "fleeing": 1,
            "chaser": 2,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature, sent_ids, label = (
            self.data[index][0][0],
            self.data[index][0][1],
            self.data[index][1],
        )
        sent_embs = np.stack([self.id2send[sent_id] for sent_id in sent_ids])
        return feature, sent_embs, label


# Split the dataset into 80% training and 20% test
dataset = CustomDataset(raw_dataset, id2sentemb)
# print(dataset[0])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# raw_dataset[2]
# id2sentemb[2]
for v in dataset[0]:
    print(v.shape)


# %%
class AttentionModel(nn.Module):
    def __init__(self, input_size=33, embedding_size=768, hidden_size=32):
        super(AttentionModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Query and Key Linear Transformations
        # self.query_transform = nn.Linear(input_size, hidden_size)
        # mlp instead
        self.query_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(embedding_size, hidden_size),
        )
        self.key_transform = nn.Linear(embedding_size, hidden_size)

        # Scaling factor for dot product attention
        self.scale = np.sqrt(hidden_size)

    def forward(self, x, sent_embs):
        # print(x.shape, sent_embs.shape)
        # x: (batch_size, input_size)
        # sent_embs: (batch_size, num_sent_ids, embedding_size)

        # Transform the input features (query) and sentence embeddings (key)
        query = self.query_transform(x)  # (batch_size, hidden_size)
        key = self.key_transform(sent_embs)  # (batch_size, num_sent_ids, hidden_size)
        # print(query.shape, key.shape)

        # Compute dot product for each transformed sentence embedding in the batch
        dot_products = torch.bmm(key, query.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, num_sent_ids)

        # Normalize by the scale factor
        logits = dot_products / self.scale

        # Aggregate the scores (e.g., summing or averaging)
        # logits = normalized_dot_products.mean(dim=1)  # (batch_size,)
        # print(logits.dtype)

        # Compute softmax probabilities
        # scores = torch.nn.functional.softmax(aggregated_scores, dim=-1)
        return logits


# Step 4: Set Up Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Define model, criterion, and optimizer
model = AttentionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, sent_embs, labels in train_loader:
        inputs, sent_embs, labels = (
            inputs.to(device).float(),
            sent_embs.to(device).float(),
            labels.to(device).long(),
        )

        # print(labels)
        # Forward pass
        outputs = model(inputs, sent_embs)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute loss and accuracy
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
    )

# Step 5: Evaluate the Model on Test Dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, sent_embs, labels in test_loader:
        inputs, sent_embs, labels = (
            inputs.to(device).float(),
            sent_embs.to(device).float(),
            labels.to(device).long(),
        )
        outputs = model(inputs, sent_embs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
