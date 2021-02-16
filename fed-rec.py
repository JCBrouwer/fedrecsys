import random
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as f
import torch.optim as optim
from syft.frameworks.torch.fl import utils
from torch.utils.data import DataLoader, Dataset

hook = sy.TorchHook(torch)


class Arguments:
    def __init__(self):
        self.batch_size = 1
        self.test_batch_size = 50
        self.seed = 1


# bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
# alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        data_tensor = torch.stack([self.user_tensor[index], self.item_tensor[index]], dim=0)
        # print(data_tensor.shape)
        return data_tensor, self.target_tensor[index]

    def __len__(self):
        return len(self.user_tensor)


# sample_generator = SampleGenerator(ratings=ml1m_rating)
# # base_federated=dataset.federate((bob, alice))
# base = sample_generator.instance_a_train_loader(4, 32)
rs_cols = ["user_id", "movie_id", "rating", "unix_timestamp"]
train_data = pd.read_csv("Data/ua.base", sep="\t", names=rs_cols, encoding="latin-1")
user_ids = train_data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = train_data["movie_id"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
train_data["user"] = train_data["user_id"].map(user2user_encoded)
train_data["movie"] = train_data["movie_id"].map(movie2movie_encoded)

EMBEDDING_SIZE = 50
NUM_USERS = len(user2user_encoded)
NUM_MOVIES = len(movie_encoded2movie)
train_data["rating"] = train_data["rating"].values.astype(np.float32)
# federated_SVHN=UserItemRatingDataset(torch.LongTensor(train_data["user"]),torch.LongTensor(train_data["movie"]),torch.FloatTensor(train_data["rating"]))

# workers=[bob,alice]
# federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader
#                         UserItemRatingDataset(torch.LongTensor(train_data["user"]),torch.LongTensor(train_data["movie"]),torch.FloatTensor(train_data["rating"])).federate(tuple(workers)),batch_size=1024,shuffle=True,iter_per_worker=True)

# Create virtual workers

workers = []

for user_id in user_ids:
    worker = sy.VirtualWorker(hook, id="user_" + str(user_id))
    workers.append(worker)

federated_train_loader = sy.FederatedDataLoader(  # <-- this is now a FederatedDataLoader
    UserItemRatingDataset(
        torch.LongTensor(train_data["user"]),
        torch.LongTensor(train_data["movie"]),
        torch.FloatTensor(train_data["rating"]),
    ).federate(tuple(workers)),
    batch_size=1024,
    shuffle=True,
    iter_per_worker=True,
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_users = NUM_USERS
        self.num_movies = NUM_MOVIES
        self.embedding_size = EMBEDDING_SIZE

        self.user_embedding = nn.Embedding(NUM_USERS, EMBEDDING_SIZE)
        self.movie_embedding = nn.Embedding(NUM_MOVIES, EMBEDDING_SIZE)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(50, 50))
        self.output_layer = nn.Linear(EMBEDDING_SIZE, 1)

    def forward(self, train_data):
        users = torch.as_tensor(train_data[:, 0])
        movies = torch.as_tensor(train_data[:, 1])
        user_embedding_x = self.user_embedding(users)
        movie_embedding_y = self.movie_embedding(movies)
        prod = torch.mul(user_embedding_x, movie_embedding_y)
        # concatenate user and movie embeddings to form input to 1 dim
        # x = torch.cat([user_embedding_x, movie_embedding_y], 1)

        # for idx, _ in enumerate(range(len(self.fc_layers))):
        #     x = self.fc_layers[idx](x)
        #     x = F.dropout(x, p=0.2)
        #     x = F.batch_norm(x)
        #     x = F.relu(x)

        logit = self.output_layer(prod)
        # The sigmoid activation forces the rating to between 0 and 1
        rating = torch.sigmoid(logit)
        return rating


# from syft.frameworks.torch.fl import utils
# embed_size=50
# model = Model(num_users,num_movies,embed_size)
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# lr=0.1
# def trainn():
#     for epoch in range(0, 5):
#         for batch_idx, (data, target) in enumerate(federated_train_loader):
#             print(data.location)
#             print(target.location)
#             # send the model to the client device where the data is present
#             model.send(data.location)
#             # training the model
#             optimizer.zero_grad()
#             prediction = model(data)
#             loss = F.mse_loss(prediction.view(-1), target)
#             loss.backward()
#             optimizer.step()
#             # get back the improved model
#             print(loss.get())
#             model.get()
#             return utils.federated_avg({
#                 "model": model
#             })


# for epoch in range(5):
#     start_time = time.time()
#     print("Epoch Number {epoch + 1}")
#     federated_model = trainn()
#     model = federated_model
# #     test(federated_model)
#     total_time = time.time() - start_time
#     print('Communication time over the network', round(total_time, 2), 's\n')


def train_on_batches(worker, batches, model_in, device, lr):
    """Train the model on the worker on the provided batches
    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps
    Returns:
        model, loss: obtained model and loss after training
    """
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)
    loss_local = False
    LOG_INTERVAL = 25

    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.mse_loss(output.view(-1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            print(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    return model, loss


def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
    """retrieve next nr_batches of the federated data loader and group
    the batches by worker
    Args:
        fdataloader (sy.FederatedDataLoader): federated data loader
        over which the function will iterate
        nr_batches (int): number of batches (per worker) to retrieve
    Returns:
        Dict[syft.workers.BaseWorker, List[batches]]
    """
    batches = {}
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        batches[worker] = []
    try:
        for i in range(nr_batches):
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        pass
    return batches


def train(model, device, federated_train_loader, lr, federate_after_n_batches, abort_after_one=False):
    model.train()

    nr_batches = federate_after_n_batches

    models = {}
    loss_values = {}

    iter(federated_train_loader)  # initialize iterators
    batches = get_next_batches(federated_train_loader, nr_batches)
    counter = 0

    while True:
        print(f"Starting training round, batches [{counter}, {counter + nr_batches}]")
        data_for_all_workers = True
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                models[worker], loss_values[worker] = train_on_batches(worker, curr_batches, model, device, lr)
            else:
                data_for_all_workers = False
        counter += nr_batches
        if not data_for_all_workers:
            print("At least one worker ran out of data, stopping.")
            break

        model = utils.federated_avg(models)
        batches = get_next_batches(federated_train_loader, nr_batches)
        if abort_after_one:
            break
    return model


model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.1)
lr = 0.1
federate_after_n_batches = 50
device = torch.device("cpu")
for epoch in range(1, 5):
    print("Starting epoch {}/{}".format(epoch, 5))
    model = train(model, device, federated_train_loader, lr, federate_after_n_batches)

torch.save(model, "movie-recommender.pt")
