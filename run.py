#!/usr/bin/env python
# coding: utf-8

# # Federated Learning - Model Centric MNIST Example: Train FL Model

# 
# In the "[01-Create Plan](../model-centric/02-ExecutePlan.ipynb)" notebook we created the model, training plan, and averaging plan, and then hosted all of them in PyGrid.
# 
# Such hosted FL model can be now trained using client libraries, SwiftSyft, KotlinSyft, syft.js.
# 
# In this notebook, we'll use FL Client included in the PySyft to do the training.
# 
# ### Credits:
# - Original authors: 
# 
#  - Vova Manannikov - Github: [@vvmnnnkv](https://github.com/vvmnnnkv)
# 
# 
# - Reviewers: 
#  - Patrick Cason - Github: [@cereallcerny](https://github.com/cereallarceny)
# 
# 
# - New Content tested and enriched by: 
#  - Juan M. Aunon - Twitter: [@jm_aunon](https://twitter.com/jm_aunon) - Github: [@jmaunon](https://github.com/jmaunon)
#  

# In[ ]:


%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings("ignore")

import torch as th
from torchvision import datasets, transforms

import numpy as np
import urllib3
import time

import syft as sy
from syft.federated.fl_client import FLClient
from syft.federated.fl_job import FLJob
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient

urllib3.disable_warnings()
sy.make_hook(globals())

# In[ ]:


private_key = """
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAzQMcI09qonB9OZT20X3Z/oigSmybR2xfBQ1YJ1oSjQ3YgV+G
FUuhEsGDgqt0rok9BreT4toHqniFixddncTHg7EJzU79KZelk2m9I2sEsKUqEsEF
lMpkk9qkPHhJB5AQoClOijee7UNOF4yu3HYvGFphwwh4TNJXxkCg69/RsvPBIPi2
9vXFQzFE7cbN6jSxiCtVrpt/w06jJUsEYgNVQhUFABDyWN4h/67M1eArGA540vyd
kYdSIEQdknKHjPW62n4dvqDWxtnK0HyChsB+LzmjEnjTJqUzr7kM9Rzq3BY01DNi
TVcB2G8t/jICL+TegMGU08ANMKiDfSMGtpz3ZQIDAQABAoIBAD+xbKeHv+BxxGYE
Yt5ZFEYhGnOk5GU/RRIjwDSRplvOZmpjTBwHoCZcmsgZDqo/FwekNzzuch1DTnIV
M0+V2EqQ0TPJC5xFcfqnikybrhxXZAfpkhtU+gR5lDb5Q+8mkhPAYZdNioG6PGPS
oGz8BsuxINhgJEfxvbVpVNWTdun6hLOAMZaH3DHgi0uyTBg8ofARoZP5RIbHwW+D
p+5vd9x/x7tByu76nd2UbMp3yqomlB5jQktqyilexCIknEnfb3i/9jqFv8qVE5P6
e3jdYoJY+FoomWhqEvtfPpmUFTY5lx4EERCb1qhWG3a7sVBqTwO6jJJBsxy3RLIS
Ic0qZcECgYEA6GsBP11a2T4InZ7cixd5qwSeznOFCzfDVvVNI8KUw+n4DOPndpao
TUskWOpoV8MyiEGdQHgmTOgGaCXN7bC0ERembK0J64FI3TdKKg0v5nKa7xHb7Qcv
t9ccrDZVn4y/Yk5PCqjNWTR3/wDR88XouzIGaWkGlili5IJqdLEvPvUCgYEA4dA+
5MNEQmNFezyWs//FS6G3lTRWgjlWg2E6BXXvkEag6G5SBD31v3q9JIjs+sYdOmwj
kfkQrxEtbs173xgYWzcDG1FI796LTlJ/YzuoKZml8vEF3T8C4Bkbl6qj9DZljb2j
ehjTv5jA256sSUEqOa/mtNFUbFlBjgOZh3TCsLECgYAc701tdRLdXuK1tNRiIJ8O
Enou26Thm6SfC9T5sbzRkyxFdo4XbnQvgz5YL36kBnIhEoIgR5UFGBHMH4C+qbQR
OK+IchZ9ElBe8gYyrAedmgD96GxH2xAuxAIW0oDgZyZgd71RZ2iBRY322kRJJAdw
Xq77qo6eXTKpni7grjpijQKBgDHWRAs5DVeZkTwhoyEW0fRfPKUxZ+ZVwUI9sxCB
dt3guKKTtoY5JoOcEyJ9FdBC6TB7rV4KGiSJJf3OXAhgyP9YpNbimbZW52fhzTuZ
bwO/ZWC40RKDVZ8f63cNsiGz37XopKvNzu36SJYv7tY8C5WvvLsrd/ZxvIYbRUcf
/dgBAoGBAMdR5DXBcOWk3+KyEHXw2qwWcGXyzxtca5SRNLPR2uXvrBYXbhFB/PVj
h3rGBsiZbnIvSnSIE+8fFe6MshTl2Qxzw+F2WV3OhhZLLtBnN5qqeSe9PdHLHm49
XDce6NV2D1mQLBe8648OI5CScQENuRGxF2/h9igeR4oRRsM1gzJN
-----END RSA PRIVATE KEY-----
""".strip()

public_key = """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzQMcI09qonB9OZT20X3Z
/oigSmybR2xfBQ1YJ1oSjQ3YgV+GFUuhEsGDgqt0rok9BreT4toHqniFixddncTH
g7EJzU79KZelk2m9I2sEsKUqEsEFlMpkk9qkPHhJB5AQoClOijee7UNOF4yu3HYv
GFphwwh4TNJXxkCg69/RsvPBIPi29vXFQzFE7cbN6jSxiCtVrpt/w06jJUsEYgNV
QhUFABDyWN4h/67M1eArGA540vydkYdSIEQdknKHjPW62n4dvqDWxtnK0HyChsB+
LzmjEnjTJqUzr7kM9Rzq3BY01DNiTVcB2G8t/jICL+TegMGU08ANMKiDfSMGtpz3
ZQIDAQAB
-----END PUBLIC KEY-----
""".strip()


# Creating authentication token.

# In[ ]:


import jwt
auth_token = jwt.encode({}, private_key, algorithm='RS256').decode('ascii')

print(auth_token)


# Define `on_accepted`, `on_rejected`, `on_error` handlers.
# 
# The main training loop is located inside `on_accepted` routine.

# In[ ]:


cycles_log = []
status = {
    "ended": False
}

# Called when client is accepted into FL cycle
def on_accepted(job: FLJob):
    print(f"Accepted into cycle {len(cycles_log) + 1}!")

    cycle_params = job.client_config
    batch_size = cycle_params["batch_size"]
    lr = cycle_params["lr"]
    max_updates = cycle_params["max_updates"]

    mnist_dataset = th.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )

    training_plan = job.plans["training_plan"]
    model_params = job.model.tensors()
    losses = []
    accuracies = []

    for batch_idx, (X, y) in enumerate(mnist_dataset):
        X = X.view(batch_size, -1)
        y_oh = th.nn.functional.one_hot(y, 10)
        loss, acc, *model_params = training_plan.torchscript(
            X, y_oh, th.tensor(batch_size), th.tensor(lr), model_params
        )
        losses.append(loss.item())
        accuracies.append(acc.item())
        if batch_idx % 50 == 0:
            print("Batch %d, loss: %f, accuracy: %f" % (batch_idx, loss, acc))
        if batch_idx >= max_updates:
            break

    job.report(model_params)
    # Save losses/accuracies from cycle
    cycles_log.append((losses, accuracies))

# Called when the client is rejected from cycle
def on_rejected(job: FLJob, timeout):
    if timeout is None:
        print(f"Rejected from cycle without timeout (this means FL training is done)")
    else:
        print(f"Rejected from cycle with timeout: {timeout}")
    status["ended"] = True

# Called when error occured
def on_error(job: FLJob, error: Exception):
    print(f"Error: {error}")
    status["ended"] = True


# We use same PyGrid Node where the model was hosted, the model name/version of hosted model.

# In[ ]:


# PyGrid Node address
gridAddress = "ws://alice:5000"

# Hosted model name/version
model_name = "mnist"
model_version = "1.0"


# Let's define routine that creates FL client and starts the FL process.

# In[ ]:


#client.grid_worker.get_connection_speed(client.worker_id)


# In[ ]:


def new_job(self, model_name, model_version) -> FLJob:
        if self.worker_id is None:
            auth_response = self.grid_worker.authenticate(
                self.auth_token, model_name, model_version
            )
            self.worker_id = auth_response["data"]["worker_id"]

        job = FLJob(
            fl_client=self,
            grid_worker=self.grid_worker,
            model_name=model_name,
            model_version=model_version,
        )
        return job


# In[ ]:


def create_client_and_run_cycle():
    client = FLClient(url=gridAddress, auth_token=auth_token, verbose=True)
    client.worker_id = client.grid_worker.authenticate(client.auth_token,model_name,model_version)["data"]["worker_id"]
    job = client.new_job(model_name, model_version)

    # Set event handlers
    job.add_listener(job.EVENT_ACCEPTED, on_accepted)
    job.add_listener(job.EVENT_REJECTED, on_rejected)
    job.add_listener(job.EVENT_ERROR, on_error)

    # Shoot!
    job.start()


# Now we're ready to start FL training.
# 
# We're going to run multiple "workers" until the FL model is fully done and see the progress.
# 
# As we create & authenticate new client each time,
# this emulates multiple different workers requesting a cycle and working on it.

# In[ ]:


while not status["ended"]:
    create_client_and_run_cycle()
    time.sleep(1)


# Let's plot loss and accuracy statistics recorded from each worker.
# Each such worker's statistics is drawn with different color.
# 
# It's visible that loss/accuracy improvement occurs after each `max_diffs` reports to PyGrid,
# because PyGrid updates the model and creates new checkpoint after
# receiving `max_diffs` updates from FL clients.

# In[ ]:


get_ipython().system('pip install matplotlib')


# In[ ]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, figsize=(10, 10))
axs[0].set_title("Loss")
axs[1].set_title("Accuracy")
offset = 0
for i, cycle_log in enumerate(cycles_log):
    losses, accuracies = cycle_log
    x = range(offset, offset + len(losses))
    axs[0].plot(x, losses)
    axs[1].plot(x, accuracies)
    offset += len(losses)
    print(f"Cycle {i + 1}:\tLoss: {np.mean(losses)}\tAcc: {np.mean(accuracies)}")


# In[ ]:




