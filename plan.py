#!/usr/bin/env python
# coding: utf-8

# # Federated Learning - Model Centric MNIST Example: Create Plan

# Data-centric federated learning is a term we use at OpenMined to define federated learning that is done when the data is hosted in a central location (i.e. PyGrid). This can be contrasted with model-centric federated learning (see [FL tutorial from data-centric approach](../data-centric/mnist/01-FL-mnist-populate-a-grid-node.ipynb)), which is where you have a model that is hosted in a central location (i.e. PyGrid) and various workers (edge devices, like a mobile phone or a web browser) can pull down that model to train or inference against it using data located on the worker itself.

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

# # Federated Learning Training Plan: Create Plan
# 
# This notebook is the 1st part of tutorial that demonstrates how to create Model-Centric Federated Learning process that trains a simple MNIST classifier model.
# 
# This part will walk you through following steps:
# 1. Defining the Model
# 1. Defining the Training Plan (that runs on the client)
# 1. Defining the Averaging Plan (that runs on the server)
# 1. Hosting all created assets in the PyGrid
# 
# Current list of problems:
#  * `tensor.shape` is not traceable inside the Plan (issue [#3554](https://github.com/OpenMined/PySyft/issues/3554)).
#  * Autograd/Plan tracing doesn't work with native torch's loss functions and optimizers.
#  * others?
# 

# ## 0 - Previous setup
# 
# Components:
# 
#  - PyGrid Node Alice (http://alice:5000)
# 
# This tutorial assumes that these components are running in background. See [instructions](https://github.com/OpenMined/PyGrid/tree/dev/examples#how-to-run-this-tutorial) for more details.

# ### Import dependencies
# Here we import core dependencies

# In[ ]:


%load_ext autoreload
%autoreload 2

import syft as sy
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from syft.execution.state import State
from syft.execution.placeholder import PlaceHolder
from syft.execution.translation import TranslationTarget

import torch as th
from torch import nn

import os
from websocket import create_connection
import websockets
import json
import requests

sy.make_hook(globals())
hook.local_worker.framework = None # force protobuf serialization for tensors
th.random.manual_seed(1)

# This utility function will set tensors as model parameters.

# In[ ]:


def set_model_params(module, params_list, start_param_idx=0):
    """ Set params list into model recursively
    """
    param_idx = start_param_idx

    for name, param in module._parameters.items():
        module._parameters[name] = params_list[param_idx]
        param_idx += 1

    for name, child in module._modules.items():
        if child is not None:
            param_idx = set_model_params(child, params_list, param_idx)

    return param_idx


# ## Step 1: Define the model
# 
# This model will train on MNIST data, it's very simple yet can demonstrate learning process.
# There're 2 linear layers: 
# 
# * Linear 784x392
# * ReLU
# * Linear 392x10 

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = Net()


# ## Step 2: Define Training Plan
# ### Loss function 
# Batch size needs to be passed because otherwise `target.shape[0]` is not traced inside Plan yet (Issue [#3554](https://github.com/OpenMined/PySyft/issues/3554)).
# 

# In[ ]:


def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """ Calculates softmax entropy
        Args:
            * logits: (NxC) outputs of dense layer
            * targets: (NxC) one-hot encoded labels
            * batch_size: value of N, temporarily required because Plan cannot trace .shape
    """
    # numstable logsoftmax
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    # NLL, reduction = mean
    return -(targets * log_probs).sum() / batch_size


# ### Optimization function
#  
# Just updates weights with grad*lr.
# 
# Note: can't do inplace update because of Autograd/Plan tracing specifics.

# In[ ]:


def naive_sgd(param, **kwargs):
    return param - kwargs['lr'] * param.grad


# ### Training Plan procedure
# 
# We define a routine that will take one batch of training data and model parameters,
# and will update model parameters to optimize them for given loss function using SGD.

# In[ ]:


@sy.func2plan()
def training_plan(X, y, batch_size, lr, model_params):
    # inject params into model
    set_model_params(model, model_params)

    # forward pass
    logits = model.forward(X)
    
    # loss
    loss = softmax_cross_entropy_with_logits(logits, y, batch_size)

    # backprop
    loss.backward()

    # step
    updated_params = [
        naive_sgd(param, lr=lr)
        for param in model_params
    ]
    
    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum().float() / batch_size

    return (
        loss,
        acc,
        *updated_params
    )


# Let's build this procedure into the Plan.

# In[ ]:


# Dummy input parameters to make the trace
model_params = [param.data for param in model.parameters()]  # raw tensors instead of nn.Parameter
X = th.randn(3, 28 * 28)
y = nn.functional.one_hot(th.tensor([1, 2, 3]), 10)
lr = th.tensor([0.01])
batch_size = th.tensor([3.0])

_ = training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)


# Let's look inside the Syft Plan and print out the list of operations recorded.

# In[ ]:


print(training_plan.code)


# Plan should be automatically translated to torchscript and tensorflow.js, too.
# Let's examine torchscript code:

# In[ ]:


print(training_plan.torchscript.code)


# Tensorflow.js code:

# In[ ]:


training_plan.base_framework = TranslationTarget.TENSORFLOW_JS.value
print(training_plan.code)
training_plan.base_framework = TranslationTarget.PYTORCH.value


# ## Step 3: Define Averaging Plan
# 
# Averaging Plan is executed by PyGrid at the end of the cycle,
# to average _diffs_ submitted by workers and update the model
# and create new checkpoint for the next cycle.
# 
# _Diff_ is the difference between client-trained
# model params and original model params,
# so it has same number of tensors and tensor's shapes
# as the model parameters.
# 
# We define Plan that processes one diff at a time.
# Such Plans require `iterative_plan` flag set to `True`
# in `server_config` when hosting FL model to PyGrid.
# 
# Plan below will calculate simple mean of each parameter.

# In[ ]:


@sy.func2plan()
def avg_plan(avg, item, num):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg

# Build the Plan
_ = avg_plan.build(model_params, model_params, th.tensor([1.0]))


# In[ ]:


# Let's check Plan contents
print(avg_plan.code)


# In[ ]:


# Test averaging plan
# Pretend there're diffs, all params of which are ones * dummy_coeffs
dummy_coeffs = [1, 5.5, 7, 55]
dummy_diffs = [[th.ones_like(param) * i for param in model_params] for i in dummy_coeffs]
mean_coeff = th.tensor(dummy_coeffs).mean().item()

# Remove original function to make sure we execute traced Plan
avg_plan.forward = None

# Calculate avg value using our plan
avg = dummy_diffs[0]
for i, diff in enumerate(dummy_diffs[1:]):
    avg = avg_plan(list(avg), diff, th.tensor([i + 1]))

# Avg should be ones*mean_coeff for each param
for i, param in enumerate(model_params):
    expected = th.ones_like(param) * mean_coeff
    assert avg[i].eq(expected).all(), f"param #{i}"


# ## Step 4: Host in PyGrid
# 
# Let's now host everything in PyGrid so that it can be accessed by worker libraries (syft.js, KotlinSyft, SwiftSyft, or even PySyft itself).

# Follow PyGrid [README](https://github.com/OpenMined/PyGrid/#getting-started) to start PyGrid Node. In the code below we assume that the PyGrid Node is running on `127.0.0.1`, port `5000`.

# Define name, version, configs.

# In[ ]:


gridAddress = "alice:5000"


# In[ ]:


# PyGrid Node address

grid = ModelCentricFLClient(id="test", address=gridAddress, secure=False)
grid.connect() # These name/version you use in worker

name = "mnist" 
version = "1.0"

client_config = {
    "name": name,
    "version": version,
    "batch_size": 64,
    "lr": 0.005,
    "max_updates": 100  # custom syft.js option that limits number of training loops per worker
}

server_config = {
    "min_workers": 5,
    "max_workers": 5,
    "pool_selection": "random",
    "do_not_reuse_workers_until_cycle": 6,
    "cycle_length": 28800,  # max cycle length in seconds
    "num_cycles": 5,  # max number of cycles
    "max_diffs": 1,  # number of diffs to collect before avg
    "minimum_upload_speed": 0,
    "minimum_download_speed": 0,
    "iterative_plan": True  # tells PyGrid that avg plan is executed per diff
}


# ### Authentication (optional)
# Let's additionally protect the model with simple authentication for workers.
# 
# PyGrid supports authentication via JWT token (HMAC, RSA) or opaque token
# via remote API.
# 
# We'll try JWT/RSA. Suppose we generate RSA keys:
# ```
# openssl genrsa -out private.pem
# openssl rsa -in private.pem -pubout -out public.pem
# ```

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


# If we set __public key__ into model authentication config,
# then PyGrid will validate that submitted JWT auth token is signed with private key.

# In[ ]:


server_config["authentication"] = {
    "type": "jwt",
    "pub_key": public_key,
}


# Now we're ready to host our federated Training Plan!

# In[ ]:


model_params_state = State(
    state_placeholders=[
        PlaceHolder().instantiate(param)
        for param in model_params
    ]
)

response = grid.host_federated_training(
    model=model_params_state,
    client_plans={'training_plan': training_plan},
    client_protocols={},
    server_averaging_plan=avg_plan,
    client_config=client_config,
    server_config=server_config
)

#print("Host response:", response)


# In[ ]:


print("Host response:", response)


# If you see `status: success` this means the plan is successfully hosted in the PyGrid!

# ### Check hosted data

# This section is optional, here we just double-check the data is properly hosted in the PyGrid by "manually" authenticating, requesting a training cycle and downloading Model and different variants of the Training Plan.

# In[ ]:


# Helper function to make WS requests
    
def sendWsMessage(data):

    ws = create_connection('ws://' + gridAddress)

    ws.send(json.dumps(data))
    message = ws.recv()
    return json.loads(message)


# First, create authentication token.

# In[ ]:


get_ipython().system('pip install pyjwt')
import jwt
auth_token = jwt.encode({}, private_key, algorithm='RS256').decode('ascii')

print(auth_token)


# Make authentication request:

# In[ ]:


auth_request = {
    "type": "model-centric/authenticate",
    "data": {
        "model_name": name,
        "model_version": version,
        "auth_token": auth_token,
    }
}
auth_response = sendWsMessage(auth_request)
print('Auth response: ', json.dumps(auth_response, indent=2))


# Make the cycle request:

# In[ ]:


cycle_request = {
    "type": "model-centric/cycle-request",
    "data": {
        "worker_id": auth_response['data']['worker_id'],
        "model": name,
        
        "version": version,
        "ping": 1,
        "download": 10000,
        "upload": 10000,
    }
}
cycle_response = sendWsMessage(cycle_request)
print('Cycle response:', json.dumps(cycle_response, indent=2))

worker_id = auth_response['data']['worker_id']
request_key = cycle_response['data']['request_key']
model_id = cycle_response['data']['model_id'] 
training_plan_id = cycle_response['data']['plans']['training_plan']


# Let's download Model and Training Plan (in various trainslations) and check they are actually workable.
# 

# In[ ]:


# Model
req = requests.get(f"http://{gridAddress}/model-centric/get-model?worker_id={worker_id}&request_key={request_key}&model_id={model_id}")
model_data = req.content
pb = StatePB()
pb.ParseFromString(req.content)
model_params_downloaded = protobuf.serde._unbufferize(hook.local_worker, pb)
print("Params shapes:", [p.shape for p in model_params_downloaded.tensors()])


# In[ ]:


# Plan "list of ops"
req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=list")
pb = PlanPB()
pb.ParseFromString(req.content)
plan_ops = protobuf.serde._unbufferize(hook.local_worker, pb)
print(plan_ops.code)
print(plan_ops.torchscript)


# In[ ]:


# Plan "torchscript"
req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=torchscript")
pb = PlanPB()
pb.ParseFromString(req.content)
plan_ts = protobuf.serde._unbufferize(hook.local_worker, pb)
print(plan_ts.code)
print(plan_ts.torchscript.code)


# In[ ]:


# Plan "tfjs"
req = requests.get(f"http://{gridAddress}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=tfjs")
pb = PlanPB()
pb.ParseFromString(req.content)
plan_tfjs = protobuf.serde._unbufferize(hook.local_worker, pb)
print(plan_tfjs.code)


# ## Step 5: Train
# 
# To train hosted model, use one of the existing FL workers:
#  * PySyft - see the "[Part 02 - Execute Plan](Part%2002%20-%20Execute%20Plan.ipynb)" notebook that
# has example of using Python FL worker.
#  * [SwiftSyft](https://github.com/OpenMined/SwiftSyft)
#  * [KotlinSyft](https://github.com/OpenMined/KotlinSyft)
#  * [syft.js](https://github.com/OpenMined/syft.js)

# # Congratulations!!! - Time to Join the Community!
# 
# Congratulations on completing this notebook tutorial! If you enjoyed this and would like to join the movement toward privacy preserving, decentralized ownership of AI and the AI supply chain (data), you can do so in the following ways!
# 
# ### Star PyGrid on GitHub
# 
# The easiest way to help our community is just by starring the GitHub repos! This helps raise awareness of the cool tools we're building.
# 
# - [Star PyGrid](https://github.com/OpenMined/PyGrid)
# 
# ### Join our Slack!
# 
# The best way to keep up to date on the latest advancements is to join our community! You can do so by filling out the form at [http://slack.openmined.org](http://slack.openmined.org)
# 
# ### Join a Code Project!
# 
# The best way to contribute to our community is to become a code contributor! At any time you can go to PySyft GitHub Issues page and filter for "Projects". This will show you all the top level Tickets giving an overview of what projects you can join! If you don't want to join a project, but you would like to do a bit of coding, you can also look for more "one off" mini-projects by searching for GitHub issues marked "good first issue".
# 
# - [PySyft Projects](https://github.com/OpenMined/PySyft/issues?q=is%3Aopen+is%3Aissue+label%3AProject)
# - [Good First Issue Tickets](https://github.com/OpenMined/PyGrid/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
# 
# ### Donate
# 
# If you don't have time to contribute to our codebase, but would still like to lend support, you can also become a Backer on our Open Collective. All donations go toward our web hosting and other community expenses such as hackathons and meetups!
# 
# [OpenMined's Open Collective Page](https://opencollective.com/openmined)
