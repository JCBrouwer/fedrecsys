import syft as sy
import torch as th
from syft.execution.placeholder import PlaceHolder
from syft.execution.state import State
from syft.execution.translation import TranslationTarget
from syft.grid.clients.model_centric_fl_client import ModelCentricFLClient
from torch import nn

import config

sy.make_hook(globals())
hook.local_worker.framework = None  # force protobuf serialization for tensors
th.random.manual_seed(1)

# This utility function will set tensors as model parameters.
def set_model_params(module, params_list, start_param_idx=0):
    """Set params list into model recursively"""
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


def softmax_cross_entropy_with_logits(logits, targets, batch_size):
    """Calculates softmax entropy
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
def naive_sgd(param, **kwargs):
    return param - kwargs["lr"] * param.grad


# ### Training Plan procedure
#
# We define a routine that will take one batch of training data and model parameters,
# and will update model parameters to optimize them for given loss function using SGD.
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
    updated_params = [naive_sgd(param, lr=lr) for param in model_params]

    # accuracy
    pred = th.argmax(logits, dim=1)
    target = th.argmax(y, dim=1)
    acc = pred.eq(target).sum().float() / batch_size

    return (loss, acc, *updated_params)


# Let's build this procedure into the Plan.
# Dummy input parameters to make the trace
model_params = [param.data for param in model.parameters()]  # raw tensors instead of nn.Parameter
X = th.randn(3, 28 * 28)
y = nn.functional.one_hot(th.tensor([1, 2, 3]), 10)
lr = th.tensor([0.01])
batch_size = th.tensor([3.0])

_ = training_plan.build(X, y, batch_size, lr, model_params, trace_autograd=True)

# Let's look inside the Syft Plan and print out the list of operations recorded.
print(training_plan.code)


# Plan should be automatically translated to torchscript and tensorflow.js, too.
# Let's examine torchscript code:
print(training_plan.torchscript.code)


# Tensorflow.js code:
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


@sy.func2plan()
def avg_plan(avg, item, num):
    new_avg = []
    for i, param in enumerate(avg):
        new_avg.append((avg[i] * num + item[i]) / (num + 1))
    return new_avg


# Build the Plan
_ = avg_plan.build(model_params, model_params, th.tensor([1.0]))

# Let's check Plan contents
print(avg_plan.code)

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
# PyGrid Node address
grid = ModelCentricFLClient(id="alice", address=config.GRID_ADDRESS, secure=False)
grid.connect()  # These name/version you use in worker

client_config = {
    "name": config.MODEL_NAME,
    "version": config.MODEL_VERSION,
    "batch_size": 64,
    "lr": 0.005,
    "max_updates": 100,  # custom syft.js option that limits number of training loops per worker
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
    "iterative_plan": True,  # tells PyGrid that avg plan is executed per diff
}

# If we set __public key__ into model authentication config,
# then PyGrid will validate that submitted JWT auth token is signed with private key.
server_config["authentication"] = {
    "type": "jwt",
    "pub_key": config.PUBLIC_KEY,
}

# Now we're ready to host our federated Training Plan!
model_params_state = State(state_placeholders=[PlaceHolder().instantiate(param) for param in model_params])
response = grid.host_federated_training(
    model=model_params_state,
    client_plans={"training_plan": training_plan},
    client_protocols={},
    server_averaging_plan=avg_plan,
    client_config=client_config,
    server_config=server_config,
)
print("Host response:", response)

# If you see `status: success` this means the plan is successfully hosted in the PyGrid!
