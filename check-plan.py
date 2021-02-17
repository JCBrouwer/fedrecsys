import json

import jwt
import requests
import syft as sy
import torch as th
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB
from syft_proto.execution.v1.state_pb2 import State as StatePB
from websocket import create_connection

import config

sy.make_hook(globals())
hook.local_worker.framework = None  # force protobuf serialization for tensors
th.random.manual_seed(1)

# ### Check hosted data
# This section is optional, here we just double-check the data is properly hosted in the PyGrid by "manually" authenticating, requesting a training cycle and downloading Model and different variants of the Training Plan.

# Helper function to make WS requests
def sendWsMessage(data):
    ws = create_connection("ws://" + config.GRID_ADDRESS)
    ws.send(json.dumps(data))
    message = ws.recv()
    return json.loads(message)


# First, create authentication token.
auth_token = jwt.encode({}, config.PRIVATE_KEY, algorithm="RS256").decode("ascii")
print(auth_token)

# Make authentication request:
auth_request = {
    "type": "model-centric/authenticate",
    "data": {
        "model_name": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "auth_token": auth_token,
    },
}
auth_response = sendWsMessage(auth_request)
print("Auth response: ", json.dumps(auth_response, indent=2))

# Make the cycle request:
cycle_request = {
    "type": "model-centric/cycle-request",
    "data": {
        "worker_id": auth_response["data"]["worker_id"],
        "model": config.MODEL_NAME,
        "version": config.MODEL_VERSION,
        "ping": 1,
        "download": 10000,
        "upload": 10000,
    },
}
cycle_response = sendWsMessage(cycle_request)
print("Cycle response:", json.dumps(cycle_response, indent=2))

worker_id = auth_response["data"]["worker_id"]
request_key = cycle_response["data"]["request_key"]
model_id = cycle_response["data"]["model_id"]
training_plan_id = cycle_response["data"]["plans"]["training_plan"]

# Let's download Model and Training Plan (in various trainslations) and check they are actually workable.
req = requests.get(
    f"http://{config.GRID_ADDRESS}/model-centric/get-model?worker_id={worker_id}&request_key={request_key}&model_id={model_id}"
)
model_data = req.content
pb = StatePB()
pb.ParseFromString(req.content)
model_params_downloaded = protobuf.serde._unbufferize(hook.local_worker, pb)
print("Params shapes:", [p.shape for p in model_params_downloaded.tensors()])

# Plan "list of ops"
req = requests.get(
    f"http://{config.GRID_ADDRESS}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=list"
)
pb = PlanPB()
pb.ParseFromString(req.content)
plan_ops = protobuf.serde._unbufferize(hook.local_worker, pb)
print(plan_ops.code)
print(plan_ops.torchscript)

# Plan "torchscript"
req = requests.get(
    f"http://{config.GRID_ADDRESS}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=torchscript"
)
pb = PlanPB()
pb.ParseFromString(req.content)
plan_ts = protobuf.serde._unbufferize(hook.local_worker, pb)
print(plan_ts.code)
print(plan_ts.torchscript.code)

# Plan "tfjs"
req = requests.get(
    f"http://{config.GRID_ADDRESS}/model-centric/get-plan?worker_id={worker_id}&request_key={request_key}&plan_id={training_plan_id}&receive_operations_as=tfjs"
)
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
