# Federated Recommender System with Syft

```language=bash
git clone --recursive https://github.com/JCBrouwer/fedrecsys.git
cd fedrecsys
pip install -r requirements.txt


# run PyGrid federated MNIST training

cd PyGrid/apps/node/
./run.sh --id bob --port 5050 --start_local_db &
cd -
python host-plan.py
python check-plan.py
python exec-plan.py
kill -- -$(ps -o pgid=$! | grep -o '[0-9]*')


# run federated recommender training

python fedrec.py
```
