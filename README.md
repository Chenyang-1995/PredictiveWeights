# Using Predicted Weights for Ad Delivery

Code accompanying the paper
> [Using Predicted Weights for Ad Delivery](https://arxiv.org/abs/2106.01079) \
> Thomas Lavastida, Benjamin Moseley, R. Ravi and Chenyang Xu


# Requirements

Python >= 3.6.11


matplotlib >= 3.3.1


# Dataset

We use the Yahoo! Search Marketing Advertiser Bid-Impression-Click
data, which is available at https://webscope.sandbox.yahoo.com/catalog.php?datatype=a upon request.

# Experiments

We first preprocess the original data,
```
python data_preprocess.py
```

Then to conduct the learnability experiments, run
```
python learnability_test.py
```

To conduct the robustness experiments, run
```
python robustness_test.py
```
Note that the default setting of the experiments is least-degree quota and random arrival order.

We provide a function in ``draw.py``, which could be useful when drawing figures for the experiments.

