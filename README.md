Code use to run the experiments for my bachelor's degree thesis on the Study of Manifold Geometry using Non-Negative Kernel Graphs. The repository consists of three folders with Python code. Additional folders can be created for storing data and results.

### src

Folder containing the code that can be used on any given dataset.

- *data/generate.py*: Python script that generates a series of synthetic datasets.

- *nnk_graph.py*: Function used to perform NNK optimization. Sigma can be defined as global or local inside the *get_nnk_weighted_graph* function.

- *nnk_metrics.py*: Set of functions that easily allow us to retrieve the value of a series of metrics related to NNK graphs.

### src_cifar10

This folder contains the code necessary for performing the experiments that use the tools from the previous folder, applied to the feature vectors of deep neural networks.

We create a Pytorch class that inherits from the nn.Module and in which the NNK process is performed. NNK is performed only in validation, and after each of the pooling layers. We save the results of each NNK layer with the timestamp on the name. This allows us to relate the layers' results to the layers using the order and time of execution.

Train: python3 src_cifar10/main_train.py 

Run NNK: Train: python3 src_cifar10/main_nnk.py -r 

### src_visualization

The folder contains a series of scripts used to create the figures shown on the report.
