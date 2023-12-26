# TorchQL

TorchQL is a framework for programming integrity constraints over machine learning applications in Python.
TorchQL comes with PyTorch support to allow for easier integration of TorchQL queries into the machine learning pipeline.
Users can write queries that specify integrity constraints through a combination of query operations and arbitrary
user-defined functions.
These integrity constraints can then be directly evaluated over models and datasets.

## Demo

You can explore TorchQL without needing to install it using our demo in this [colab notebook](https://colab.research.google.com/drive/1dXsyx20GK6OXuRsQzwANlZzu_0mFqtrZ#scrollTo=ekf17BGSbT0O).


## Installation

To install TorchQL, run
```bash
pip install -e .
```

## Writing your first query

Here is a simple example of queries that can be written in TorchQL to load the MNIST training data and only extract
samples with the label equal to 7.

First, we set up the TorchQL database:

```python
from torchvision import datasets

from torchql import Database, Query


train_data = datasets.MNIST(
        root = 'data',
        train = True,
        download = True,
    )


db = Database("mnist")
db.register_dataset(train_data, "train")
```

Observe that we can directly instantiate a TorchQL table from the PyTorch MNIST train dataset.
Now we write the query and run it on this dataset:

```python
q = Query('seven', base='train').filter(lambda img, label : label == 7)
q(db).sample()
```

The TorchQL `Query` object is instantiated with a name (here `seven`), and a base table over which operations can be
specified (here `train`).
We then specify a `filter` operation to only keep the records that have the label as 7.
Each record contains an image and its label.

We run this query on the database using `q(db)`, and randomly sample a single record from the resulting table.
This is the output of running the above code:
```
Filtering: 100%|██████████| 60000/60000 [00:00<00:00, 992096.76it/s]

(<PIL.Image.Image image mode=L size=28x28>, 7)
```

Please refer to the documentation and the demo for in-depth description of each functionality of TorchQL.


## Documentation

You can find more documentation for TorchQL [here](https://torchql.github.io/torchql/).

## Papers

```
@inproceedings{naik2023torchql,
      title={TorchQL: A Programming Framework for Integrity Constraints in Machine Learning},
      author={Aaditya Naik and Adam Stein and Yinjun Wu and Mayur Naik and Eric Wong},
      booktitle={OOPSLA}
      year={2024}
}
```
