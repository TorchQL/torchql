# Getting Started

## Installation

Please follow the [installation instructions](./installation.md) before proceeding with the following steps.

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


Please refer to the rest of the documentation and the demo for more in-depth description of each functionality of TorchQL.
