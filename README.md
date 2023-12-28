# TorchQL

TorchQL is a query language for Python-based machine learning models and datasets.

## Demo

Try TorchQL using our demo in this [colab notebook](https://colab.research.google.com/drive/1dXsyx20GK6OXuRsQzwANlZzu_0mFqtrZ).


## Installation

The easiest way to install TorchQL is as a Python package:
```bash
pip install torchql
```

Alternatively, install TorchQL from source to contribute and keep up with the latest updates:
```bash
git clone https://github.com/TorchQL/torchql.git
cd torchql

pip install . # add the -e option to install an editable version of the package
```

## Writing your first query

Here is a simple example of a query that can be written in TorchQL.  It loads the MNIST training dataset and extracts
samples with the label equal to 7.

First, we set up a TorchQL database:

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
Next, we write the query and execute it on this dataset:

```python
q = Query('seven', base='train').filter(lambda img, label : label == 7)
print(q(db).sample())
```

The TorchQL `Query` object is instantiated with a name (here `seven`), and a base table over which operations
can be specified (here `train`).
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

You can find more documentation on TorchQL [here](https://torchql.github.io/torchql/).

## Papers

```
@inproceedings{naik2023torchql,
      title={TorchQL: A Programming Framework for Integrity Constraints in Machine Learning},
      author={Aaditya Naik and Adam Stein and Yinjun Wu and Mayur Naik and Eric Wong},
      booktitle={OOPSLA}
      year={2024}
}
```
