import unittest, torch

from torchvision import datasets
from torchvision.transforms import ToTensor

from torchql import Database, Query

class Testtorchql(unittest.TestCase):
    def setUp(self):
        self.train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )

        self.test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )

    def test_register(self):
        self.database = Database("mnist_db")
        self.database.register_dataset("train", self.train_data)
        self.database.register_dataset("test", self.test_data)

