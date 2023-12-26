import unittest, torch, os

from torchvision import datasets
from torchvision.transforms import ToTensor

from torchql import Database, Query

class DatabaseTest(unittest.TestCase):
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

    def tearDown(self):
        if os.path.exists('data/'):
            for file in os.listdir('data/'):
                if file.endswith(".pt"):
                    os.remove(os.path.join('data', file))
    
    def test_register(self):
        self.database = Database("mnist_db")
        self.database.register_dataset(self.train_data, "train")
        self.database.register_dataset(self.test_data, "test")

        self.assertTrue('train' in self.database.tables)
        self.assertTrue('test' in self.database.tables)

        self.assertTrue(len(self.database.tables['train']) == len(self.train_data))
        self.assertTrue(len(self.database.tables['test']) == len(self.test_data))

    def test_load_store(self):
        db = Database("mnist")
        db.register_dataset(self.train_data, "train")
        db.register_dataset(self.test_data, "test")
        db.store('data/')
        
        saved_db = Database("saved_mnist")
        saved_db.load('data')

        self.assertTrue('train' in saved_db.tables)
        self.assertTrue('test' in saved_db.tables)

        for expected, result in zip(self.train_data, saved_db.tables['train']):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

        for expected, result in zip(self.test_data, saved_db.tables['test']):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_save_individual_table(self):
        misc = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        db = Database("mnist")
        db.register_dataset(self.train_data, "train")
        db.register_dataset(self.test_data, "test")
        db.register_dataset(misc, "misc")
        db.store('data/', ['test', 'misc'])
        
        self.assertTrue(os.path.exists('data/'))

        files = os.listdir('data/')

        self.assertFalse('train.pt' in files)
        self.assertTrue('test.pt' in files)
        self.assertTrue('misc.pt' in files)

    def test_save_table_invalid_name(self):
        db = Database("mnist")
        db.register_dataset(self.test_data, "test")
        db.store('data/', ['invalid'])
        
        self.assertTrue(os.path.exists('data/'))

        files = os.listdir('data/')

        self.assertFalse('invalid.pt' in files)

    def test_load_individual_tables(self):
        db = Database("mnist")
        db.register_dataset(self.train_data, "train")
        db.register_dataset(self.test_data, "test")
        db.store('data/')

        db = Database()
        db.load('data/', ['test', 'train'])
        
        self.assertTrue('train' in db.tables)
        self.assertTrue('test' in db.tables)
        self.assertFalse('misc' in db.tables)

        for expected, result in zip(self.train_data, db.tables['train']):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

        for expected, result in zip(self.test_data, db.tables['test']):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_load_table_invalid_name(self):
        db = Database()
        db.load('data/', ['invalid'])
        
        self.assertEqual(0, len(db.tables))
        self.assertFalse('invalid' in db.tables)

if __name__ == '__main__':
    unittest.main(buffer=True)