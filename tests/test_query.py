import unittest, torch

from random import seed, sample

from torchvision import datasets
from torchvision.transforms import ToTensor

from torchql import Database, Query

class TestQuery(unittest.TestCase):
    def setUp(self):
        self.test_data = datasets.MNIST(
            root = 'data',
            train = False,                         
            transform = ToTensor(), 
            download = True,            
        )
        self.train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )

        self.db = Database("mnist")
        self.db.register_dataset(self.test_data, "test")
        self.db.register_dataset(self.train_data, "train")

    def test_join(self):
        doubled_label = Query('doubled_label', base='test').project(lambda *row: row[1] * 2)(self.db)
        joined = Query('join', base='test').join('doubled_label', key=lambda *row: row[0], fkey=lambda *row: row[0])(self.db)

        expected_join = [(x[0], x[1], x[1] * 2) for x in self.test_data]

        for expected, result in zip(expected_join, joined):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_union(self):
        first_half = [self.test_data[i] for i in range(0, len(self.test_data) // 2)]
        second_half = [self.test_data[i] for i in range(len(self.test_data) // 2, len(self.test_data))]

        self.db.register_dataset(first_half, "first_half")
        self.db.register_dataset(second_half, "second_half")
        
        unioned = Query('union', base='first_half').union('second_half')(self.db)

        for expected, result in zip(self.test_data, unioned):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_intersect(self):
        even = [self.test_data[i] for i in range(0, len(self.test_data), 2)]
        even_id_index = {i : i // 2 for i in range(0, len(self.test_data), 2)}

        self.db.register_dataset(even, "even")
        self.db.tables['even'].id_index = even_id_index

        intersected = Query('intersect', base='test').intersect('even')(self.db)

        for expected, result in zip(self.test_data, intersected):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_filter(self):
        filtered = Query('filter', base='test').filter(lambda *row: row[1] == 8)(self.db)
        expected_filter = [row for row in self.test_data if row[1] == 8]

        for expected, result in zip(expected_filter, filtered):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_project(self):
        projected = Query('project', base='test').project(lambda *row: (row[0], row[1] ** 2))(self.db)
        expected_project = [(x[0], x[1] ** 2) for x in self.test_data]

        for expected, result in zip(expected_project, projected):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_flatten(self):
        flattened = Query('flatten', base='test').flatten()(self.db)
        expected_flatten = [x for row in self.test_data for x in row]

        for expected, result in zip(expected_flatten, flattened):
            if torch.is_tensor(expected):
                self.assertTrue(torch.equal(expected, result))
            elif isinstance(expected, int):
                self.assertTrue(expected == result)
            else:
                self.fail()

    def test_order_by(self):
        ordered = Query('order', base='test').order_by(lambda *row: row[1])(self.db)
        expected_order = sorted(self.test_data, key=lambda row: row[1])

        for expected, result in zip(expected_order, ordered):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_group_by(self):
        grouped = Query('group', base='test').group_by(lambda *row: row[1])(self.db)
        grouped_data = sorted(grouped.rows, key=lambda *row: row[0])

        groups = {}

        for t in self.test_data:
            if t[1] not in groups:
                groups[t[1]] = []
            
            groups[t[1]].append(t)

        expected_groups = list(groups.items())
        expected_groups.sort(key=lambda x: x[0])

        for i in range(len(expected_groups)):
            for expected, result in zip(expected_groups[i][1], grouped_data[i][1]):
                self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_group_by_with_index(self):
        grouped = Query('group_with_index', base='test').group_by_with_index(lambda *row: row[0] % 10 + row[2])(self.db)
        grouped_data = sorted(grouped.rows, key=lambda *row: row[0])

        groups = {}

        for i in range(len(self.test_data)):
            if i % 10 + self.test_data[i][1] not in groups:
                groups[i % 10 + self.test_data[i][1]] = []
            
            groups[i % 10 + self.test_data[i][1]].append(self.test_data[i])

        expected_groups = list(groups.items())
        expected_groups.sort(key=lambda x: x[0])

        for i in range(len(expected_groups)):
            for expected, result in zip(expected_groups[i][1], grouped_data[i][1]):
                self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_head(self):
        head3 = Query('head', base='test').head(3)(self.db)
        expected_head = [self.test_data[i] for i in range(3)]

        for expected, result in zip(expected_head, head3):
            self.assertTrue(torch.equal(expected[0], result[0]) and expected[1] == result[1])

    def test_rebase(self):
        q = Query('name1', base='test').group_by(lambda *row: row[1])
        t_test = q(self.db)
        q = q.base('train')
        t_train = q(self.db)

        self.assertFalse(t_test == t_train)


    def test_base(self):
        t1 = Query('og', base='test').group_by(lambda *row: row[1])(self.db)
        t2 = Query('new').group_by(lambda *row: row[1]).base('test')(self.db)

        self.assertTrue(t1 == t2)

if __name__ == '__main__':
    unittest.main(buffer=True)