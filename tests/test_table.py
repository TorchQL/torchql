from torchql.table import Table
import unittest
from random import randint, seed, sample

class TableTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_one_col(self):
        integers = [i for i in range(10)]
        table = Table(integers)

        id_index = {i: i for i in integers}

        self.assertEqual(integers, table.rows)
        self.assertEqual(id_index, table.id_index)

    def test_one_col_set_id_index(self):
        integers = [i for i in range(10)]
        id_index = {i: chr(ord('a') + i) for i in integers}

        table = Table(integers, id_index)

        self.assertEqual(integers, table.rows)
        self.assertEqual(id_index, table.id_index)

    def test_tuple_table_creation(self):
        integers = tuple(i for i in range(10))
        table = Table(integers)

        id_index = {i: i for i in integers}

        self.assertEqual(list(integers), table.rows)
        self.assertEqual(id_index, table.id_index)

    def test_table_not_implemented(self):
        self.assertRaises(NotImplementedError, Table, 3)

    def test_dict_table_creation(self):
        alphabet = {i: chr(ord('a') + i) for i in range(26)}
        table = Table(alphabet)

        id_index = {i: i for i in range(26)}

        self.assertEqual(list(alphabet.values()), table.rows)
        self.assertEqual(id_index, table.id_index)

    def test_dataset_table_creation(self):
        table = Table("string")

        id_index = {i: i for i in range(len("string"))}

        self.assertEqual(['s', 't', 'r', 'i', 'n', 'g'], table.rows)
        self.assertEqual(id_index, table.id_index)

    def test_multi_columns_table(self):
        integers = [[2 * j + i for i in range(10)] for j in range(5)]
        table = Table(integers)

        id_index = {i: i for i in range(len(integers))}

        self.assertEqual(integers, table.rows)
        self.assertEqual(id_index, table.id_index)


class TableOperationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = [[randint(0, 20) for i in range(10)] for j in range(10)]
        self.table = Table(self.data)
    
    def test_join(self):
        aux_data = [[j for i in range(10)] for j in range(10)]
        aux_table = Table(aux_data)

        new_table = self.table.join(aux_table)
        expected = [tuple(self.data[i] + aux_data[i]) for i in range(len(self.data))]
        id_index = {(i, i): i for i in range(len(self.data))}

        self.assertEqual(expected, new_table.rows)
        self.assertEqual(id_index, new_table.id_index)

    def test_union(self):
        aux_data = [[j for i in range(10)] for j in range(10)]
        aux_table = Table(aux_data)

        new_table = self.table.union(aux_table)
        expected = self.data + aux_data
        id_index = {i % 10: i for i in range(len(expected))}

        self.assertEqual(expected, new_table.rows)
        self.assertEqual(id_index, new_table.id_index)

    def test_intersect(self):
        multiples2 = [[j for i in range(10)] for j in range(0, 20, 2)]
        multiples2_id_index = {i: i // 2 for i in range(0, 20, 2)}
        multiples2_table = Table(multiples2, multiples2_id_index)

        multiples3 = [[j for i in range(10)] for j in range(0, 20, 3)]
        multiples3_id_index = {i: i // 3 for i in range(0, 20, 3)}
        multiples3_table = Table(multiples3, multiples3_id_index)

        new_table = multiples2_table.intersect(multiples3_table)
        expected = [[j for i in range(10)] for j in range(0, 20, 6)]
        id_index = {i: i // 6 for i in range(0, 20, 6)}

        self.assertEqual(expected, new_table.rows)
        self.assertEqual(id_index, new_table.id_index)

    def test_filter(self):
        new_table = self.table.filter(lambda *x: x[0] > 10)
        expected = [(i, self.data[i]) for i in range(len(self.data)) if self.data[i][0] > 10]
        id_index = {expected[i][0]: i for i in range(len(expected))}
        expected = [e[1] for e in expected]

        self.assertEqual(expected, new_table.rows)
        self.assertEqual(id_index, new_table.id_index)

    def test_project(self):
        new_table = self.table.project(lambda *x: (x[3], x[6]))
        expected = [(x[3], x[6]) for x in self.data]
        id_index = {i: i for i in range(len(expected))}

        self.assertEqual(expected, new_table.rows)
        self.assertEqual(id_index, new_table.id_index)

    def test_unique(self):
        repeat_data = [tuple(i for k in range(10)) for i in range(5) for j in range(3)]
        repeat_table = Table(repeat_data)

        unique = repeat_table.unique()
        sort_unique = sorted(unique.rows, key=lambda x: x[0])

        expected = [tuple(i for j in range(10)) for i in range(5)]

        self.assertEqual(expected, sort_unique)

    def test_batch(self):
        batches = self.table.batch(2, False)
        expected = [self.data[i:i+2] for i in range(0, len(self.data), 2)]

        self.assertEqual(expected, batches.rows)

    def test_flatten(self):
        flat = self.table.flatten()
        expected = [x for row in self.data for x in row]
        id_index = { i : i for i in range(len(expected)) }

        self.assertEqual(expected, flat.rows)
        self.assertEqual(id_index, flat.id_index)

    def test_order_by(self):
        new_table = self.table.order_by(lambda *x: x[3])
        expected = sorted(self.data, key=lambda x: x[3])
        id_index = {i: expected.index(self.data[i]) for i in range(len(self.data))}

        self.assertEqual(expected, new_table.rows)
        self.assertEqual(id_index, new_table.id_index)

    def test_group_by(self):
        indexed_random_data = [(i, randint(0, 20)) for j in range(10) for i in range(5)]
        indexed_table = Table(indexed_random_data)

        grouped_table = indexed_table.group_by(key=lambda *x: x[0])

        groups = {}

        for t in indexed_random_data:
            if t[0] not in groups:
                groups[t[0]] = []
            
            groups[t[0]].append(t)

        # expected = list(groups.items())
        expected = [ (k, Table(v)) for k, v in groups.items() ]

        self.assertEqual(expected, grouped_table.rows)

    def test_group_by_with_index(self):
        indexed_random_data = [(i, randint(0, 20)) for j in range(10) for i in range(5)]
        indexed_table = Table(indexed_random_data)

        grouped_table = indexed_table.group_by(key=lambda *x: x[0])

        groups = {}

        for t in indexed_random_data:
            if t[0] not in groups:
                groups[t[0]] = []
            
            groups[t[0]].append(t)

        # expected = list(groups.items())
        expected = [ (k, Table(v)) for k, v in groups.items() ]

        self.assertEqual(expected, grouped_table.rows)

    def test_head(self):
        self.assertEqual(self.data[:3], self.table.head(3))

    def test_sample_many(self):
        seed(0)
        expected = sample(self.data, 5)
        
        seed(0)
        self.assertEqual(expected, self.table.sample_many(5))

    def test_sample(self):
        seed(0)
        expected = self.data[randint(0, len(self.data) - 1)]
        
        seed(0)
        self.assertEqual(expected, self.table.sample())
    
if __name__ == '__main__':
    unittest.main(buffer=True)