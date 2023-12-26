from __future__ import annotations
import random
from typing import Any, Callable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .utils import get_iterable
from functools import partialmethod
import copy
from multiprocessing.pool import ThreadPool
import torch

# Uncomment to disable tqdm
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# def nop(it, *a, **k):
#     return it

# tqdm = nop

def is_dataset(object):
    return (hasattr(object, "__getitem__") and callable(getattr(object, "__getitem__"))
            and hasattr(object, "__len__") and callable(getattr(object, "__len__")))

class Table(Dataset):
    """
    A Table is a collection of samples.
    It is specifically designed to be queried from within a database.
    It is a subclass of torch.utils.data.Dataset, so it can be used as a dataset.
    """
    def __init__(self, samples, id_index: dict=None, transform=None, disable=False) -> None:
        super().__init__()
        self.transform = transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if id_index is None:
            self.id_index = {}
            build_idx = True
        else:
            self.id_index = id_index
            build_idx = False

        # building the index
        if isinstance(samples, (list, tuple)):
            self.rows = list(samples)
            if build_idx:
                for i in range(len(self.rows)):
                    self.id_index[i] = i
        elif isinstance(samples, dict):
            self.rows = []
            self.id_index = {}
            for key_idx, key in enumerate(samples):
                self.rows.append(samples[key])
                self.id_index[key] = key_idx
                assert self.rows[key_idx] == samples[key]
        elif is_dataset(samples):
            self.rows = []
            for row_idx, row in tqdm(enumerate(samples), total=len(samples), desc="Building table", disable=disable):
                self.rows.append(row)
                if build_idx:
                    self.id_index[row_idx] = row_idx
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.rows)
    
    def __iter__(self):
        return iter(self.rows)
    
    def __getitem__(self, index):
        return get_iterable(self.transform(self.rows[index]) if self.transform is not None else self.rows[index])
    
    def __contains__(self, item):
        return item in self.rows
    
    def __eq__(self, t: Table) -> bool:
        return self.id_index == t.id_index and self.rows == t.rows

    def transform(self, transform) -> Table:
        """
        Register a PyTorch Transform to apply to this table.

        Args:
            transform (Callable): The PyTorch transform to apply to the records of this table.

        Returns:
            A new table with the transformed records.
        """
        return Table(self.rows, self.id_index, transform=transform)
    
    def join(self, table: Dataset, key=None, fkey=None, batch_size=0, disable=False) -> Table:
        """
        Join this table with another table.
        Records are joined if the key function returns the same value for both tables.
        If no key function is provided, the index is used.

        Args:
            table (Dataset): The table to join with.

            key (Callable, optional): The key function to use for this table. Defaults to None.
                Must take a set of columns from a row as input and return a hashable value that serves as a key.

            fkey (Callable, optional): The foreign key function to use for the other table. Defaults to None.
                Must take a set of columns from a row as input and return a hashable value that serves as a key.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the joined records.
        """
        l_indices = {}
    
        for id, index in tqdm(self.id_index.items(), desc="Building index for left table", disable=disable):
            if key is None:
                l_indices[id] = {id, }
            else:
                row = self[index]
                if id not in l_indices:
                    l_indices[id] = set()
                l_indices[id].add(key(id, *row))
        
        r_indices = {}

        def table_iter():
            if isinstance(table, Table):
                for id, index in tqdm(table.id_index.items(), desc="Building index for right table", disable=disable):
                    yield id, table[index]
            else:
                for id, row in tqdm(enumerate(table), desc="Building index for right table", disable=disable):
                    yield id, get_iterable(row)

        def get_table_item(id):
            if isinstance(table, Table):
                return table[table.id_index[id]]
            else:
                return table[id]

        for id, row in table_iter():
            if fkey is None:
                r_indices[id] = {id, }
            else:
                fk_val = fkey(id, *row)
                if fk_val not in r_indices:
                    r_indices[fk_val] = set()
                r_indices[fk_val].add(id)

        joined_rows = []
        joined_indices = {}

        for lid, lvals in tqdm(l_indices.items(), desc="Joining", disable=disable):
            for lval in lvals:
                if lval in r_indices:
                    for rid in r_indices[lval]:
                        l_entry = self[self.id_index[lid]]
                        l_entry = get_iterable(l_entry)
                        r_entry = get_table_item(rid)
                        r_entry = get_iterable(r_entry)
                        joined_rows.append((*l_entry, *r_entry))
                        joined_indices[(lid, rid)] = len(joined_rows) - 1

        return Table(joined_rows, joined_indices, self.transform)
    
    def union(self, table: Dataset, batch_size=0, disable=False) -> Table:
        """
        Union this table with another table.
        Records of the other table will be added to the bottom of the table.

        Args:
            table (Dataset): The table to union with.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the combined records.
        """
        unioned_rows = []
        unioned_indices = {}

        if batch_size <= 0:
            for id, index in tqdm(self.id_index.items(), desc="Getting records of self table", disable=disable):
                row = self[index]
                row = get_iterable(row)
                
                unioned_rows.append(row)
                unioned_indices[id] = len(unioned_rows) - 1
        else:
            for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Getting records of self table", disable=disable):
                unioned_rows.extend(row_batch)

        def table_iter():
            if isinstance(table, Table):
                for id, index in tqdm(table.id_index.items(), desc="Getting records of other table"):
                    yield id, table[index]
            else:
                for id, row in tqdm(enumerate(table), desc="Getting records of other table"):
                    yield id, get_iterable(row)
        
        for id, row in table_iter():
            if isinstance(table, Table):
                row = get_iterable(row)

            unioned_rows.append(row)
            unioned_indices[id] = len(unioned_rows) - 1

        return Table(unioned_rows, unioned_indices, self.transform)
    
    def intersect(self, table: Dataset, batch_size=0, disable=False) -> Table:
        """
        Intersect this table with another table.
        Common records between this and other table will be used to create a new table.
        Common records are identified by the id of the row.
        The columns of the other table will be used in the new table.

        Args:
            table (Dataset): The table to intersect with.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the common records.
        """
        self_rows = []
        self_indices = {}

        if batch_size <= 0:
            for id, index in tqdm(self.id_index.items(), desc="Getting records of self table", disable=disable):
                row = self[index]
                row = get_iterable(row)
                
                self_rows.append(row)
                self_indices[id] = len(self_rows) - 1
        else:
            for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Getting records of self table", disable=disable):
                self_rows.extend(row_batch)

        def table_iter():
            if isinstance(table, Table):
                for id, index in tqdm(table.id_index.items(), desc="Getting records of other table", disable=disable):
                    yield id, table[index]
            else:
                for id, row in tqdm(enumerate(table), desc="Getting records of other table", disable=disable):
                    yield id, get_iterable(row)
        
        intersect_rows = []
        intersect_indices = {}

        for id, row in table_iter():
            if id in self_indices:

                if isinstance(table, Table):
                    row = get_iterable(row)

                intersect_rows.append(row)
                intersect_indices[id] = len(intersect_rows) - 1

        return Table(intersect_rows, intersect_indices, self.transform)
    
    def filter(self, cond: Callable[..., bool], batch_size=0, disable=False) -> Table:
        """
        Filter this table by a condition.

        Args:
            cond (Callable): The condition to filter by.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.
        
        Returns:
            A new table with the filtered records.
        """
        filtered_rows = []
        filtered_indices = {}
        if batch_size <= 0:
            for id, index in tqdm(self.id_index.items(), desc="Filtering", disable=disable):
                row = self[index]
                row = get_iterable(row)
                if cond(*row):
                    filtered_rows.append(row)
                    filtered_indices[id] = len(filtered_rows) - 1
        else:
            for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Filtering", disable=disable):
                condArr = cond(*row_batch)
                for i in range(len(condArr)):
                    if condArr[i] :
                        filtered_rows.append(row_batch[i])

        return Table(filtered_rows, filtered_indices, self.transform)

    def project(self, cols: Callable[..., list], batch_size=0, disable=False) -> Table:
        """
        Select or perform an operation on the columns of this table.

        Args:
            cols (Callable): A function that takes the columns of this table as arguments and returns a list of the
            projected columns.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.
        
        Returns:
            A new table with the projected columns.
        """
        projected_rows = []
        projected_indices = {}

        if batch_size <= 0:
            for id, index in tqdm(self.id_index.items(), desc="Projecting", disable=disable):
                row = self[index]
                row = get_iterable(row)
                projected_rows.append(cols(*row))
                projected_indices[id] = len(projected_rows) - 1
            
            return Table(projected_rows, projected_indices, transform=self.transform)
        else:
            for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Projecting", disable=disable):
                res = cols(*row_batch)
                projected_rows.extend(res)

            return Table(projected_rows, transform=self.transform)
    
    def unique(self, batch_size=0, disable=False) -> Table:
        """
        Select the unique records of this table.

        Args:
            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the unique records.
        """
        return Table(list(set(self.rows)), transform=self.transform)
    
    def batch(self, size, shuffle, batch_size=0, disable=False) -> Table:
        """
        Batch this table.

        Args:
            size (int): The size of the batches.

            random (bool): Whether to shuffle the records before batching.

        Returns:
            A new table with the batches.
        """
        if shuffle:
            rows = list(self.rows)
            random.shuffle(rows)
        else:
            rows = list(self.rows)
        batches = [rows[i:i+size] for i in range(0, len(rows), size)]
        return Table(batches, transform=self.transform)
    
    def flatten(self, batch_size=0, disable=False) -> Table:
        """
        Flatten this table. If the records of this table are lists, the records of the new table will be the elements of the
        lists.

        Args:
            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with flattened records
        """
        flattened_rows = []
        flattened_indices = {}
        for id, index in tqdm(self.id_index.items(), desc="Flattening", disable=disable):
            rowlist = self.rows[index]
            for subidx, row in enumerate(rowlist):
                flattened_rows.append(row)
                flattened_indices[len(flattened_rows) - 1] = len(flattened_rows) - 1
        return Table(flattened_rows, flattened_indices, transform=self.transform)

    def order_by(self, key: Callable[..., Any], reverse=False, batch_size=0, disable=False) -> Table:
        """
        Order this table by a key.

        Args:
            key (Callable): The key to order by.

            reverse (bool, optional): Whether to reverse the order. Defaults to False.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the ordered records.
        """
        ordered_rows = []
        ordered_indices = {}
        if batch_size <= 0 :
            for id, index in tqdm(self.id_index.items(), desc="Ordering", disable=disable):
                row = self[index]
                ordered_rows.append((key(*row), id, row))
        # else:
        #     for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Projecting", disable=disable):
        #         ordered_rows.extend(row_batch)

        ordered_rows.sort(key=lambda x: x[0], reverse=reverse)
        for idx, (key, id, row) in enumerate(ordered_rows):
            ordered_indices[id] = idx
        return Table([row for _, _, row in ordered_rows], ordered_indices, transform=self.transform)

    def group_by(self, key: Callable[..., Any], batch_size=0, disable=False) -> Table:
        """
        Group this table by a key.
        Records are grouped by the key function. The key function should return a hashable value. The records of the new table
        will be tuples of the key and a list of the records that have that key.

        Args:
            key (Callable): The key to group by.
                Must return a hashable value.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the grouped records.
        """
        grouped_rows = []
        groups = {}
        if batch_size <= 0:
            for idx, index in tqdm(self.id_index.items(), desc="Grouping", disable=disable):
                row = self[index]
                grouping_key = key(*row)
                if grouping_key not in groups:
                    groups[grouping_key] = []
                groups[grouping_key].append(row)
        else:
           for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Filtering", disable=disable):
                # key1:{idx_11,idx_12...},key2:..
                grouping_keys = key(*row_batch)
                for grouping_key,idx_arr in grouping_keys:
                    if grouping_key not in groups:
                        groups[grouping_key] = []
                    groups[grouping_key].extend(row_batch[idx_arr])                

        for grouping_key, group_rows in groups.items():
            grouped_rows.append((grouping_key, Table(group_rows)))

        return Table(grouped_rows, transform=self.transform)

    def group_by_with_index(self, key: Callable[..., Any], batch_size=0, disable=False) -> Table:
        """
        Group this table by a key.
        Records are grouped by the key function. The key function should return a hashable value. The records of the new table
        will be tuples of the key and a list of the records that have that key.

        Args:
            key (Callable): The key to group by.
                Must return a hashable value.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the grouped records.
        """
        grouped_rows = []
        groups = {}
        if batch_size <= 0:
            for idx, index in tqdm(self.id_index.items(), desc="Grouping", disable=disable):
                row = self[index]
                grouping_key = key(idx, *row)
                if grouping_key not in groups:
                    groups[grouping_key] = []
                groups[grouping_key].append(row)
        else:
           for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Filtering", disable=disable):
                # key1:{idx_11,idx_12...},key2:..
                grouping_keys = key(idx, *row_batch)
                for grouping_key,idx_arr in grouping_keys:
                    if grouping_key not in groups:
                        groups[grouping_key] = []
                    groups[grouping_key].extend(row_batch[idx_arr])

        for grouping_key, group_rows in groups.items():
            grouped_rows.append((grouping_key, Table(group_rows)))

        return Table(grouped_rows, transform=self.transform)
    
    def reduce(self, reduction: Callable[..., Any], batch_size=0, disable=False) -> Table:
        """
        Reduce the records of this table using a reduction function. This function operates over all the records of the table
        as opposed to each row individually.

        Args:
            reduction (Callable): The reduction function that takes in the records of the table.

            batch_size (int): The batch size to enable batch-processing of the query. Note that a batch size >= 1 assumes
                your supplied functions run on batches of records as opposed to a single record.

            disable (boolean): A flag that disables progress bars if set to True.

        Returns:
            A new table with the reduced records.
        """

        return reduction(Table(self.rows, self.id_index, transform=self.transform))
    
    def group_reduce(self, key: Callable[..., Any], reduction: Callable[..., Any], batch_size=0, disable=False) -> Table:
        """
        Group this table by a key and reduce the records of each group using a reduction function.

        Args:
            key (Callable): The key to group by.
                Must return a hashable value.

            reduction (Callable): The reduction function that takes in the records of each group.

        Returns:   
            A new table with the grouped and reduced records.
        """

        return self.group_by(key, disable=disable).project(lambda key, group: (key, reduction(group)), disable=disable, batch_size=batch_size)

    
    def head(self, n=10, print_id=False):
        """
        Get the first n records of this table.

        Args:
            n (int, optional): The number of records to get. Defaults to 10.

            print_id (bool, optional): Whether to print the id of each row. Defaults to False.

        Returns:
            A table with the first n records of this table.
        """
        i = 0
        l = []
        for id, index in self.id_index.items():
            row = self[index]
            if print_id:
                l.append((id, row))
            else:
                l.append(row)

            i += 1
            if i == n:
                break

        return l

    def sample_many(self, n=10, print_id=False):
        """
        Get n random records of this table.

        Args:
            n (int, optional): The number of records to get. Defaults to 10.

            print_id (bool, optional): Whether to print the id of each row. Defaults to False.

        Returns:
            A list of n random records of this table.
        """
        l = []
        for id, index in random.sample(list(self.id_index.items()), n):
            row = self[index]
            if print_id:
                l.append((id, row))
            else:
                l.append(row)
        return l
    
    def sample(self, print_id=False):
        """
        Get a random row of this table.

        Args:
            print_id (bool, optional): Whether to print the id of the row. Defaults to False.

        Returns:
            A random row of this table.
        """
        id, index = random.choice(list(self.id_index.items()))
        row = self[index]
        if print_id:
            return (id, row)
        else:
            return row
