# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
import random
from itertools import count
from typing import Iterable, Generator
from typing import List

from src.utils.common_utils import GlobalNames
from .dataset import Record, zip_records

__all__ = [
    'DataIterator'
]

random.seed(GlobalNames.SEED)


def argsort(seq: List):
    return [p[0] for p in sorted([(i, e) for i, e in enumerate(seq)], key=lambda p: p[1])]


class Batch(object):
    """
    'Batch' is a list of 'Record's which can coalesce into one batch.

    'content' is a list of records which will be packed into one batch
    """

    def __init__(self, *records):

        self.content = list(records)

    def unpack(self):
        """ Unpack a 'Batch' instance into batched data.

        records in a batch will be split into several list according to the number of
        fields. For example, if a batch has three records R1, R2, R3. Ri has two fields, the
        the value of which are [a, b], then the result of unpack will be two lists, i.e.
        [a1, a2, a3], [b1, b2, b3]
        """
        n_fields = self.content[0].n_fields  # all the records must have the same field

        outs = tuple([r.fields[ii] for r in self.content] for ii in range(n_fields))

        if n_fields == 1:
            return outs[0]
        else:
            return outs

    @classmethod
    def pack(cls, *records: Record) -> 'Batch':
        """
        Pack a list of records into a batch.
        """

        return cls(*records)


def fill_buffer(data_iter, buffer_size):
    """
    Initialize a buffer from a iterator
    """

    records = list(itertools.islice(data_iter, buffer_size))

    return records


def numbering_records_iterator(record_iter: Iterable[Record]):
    """Numbering iterator from dataset.
    """
    for ii in count():
        try:
            record = next(record_iter)
        except StopIteration:
            break

        yield zip_records(Record(ii, size=-float('inf')), record)


def shuffle_iterator(iterator: Iterable[Record], buffer_size) -> Generator[Record, None, None]:
    buffer = fill_buffer(iterator, buffer_size=buffer_size)
    random.shuffle(buffer)

    for item in iterator:
        idx = random.randint(0, buffer_size - 1)

        yield buffer[idx]

        buffer[idx] = item

    for item in buffer:
        yield item


def split_shards_iterator(iterator: Iterable[Record], number_shards, n_shard) -> Generator[Record, None, None]:
    for item in itertools.islice(iterator, n_shard, None, number_shards):
        yield item


def batching_iterator(iterator: Iterable[Record], batch_size, batching_key,
                      use_bucket=False, buffer_size=10000) -> Generator[Batch, None, None]:
    buffer = []
    batch = []  # buffer for building a batch, save the indices of records in the buffer

    while True:
        # allocate a buffer for bucketing, merge unbatched records from last iteration
        # new buffer is composed of residual elements in previous batch
        # and newly allocated buffer

        buffer_inc = fill_buffer(iterator, buffer_size=buffer_size - len(batch))

        if len(buffer_inc) == 0:
            break
        else:
            buffer = [buffer[idx] for idx in batch] + buffer_inc
            del buffer_inc
            batch = []

        num_samples = 0
        max_token_per_batch = 0

        # If use bucketing
        # sort records in buffer by size
        if use_bucket:
            sorted_indices = argsort([record.size for record in buffer])
        else:
            sorted_indices = list(range(len(buffer)))

        # batching records in the buffer
        batch_queue = []
        for idx in sorted_indices:

            batch.append(idx)
            num_samples += 1

            if batching_key == "samples":
                if num_samples >= batch_size:
                    batch_queue.append(batch[:])
                    num_samples = 0
                    batch = []
            else:
                max_token_per_batch = max(max_token_per_batch, buffer[idx].size)

                if max_token_per_batch * num_samples >= batch_size:
                    batch_queue.append(batch[:])
                    num_samples = 0
                    max_token_per_batch = 0
                    batch = []

        # shuffle the order of batches
        random.shuffle(batch_queue)

        for _batch in batch_queue:
            yield Batch.pack(*[buffer[idx] for idx in _batch])

        batch_queue = []

    if len(batch) > 0:
        yield Batch.pack(*[buffer[idx] for idx in batch])


class DataIterator(object):
    """
    ```DataIterator``` defines the way to group your data into a batch. You can choose the way to batchify your data.
    In current implementation, we only provide "samples" and "tokens", which are the two main methods in machine
    translation.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 buffer_size=10000,
                 use_bucket=True,
                 batching_func="samples",
                 numbering=False,
                 shuffle=False,
                 world_size=1,
                 rank=0
                 ):

        """ Build data iterator given a dataset

        Args:
            dataset: An Dataset Object
            batch_size: Integer. Size of a batch. When batching_key is "samples", it represents the
                the number of samples. When batching_key is "tokens", it represents the tokens in a batch.
            use_bucket: Boolean value. Whether to use bucket.
            2batching_key: Criterion to allocate a batch. Can only be "samples" or "tokens"
            buffer_size: How many samples can be stored in memory for buffer. There are two places which need buffer--
                shuffle_iterator and bucket_iterator, so the actual memory consumption would be two times if you enable
                this two iterators at the same time.
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Batching Key
        #
        # We have two kinds of batching key, ```tokens``` and ```samples```.
        # For tokens, we allocate a batch according to the number of tokens in it. For example,
        # in machine translation, if we use "tokens" as the key and set the batch_size as 4096,
        # we allocate a batch when the number of tokens at source or target side reach 4096.
        # For samples, we allocate a batch according to the number of samples in it. In machine
        # translation, 50 batch size with "samples" as key means 50 bi-text sentences.

        if batching_func not in {"samples", "tokens"}:
            raise ValueError("Unknown batching key {0}".format(batching_func))
        self._batching_key = batching_func

        # buffer size for bucketing
        # buffer size is the max number of batches in a buffer
        # if batching key is 'samples', buffer size is 100 times of batch size,
        # else we suppose that their are 50 tokens in one sample and then estimate
        # the number of samples in one batch as self.batch_size // 50

        self.buffer_size = buffer_size
        self.use_bucket = use_bucket
        self.numbering = numbering

        # For distributed learning
        self.world_size = world_size
        self.rank = rank

        self.reset()

    def __len__(self):
        return len(self.dataset)

    def reset(self):

        self.buffer = []

        # 1. build data_iterator from dataset
        data_iter = self.dataset.read()

        # 2. numbering (optional)
        if self.numbering:
            data_iter = numbering_records_iterator(data_iter)

        # 3. distributed(optional)
        if self.world_size > 1:
            data_iter = split_shards_iterator(data_iter, number_shards=self.world_size, n_shard=self.rank)

        # 4. shuffle (optional)
        if self.shuffle:
            data_iter = shuffle_iterator(data_iter, buffer_size=self.buffer_size)

        # 5. bucketing (optional) & batching

        data_iter = batching_iterator(data_iter, batch_size=self.batch_size, batching_key=self._batching_key,
                                      use_bucket=self.use_bucket, buffer_size=self.buffer_size)

        self.data_iter = data_iter

    def build_generator(self, batch_size=None):

        while True:

            # Accumulated batches until reach the batch_size
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.reset()
                break

            yield batch.unpack()
