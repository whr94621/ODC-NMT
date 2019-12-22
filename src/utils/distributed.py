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

import os
import pickle
import uuid

import horovod.torch as hvd
import torch
from horovod.torch import mpi_ops
from horovod.torch.compression import Compression

__all__ = [
    'synchronize_all_processes',
    'all_gather_with_shared_fs',
    'all_gather',
    'broadcast',
    'all_reduce',
    'DistributedOptimizer'
]


def gen_random_name():
    """Return a random name for temp file"""
    return uuid.UUID(bytes=os.urandom(16), version=4).hex


class SharedFSTransferProtocol(object):
    """
    Protocol for transfering data between processes by shared filesystem.

    This is useful when you want to transfer some relative big data.
    """

    def __init__(self, prefix="/tmp", name=None):

        self.prefix = prefix

        if name is None:
            name = gen_random_name()

        self.name = name

        self.path = None

    def __getstate__(self):

        return {"prefix": self.prefix, "name": self.name, "path": self.path}

    def __setstate__(self, state):

        self.prefix = state['prefix']
        self.name = state['name']
        self.path = state['path']

    def read(self):

        if self.path is None:
            raise ValueError

        with open(self.path, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def _write(self, obj):

        self.path = os.path.join(self.prefix, self.name) + ".pkl"

        with open(self.path, "wb") as f:
            pickle.dump(obj, f)

    def close(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            # file has been removed by another process
            pass

    @classmethod
    def write(cls, obj, shared_fs_root="/tmp"):

        protoc = cls(prefix=shared_fs_root)
        protoc._write(obj)

        return protoc


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def synchronize_all_processes():
    """Synchronize all processes by reducing a null tensor"""
    null_tensor = torch.zeros(1)

    _ = hvd.allreduce(null_tensor)


def all_gather_with_shared_fs(data, shared_fs_root="/tmp"):
    tmp_protoc = SharedFSTransferProtocol.write(data, shared_fs_root=shared_fs_root)

    gathered_tmp_protoc = all_gather(tmp_protoc)

    gathered_data = [protoc.read() for protoc in gathered_tmp_protoc]

    synchronize_all_processes()

    if hvd.rank() == 0:
        for protoc in gathered_tmp_protoc:
            protoc.close()

    return gathered_data


def all_gather(data, max_size=65000):
    """ Gathers arbitrary data from all nodes into a list.

    This function is heavily borrowed from fairseq (https://github.com/pytorch/fairseq)

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """

    world_size = hvd.size()

    buffer_size = max_size
    if not hasattr(all_gather, '_buffer') or \
            all_gather._buffer.numel() < buffer_size:
        all_gather._buffer = torch.cuda.ByteTensor(buffer_size)

    buffer = all_gather._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))

    buffer_rank = buffer
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size + 2] = torch.ByteTensor(list(enc))

    buffer_gathered = hvd.allgather(buffer)

    result = []
    for i in range(world_size):
        out_buffer = buffer_gathered[i * max_size: (i + 1) * max_size]
        size = (255 * item(out_buffer[0])) + item(out_buffer[1])
        if size > 0:
            result.append(
                pickle.loads(bytes(out_buffer[2:size + 2].tolist()))
            )
    return result


def all_reduce(data, max_size=65000, average=False):
    all_gathered_data = all_gather(data=data, max_size=max_size)

    result = sum(all_gathered_data)

    if not average:
        return result
    else:
        return result / all(all_gathered_data)


def broadcast(data, root_rank, max_size=65000):
    # 1. allocate buffer
    buffer_size = max_size
    if not hasattr(broadcast, '_buffer') or \
            broadcast._buffer.numel() < buffer_size:
        broadcast._buffer = torch.cuda.ByteTensor(buffer_size)

    buffer = broadcast._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))

    buffer_broadcasted = buffer
    buffer_broadcasted[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_broadcasted[1] = enc_size % 255
    buffer_broadcasted[2:enc_size + 2] = torch.ByteTensor(list(enc))

    hvd.broadcast(buffer_broadcasted, root_rank=root_rank)

    size = (255 * buffer_broadcasted[0]) + item(buffer_broadcasted[1])
    obj = pickle.loads(bytes(buffer_broadcasted[2:size + 2].tolist()))

    return obj


# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Below is a modified version of DistributedOptimizer

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression,
                 backward_passes_per_step=1):
        super(self.__class__, self).__init__(params)
        self._compression = compression

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        self._parameter_names = {v: k for k, v
                                 in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        if mpi_ops.size() > 1:
            self._register_hooks()

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _allreduce_grad(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad.data
        tensor_compressed, ctx = self._compression.compress(tensor)
        # We use sum here and manually rescaled gradient by passing a denominator to optimizer.
        handle = mpi_ops.allreduce_async_(tensor_compressed, average=False, name=name)
        return handle, ctx

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._allreduce_grad(p)
            self._handles[p] = (handle, ctx)

        return hook

    def synchronize(self):
        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._allreduce_grad(p)
                self._handles[p] = (handle, ctx)
        for p, (handle, _) in self._handles.items():
            output = mpi_ops.synchronize(handle)
            self._allreduce_delay[p] = self.backward_passes_per_step
            p.grad.data.set_(self._compression.decompress(output, ctx))
        self._handles.clear()

    def step(self, closure=None):
        return super(self.__class__, self).step(closure)


def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.
    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compression, backward_passes_per_step)
