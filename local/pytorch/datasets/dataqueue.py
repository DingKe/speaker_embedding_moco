# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import sys
import time
import types
import numpy as np

import queue
import threading
import torch
from torch.multiprocessing import Process, Queue, Event, Condition

from ..nnet.core import np2tensor, tensor2pin


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """

        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.

        credit@justheuristic: https://github.com/justheuristic/prefetch_generator
        """
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class Prefetcher(object):
    """
        Wrap a DataQuee object for prefetching data from the inter-process queue to a inter-thread queue
        (size of buffer_size). Same User Inteface as DataQueue -- use get() to get data infinitely.
        The benifit is two fold:
        1. We can transfer the data to GPU, thus no data copy when NN forward.
        2. The shared memory may be very limited (e.g. in the GPU docker circumstances).

       # Arguments
            data_queue: instance of DataQueue
            buffer_size: max size of thread queue
            pin_memory: if True, transfer Tensor to pinned memory
            timeout: if not None, wait data for at most this mush time, raise exception otherwise
            postprocess: if not None, invoke this on the data
            stream: optional cuda stream. If given, synchronize it before return data.
                    Useful for async (non_blocking) gpu copy
    """
    def __init__(self,
                 data_queue,
                 buffer_size=1,
                 pin_memory=False,
                 timeout=None,
                 postprocess=None,
                 stream=None):
        self.buffer_size = buffer_size
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.postprocess = postprocess
        self.stream = stream

        self.in_queue = data_queue
        self.out_queue = queue.Queue(self.buffer_size)
        self.cv = threading.Condition()

        # sginal event, unset by default
        self.terminate_sig = threading.Event()

        self.thread = threading.Thread(target=self.fetch,
                                       args=(self.in_queue, self.out_queue,
                                             self.terminate_sig, self.cv,
                                             self.pin_memory, self.timeout,
                                             self.postprocess, buffer_size))
        self.thread.daemon = True
        self.thread.start()

    def get(self):
        with self.cv:
            if not self.terminate_sig.is_set() and self.out_queue.qsize() == 0:
                self.cv.wait()

        if self.terminate_sig.is_set():
            raise Exception("prefetch thread terminated!")

        data = self.out_queue.get()

        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)

        with self.cv:
            self.cv.notify()

        return data

    @staticmethod
    def fetch(in_q, out_q, terminate_sig, cv, pin_memory, timeout, postprocess,
              max_qsize):
        while True:
            with cv:
                if not terminate_sig.is_set() and out_q.qsize() >= max_qsize:
                    cv.wait()

            if terminate_sig.is_set(): return

            try:
                data = in_q.get(timeout=timeout)
                if data is None: continue
                if postprocess is not None:
                    data = postprocess(data)
                if pin_memory:
                    data = tensor2pin(data)

                out_q.put(data)
                with cv:
                    cv.notify()
            except Exception as e:
                with cv:
                    terminate_sig.set()
                    cv.notify()
                print("In queue size:", in_q.qsize(), file=sys.stderr)
                print("Error Message", e, file=sys.stderr)

    def __del__(self):
        with self.cv:
            self.terminate_sig.set()
            self.cv.notify()
        self.thread.join()


class DataQueue(object):
    '''Queue for data prefetching
       DataQueue launch a subprocess to avoid python's GIL
       # Arguments
            generator: instance of generator which feeds data infinitely
            max_queue_size: maximum queue size
            nb_worker: control concurrency,
                       only take effect when do preprocessing
    '''
    def __init__(self, generator, max_queue_size=5, nb_worker=1):
        self.generator = generator
        self.nb_worker = nb_worker
        self.max_queue_size = max_queue_size

        self._queue = Queue()
        self._signal = Event()
        self._available_cv = Condition()
        self._full_cv = Condition()

        args = (generator, self._queue, self._signal, self._available_cv,
                self._full_cv, self.nb_worker, self.max_queue_size)
        self.working_process = Process(target=self.generator_process,
                                       args=args)
        self.working_process.daemon = True
        self.working_process.start()

    def get(self, timeout=None):
        with self._available_cv:
            if not self._signal.is_set() and self._queue.qsize() == 0:
                self._available_cv.wait()

        if self._signal.is_set():
            raise Exception("prefetch process terminated!")

        try:
            data = self._queue.get()
            with self._full_cv:
                self._full_cv.notify()
        except Exception as e:
            with self._full_cv:
                self._signal.set()
                self._full_cv.notify_all()
                raise e

        return data

    def qsize(self):
        return self._queue.qsize()

    def __del__(self):
        with self._full_cv:
            self._signal.set()
            self._full_cv.notify_all()
        #self.working_process.terminate()
        self.working_process.join()

    @staticmethod
    def generator_process(generator, queue, signal, available_cv, full_cv,
                          nb_worker, max_qsize):
        preprocess = generator.preprocess
        generator = BackgroundGenerator(generator())  # invoke call()

        # put data in the queue
        def enqueue_fn(generator, preprocess, queue, signal, available_cv,
                       full_cv, lock, max_qsize):
            while True:
                try:
                    with lock:
                        data = next(generator)
                    data = preprocess(data)

                    if not isinstance(data, types.GeneratorType):
                        data = [data]

                    for ele in data:
                        ele = np2tensor(ele)  # numpy array to pytorch's tensor
                        with full_cv:
                            while not signal.is_set(
                            ) and queue.qsize() >= max_qsize:
                                full_cv.wait()

                        if signal.is_set(): return

                        queue.put(ele)

                        with available_cv:
                            available_cv.notify()
                except Exception as e:
                    print("Error Message", e, file=sys.stderr)
                    with full_cv:
                        signal.set()
                        full_cv.notify_all()
                    with available_cv:
                        signal.set()
                        available_cv.notify_all()
                    raise Exception("generator thread went wrong!")

        # start threads
        lock = threading.Lock()
        args = (generator, preprocess, queue, signal, available_cv, full_cv,
                lock, max_qsize)
        generator_threads = [
            threading.Thread(target=enqueue_fn, args=args)
            for _ in range(nb_worker)
        ]

        for thread in generator_threads:
            thread.daemon = True
            thread.start()

        for thread in generator_threads:
            thread.join()
