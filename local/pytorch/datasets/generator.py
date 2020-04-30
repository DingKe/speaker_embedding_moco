# -*- coding: utf-8 -*-
import random


class BaseGenerator(object):
    """Base class for data generators
    Arguments:
        batch_size: batch size
        keep_batch_size: if keep batch size to batch_size
        shuffle: if to shuffle data
        batched_shuffle: if to shuffle data in batch
        seed: apply when [batched_]shuffle is True, for reproductability
        predict: if True, generators won't try to load target data, return None otherwise 
        preprocess_moethod: None/str/callable, 
                None: no preprocessing, it's the default case
                str: customized generator should overwrite interal_preprocess to parse 
                     the string and do preprocessing accordingly
                callable: callable object, use preprocess_method directly
        call Foo(BaseGenerator):
            def __init__(self, foo_arg1, foo_arg2, **kwargs):
                super(Foo, self).__init__(**kwargs)
                del kwargs
                self.__dict__.update(locals())
                self.__dict__.pop('self')
                
                # Do something with the args
                
            def __call__(self, **kwargs):
                while True:
                    yield None
                
        foo = Foo(foo_arg1, foo_arg2)
        dq = DataQueue(foo)
        queue = dp.get_queue()
        for i in range(10):
            print(queue.get())
    """
    def __init__(self,
                 batch_size=1,
                 keep_batch_size=False,
                 shuffle=False,
                 batched_shuffle=False,
                 seed=1111,
                 predict=False,
                 verbose=0,
                 rank=0,
                 world_size=1,
                 preprocess_method=None,
                 **kwargs):
        self.__dict__.update(kwargs)
        del kwargs

        self.__dict__.update(locals())
        self.__dict__.pop('self')

        self.random = random.Random()
        self.random.seed(self.seed)

    def __call__(self):
        raise Exception('__call__ is not overrided yet!')

    def get_generator(self):
        return self.__call__()

    def preprocess(self, raw):
        if self.preprocess_method is None:
            return raw
        elif callable(self.preprocess_method):
            return self.preprocess_method(raw)
        elif type(self.preprocess_method) is str:
            return self.internal_preprocess(raw)

    def internal_preprocess(self, raw):
        return raw
