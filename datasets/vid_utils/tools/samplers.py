from collections import defaultdict
import numpy as np
import copy
import random

# import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        insnum_trac__list = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            ins_num_trac = []
            for idx in idxs:
                ins_num_trac.append(idx)
                if len(ins_num_trac) == self.num_instances:
                    insnum_trac__list.append(ins_num_trac)
                    ins_num_trac = []

        random.shuffle(insnum_trac__list)

        ret = []
        for ins_num_trac in insnum_trac__list:
            ret.extend(ins_num_trac)

        return iter(ret)

    def __len__(self):
        return self.length