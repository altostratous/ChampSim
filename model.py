import math
from abc import ABC, abstractmethod
from collections import defaultdict
import os

import torch
import torch.nn as nn


class CacheSimulator(object):

    def __init__(self, sets, ways, block_size) -> None:
        super().__init__()
        self.ways = ways
        self.way_shift = int(math.log2(ways))
        self.sets = sets
        self.block_size = block_size
        self.block_shift = int(math.log2(block_size))
        self.storage = defaultdict(list)
        self.label_storage = defaultdict(list)

    def parse_address(self, address):
        block = address >> self.block_shift
        way = block % self.ways
        tag = block >> self.way_shift
        return way, tag

    def load(self, address, label=None):
        way, tag = self.parse_address(address)
        hit, l = self.check(address)
        if not hit:
            self.storage[way].append(tag)
            self.label_storage[way].append(label)
            if len(self.storage[way]) > self.sets:
                self.storage[way].pop(0)
                self.label_storage[way].pop(0)
        return hit, l

    def check(self, address):
        way, tag = self.parse_address(address)
        if tag in self.storage[way]:
            return True, self.label_storage[way][self.storage[way].index(tag)]
        else:
            return False, None


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 8, 8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 8, 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8 * 50, 64),
        )
        self.softmax = nn.Softmax()

    def forward(self, input):
        return self.softmax(self.decoder(self.feature(input).view(-1, 50 * 8)))


class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass


class NextLineModel(MLPrefetchModel):

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for NextLineModel')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for NextLineModel')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training NextLineModel')

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for NextLineModel')
        prefetches = []
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # Prefetch the next two blocks
            prefetches.append((instr_id, ((load_addr >> 6) + 5) << 6))
            prefetches.append((instr_id, ((load_addr >> 6) + 10) << 6))

        return prefetches


class BestOffset(MLPrefetchModel):
    offsets = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14,
               -14, 15, -15, 16, -16, 18, -18, 20, -20, 24, -24, 30, -30, 32, -32, 36, -36, 40, -40]
    scores = [0 for _ in range(len(offsets))]
    round = 0
    best_index = 0
    second_best_index = 0
    best_index_score = 0
    temp_best_index = 0
    score_scale = eval(os.environ.get('BO_SCORE_SCALE', '1'))
    bad_score = int(10 * score_scale)
    low_score = int(20 * score_scale)
    max_score = int(31 * score_scale)
    max_round = int(100 * score_scale)
    llc = CacheSimulator(16, 2048, 64)
    rrl = {}
    rrr = {}
    dq = []
    acc = []
    acc_alt = []
    active_offsets = set()
    p = 0
    memory_latency = 200
    rr_latency = 60
    fuzzy = eval(os.environ.get('FUZZY_BO', 'False'))

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for BestOffset')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for BestOffset')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training BestOffset')

    def rr_hash(self, address):
        return ((address >> 6) + address) % 64

    def rr_add(self, cycles, address):
        self.dq.append((cycles, address))

    def rr_add_immediate(self, address, side='l'):
        if side == 'l':
            self.rrl[self.rr_hash(address)] = address
        elif side == 'r':
            self.rrr[self.rr_hash(address)] = address
        else:
            assert False

    def rr_pop(self, current_cycles):
        while self.dq:
            cycles, address = self.dq[0]
            if cycles < current_cycles - self.rr_latency:
                self.rr_add_immediate(address, side='r')
                self.dq.pop(0)
            else:
                break

    def rr_hit(self, address):
        return self.rrr.get(self.rr_hash(address)) == address or self.rrl.get(self.rr_hash(address)) == address

    def reset_bo(self):
        self.temp_best_index = -1
        self.scores = [0 for _ in range(len(self.offsets))]
        self.p = 0
        self.round = 0
        # self.acc.clear()
        # self.acc_alt.clear()

    def train_bo(self, address):
        testoffset = self.offsets[self.p]
        testlineaddr = address - testoffset

        if address >> 6 == testlineaddr >> 6 and self.rr_hit(testlineaddr):
            self.scores[self.p] += 1
            if self.scores[self.p] >= self.scores[self.temp_best_index]:
                self.temp_best_index = self.p

        if self.p == len(self.scores) - 1:
            self.round += 1
            if self.scores[self.temp_best_index] == self.max_score or self.round == self.max_round:
                self.best_index = self.temp_best_index if self.temp_best_index != -1 else 1
                self.second_best_index = sorted([(s, i) for i, s in enumerate(self.scores)])[-2][1]
                self.best_index_score = self.scores[self.best_index]
                if self.best_index_score <= self.bad_score:
                    self.best_index = -1
                self.active_offsets.add(self.best_index)
                self.reset_bo()
                return
        self.p += 1
        self.p %= len(self.scores)

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for BestOffset')
        prefetches = []
        prefetch_requests = []
        percent = len(data) // 100
        for i, (instr_id, cycle_count, load_addr, load_ip, llc_hit) in enumerate(data):
            # Prefetch the next two blocks
            hit, prefetched = self.llc.load(load_addr, False)
            while prefetch_requests and prefetch_requests[0][0] + self.memory_latency < cycle_count:
                fill_addr = prefetch_requests[0][1]
                h, p = self.llc.load(fill_addr, True)
                if not h:
                    if self.best_index == -1:
                        fill_line_addr = fill_addr >> 6
                        if self.best_index != -1:
                            offset = self.offsets[self.best_index]
                        else:
                            offset = 0
                        self.rr_add_immediate(fill_line_addr - offset)
                prefetch_requests.pop(0)
            self.rr_pop(cycle_count)
            if not hit or prefetched:
                line_addr = (load_addr >> 6)
                self.train_bo(line_addr)
                self.rr_add(cycle_count, line_addr)
                if self.best_index != -1 and self.best_index_score > self.low_score:
                    addr_1 = (line_addr + 1 * self.offsets[self.best_index]) << 6
                    addr_2 = (line_addr + 2 * self.offsets[self.best_index]) << 6
                    addr_2_alt = (line_addr + 1 * self.offsets[self.second_best_index]) << 6
                    acc = len({addr_2 >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc.append(acc)
                    acc_alt = len({addr_2_alt >> 6, addr_1 >> 6} & set(d[2] >> 6 for d in data[i + 1: i + 25]))
                    self.acc_alt.append(acc_alt)
                    # if acc_alt > acc:
                    #     addr_2 = addr_2_alt
                    prefetches.append((instr_id, addr_1))
                    prefetches.append((instr_id, addr_2))
                    prefetch_requests.append((cycle_count, addr_1))
                    prefetch_requests.append((cycle_count, addr_2))
            else:
                pass
            if i % percent == 0:
                print(i // percent, self.active_offsets, self.best_index_score,
                      sum(self.acc) / 2 / (len(self.acc) + 1),
                      sum(self.acc_alt) / 2 / (len(self.acc_alt) + 1))
                self.acc.clear()
                self.acc_alt.clear()
                self.active_offsets.clear()
        return prefetches


class TerribleMLModel(MLPrefetchModel):
    """
    This class effectively functions as a wrapper around the above custom
    pytorch nn.Module. You can approach this in another way so long as the the
    load/save/train/generate functions behave as described above.

    Disclaimer: It's terrible since the below criterion assumes a gold Y label
    for the prefetches, which we don't really have. In any case, the below
    structure more or less shows how one would use a ML framework with this
    script. Happy coding / researching! :)
    """

    degree = 2
    k = 2
    history = 4
    lookahead = int(os.environ.get('LOOKAHEAD', '0'))
    bucket = os.environ.get('BUCKET', 'page')
    window = history + lookahead + k
    filter_window = lookahead * degree
    batch_size = 4096

    def __init__(self):
        self.model = CNN()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def batch(self, data, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        bucket_data = defaultdict(list)
        batch_instr_id, batch_page, batch_x, batch_y = [], [], [], []
        for line in data:
            instr_id, cycles, load_address, ip, hit = line
            page = load_address >> 12
            bucket_buffer = bucket_data[eval(self.bucket)]
            bucket_buffer.append(load_address)
            batch_instr_id.append(instr_id)
            batch_page.append(page)
            batch_x.append(self.represent(bucket_buffer[:self.history]))
            batch_y.append(self.represent(bucket_buffer[-self.k:], box=False))
            if len(bucket_buffer) > self.window:
                bucket_buffer.pop(0)
            if len(batch_x) == batch_size:
                if torch.cuda.is_available():
                    yield batch_instr_id, batch_page, torch.Tensor(batch_x).cuda(), torch.Tensor(batch_y).cuda()
                else:
                    yield batch_instr_id, batch_page, torch.Tensor(batch_x), torch.Tensor(batch_y)
                batch_instr_id, batch_page, batch_x, batch_y = [], [], [], []

    def accuracy(self, output, label):
        return torch.sum(
            torch.logical_and(
                torch.scatter(
                    torch.zeros(output.shape, device=output.device), 1, torch.topk(output, self.k).indices, 1
                ),
                label
            )
        ) / label.shape[0] / self.k

    def train(self, data):
        print('LOOKAHEAD =', self.lookahead)
        print('BUCKET =', self.bucket)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # defining the loss function
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        # checking if GPU is available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        # converting the data into GPU format
        self.model.train()

        for epoch in range(20):
            accs = []
            losses = []
            percent = len(data) // self.batch_size // 100
            for i, (instr_id, page, x_train, y_train) in enumerate(self.batch(data)):
                # clearing the Gradients of the model parameters
                optimizer.zero_grad()

                # prediction for training and validation set
                output_train = self.model(x_train)

                # computing the training and validation loss
                loss_train = criterion(output_train, y_train)
                acc = self.accuracy(output_train, y_train)
                # print('Acc {}: {}'.format(epoch, acc))

                # computing the updated weights of all the model parameters
                loss_train.backward()
                optimizer.step()
                tr_loss = loss_train.item()
                accs.append(float(acc))
                losses.append(float(tr_loss))
                if i % percent == 0:
                    print('.', end='')
            print('Acc {}: {}'.format(epoch, sum(accs) / len(accs)))
            print('Epoch : ', epoch + 1, '\t', 'loss :', sum(losses))

    def generate(self, data):
        self.model.eval()
        prefetches = []
        accs = []
        for i, (instr_ids, pages, x, y) in enumerate(self.batch(data)):
            # breakpoint()
            pages = torch.LongTensor(pages).to(x.device)
            instr_ids = torch.LongTensor(instr_ids).to(x.device)
            y_preds = self.model(x)
            accs.append(float(self.accuracy(y_preds, y)))
            topk = torch.topk(y_preds, self.degree).indices
            shape = (topk.shape[0] * self.degree,)
            topk = topk.reshape(shape)
            pages = torch.repeat_interleave(pages, self.degree)
            instr_ids = torch.repeat_interleave(instr_ids, self.degree)
            addresses = (pages << 12) + (topk << 6)
            prefetches.extend(zip(map(int, instr_ids), map(int, addresses)))
            if i % 100 == 0:
                print('Chunk', i, 'Accuracy', sum(accs) / len(accs))
        return prefetches

    def represent(self, addresses, box=True):
        blocks = [(address >> 6) % 64 for address in addresses]
        raw = [0 for _ in range(64)]
        for block in blocks:
            raw[block] = 1
        if box:
            return [raw]
        else:
            return raw


# Replace this if you create your own model

ml_model_name = os.environ.get('ML_MODEL_NAME', 'TerribleMLModel')
Model = eval(ml_model_name)
