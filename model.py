from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn


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
            prefetches.append((instr_id, ((load_addr >> 6) + 1) << 12))
            prefetches.append((instr_id, ((load_addr >> 6) + 2) << 12))

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
    k = 4
    history = 4
    lookahead = 16
    window = history + lookahead + k
    filter_window = lookahead * degree
    batch_size = 128

    def __init__(self):
        self.model = CNN()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def batch(self, data, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        page_data = defaultdict(list)
        batch_instr_id, batch_page, batch_x, batch_y = [], [], [], []
        for line in data:
            instr_id, cycles, load_address, ip, hit = line
            block = load_address >> 6
            page = load_address >> 6
            page_buffer = page_data[page]
            page_buffer.append(block)
            batch_instr_id.append(instr_id)
            batch_page.append(page)
            batch_x.append(self.represent(page_buffer[:self.history]))
            batch_y.append(self.represent(page_buffer[-self.k:]))
            if len(page_buffer) > self.window:
                page_buffer.pop(0)
            if len(batch_x) == batch_size:
                if torch.cuda.is_available():
                    yield batch_instr_id, batch_page, torch.Tensor(batch_x).cuda(), torch.Tensor(batch_y).cuda()
                else:
                    yield batch_instr_id, batch_page, torch.Tensor(batch_x), torch.Tensor(batch_y)
                batch_instr_id, batch_page, batch_x, batch_y = [], [], [], []

    def train(self, data):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.07)
        # defining the loss function
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        # checking if GPU is available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        # converting the data into GPU format
        self.model.train()

        def accuracy(output, label):
            return torch.sum(
                torch.logical_and(
                    torch.scatter(
                        torch.zeros(output.shape), 1, torch.topk(output, self.k).indices, 1
                    ),
                    label
                )
            ) / label.shape[0] / self.degree

        for epoch in range(10):
            for instr_id, page, x_train, y_train in self.batch(data):
                # clearing the Gradients of the model parameters
                optimizer.zero_grad()

                # prediction for training and validation set
                output_train = self.model(x_train)

                # computing the training and validation loss
                loss_train = criterion(output_train, y_train)
                acc = accuracy(output_train, y_train)
                print('Acc {}: {}'.format(epoch, acc))

                # computing the updated weights of all the model parameters
                loss_train.backward()
                optimizer.step()
                tr_loss = loss_train.item()
                print('Epoch : ', epoch + 1, '\t', 'loss :', tr_loss)

    def generate(self, data):
        self.model.eval()
        prefetches = []
        for (instr_id,), (page,), x, y in enumerate(self.batch(data, batch_size=1)):
            y_pred = self.model(x)
            fltr = torch.Tensor(self.represent(prefetches[-self.filter_window:]))
            if torch.cuda.is_available():
                fltr = fltr.cuda()
            for block in torch.sort((1. * (not fltr)) * y_pred[0], descending=True).indices[:self.degree]:
                prefetches.append((instr_id, block << 6 + page << 12))
        return prefetches

    def represent(self, addresses):
        blocks = [(address >> 6) % 64 for address in addresses]
        raw = [0 for _ in range(64)]
        for block in blocks:
            raw[block] = 1
        return raw


# Replace this if you create your own model
Model = TerribleMLModel

