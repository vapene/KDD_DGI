import torch
import os.path as osp
import GCL.losses as L
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid
import random
import numpy as np
import os 

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h=z, g=g, hn=zn)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)
    # split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    split = {"train":data.train_mask, "valid":data.val_mask,"test":data.test_mask}
    result = LREvaluator()(z, data.y, split)
    return result



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize variables to sum the results
    sum_micro_f1 = 0
    sum_macro_f1 = 0
    num_trials = 10
    macro_list = []

    for trial in range(num_trials):
        seed=trial
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        path = osp.join(osp.expanduser('~'), 'datasets')
        dataset = Planetoid(path, name='Citeseer', transform=T.NormalizeFeatures())
        # data = dataset[0].to(device)
        transform= T.Compose([T.ToUndirected(), T.ToDevice(device)])
        data = transform(dataset[0])

        gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, num_layers=2).to(device)
        encoder_model = Encoder(encoder=gconv, hidden_dim=256).to(device)
        contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

        optimizer = Adam(encoder_model.parameters(), lr=0.005)

        with tqdm(total=300, desc=f'(Trial {trial+1})') as pbar:
            for epoch in range(1, 301):
                loss = train(encoder_model, contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        test_result = test(encoder_model, data)
        print(f'(Trial {trial+1}): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
        
        # Sum up the results for averaging later
        sum_micro_f1 += test_result["micro_f1"]
        sum_macro_f1 += test_result["macro_f1"]
        macro_list.append(test_result["macro_f1"])

    # Calculate the average results
    avg_micro_f1 = sum_micro_f1 / num_trials
    avg_macro_f1 = sum_macro_f1 / num_trials

    print(f'Average after {num_trials} trials: acc_list={macro_list}, average ACC={avg_macro_f1:.4f}')

# def main():
#     device = torch.device('cuda')
#     path = osp.join(osp.expanduser('~'), 'datasets')
#     dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
#     data = dataset[0].to(device)

#     gconv = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
#     encoder_model = Encoder(encoder=gconv, hidden_dim=512).to(device)
#     contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

#     optimizer = Adam(encoder_model.parameters(), lr=0.01)

#     with tqdm(total=300, desc='(T)') as pbar:
#         for epoch in range(1, 301):
#             loss = train(encoder_model, contrast_model, data, optimizer)
#             pbar.set_postfix({'loss': loss})
#             pbar.update()

#     test_result = test(encoder_model, data)
#     print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
