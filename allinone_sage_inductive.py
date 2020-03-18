import numpy as np
import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
from sageconv import SAGEConv
from dgl.data.utils import save_graphs, load_graphs

full_adj = np.load('data/ind.20ng.adj', allow_pickle=True).tocsr()
print('data/ind.20ng.adj', full_adj.shape)

tx = np.load('data/ind.20ng.tx', allow_pickle=True)
print('data/ind.20ng.tx', tx.shape)

ty = np.load('data/ind.20ng.ty', allow_pickle=True)
print('data/ind.20ng.ty', ty.shape)

allx = np.load('data/ind.20ng.allx', allow_pickle=True).toarray()
print('data/ind.20ng.allx', allx.shape)

ally = np.load('data/ind.20ng.ally', allow_pickle=True).toarray()
print('data/ind.20ng.ally', ally.shape)

n_training_docs = int(ally.sum())
print(n_training_docs, n_training_docs)
n_training_samples = allx.shape[0]
print('n_training_samples', n_training_samples)

assert (n_training_docs == ally[:n_training_docs + 1].sum())
assert (full_adj.shape[0] - n_training_samples == tx.shape[0])
assert (ally[n_training_docs:-tx.shape[0]].sum() == 0)

train_features = allx
print('train_features', train_features.shape)
test_features = np.concatenate([allx[n_training_docs:], tx], 0)
print('test_features', test_features.shape)

if os.path.isfile('graph.bin'):
    Gs, labels = load_graphs('graph.bin')
else:
    train_G = nx.from_scipy_sparse_matrix(full_adj[:n_training_samples][:, :n_training_samples])
    train_DGL = dgl.DGLGraph()
    train_DGL.from_networkx(train_G, edge_attrs=['weight'])
    assert (len(train_DGL) == train_features.shape[0])

    test_G = nx.from_scipy_sparse_matrix(full_adj[n_training_docs:][:, n_training_docs:])
    test_DGL = dgl.DGLGraph()
    test_DGL.from_networkx(test_G, edge_attrs=['weight'])
    assert (len(test_DGL) == test_features.shape[0])

    Gs = [train_DGL, test_DGL]
    save_graphs('graph.bin', Gs)
print(Gs[0])
print('load graph done')


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))  # activation None

    def forward(self, graph, features):
        h = features
        for layer in self.layers:
            h = layer(graph, h)
        return F.log_softmax(h, dim=1)


device = 'cuda:0'
model = GraphSAGE(allx.shape[1], 200, 20, 1, F.relu, 0.5, 'gcn').to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-2)
train_features = torch.from_numpy(train_features).float().to(device)
test_features = torch.from_numpy(test_features).float().to(device)
ally = torch.from_numpy(ally).argmax(1).to(device)
ty = torch.from_numpy(ty).argmax(1).to(device)

Gs[0].edata['weight'].unsqueeze_(1)
Gs[0] = Gs[0].to(torch.device(device))
Gs[1].edata['weight'].unsqueeze_(1)
Gs[1] = Gs[1].to(torch.device(device))

model.train()
save_every = 10
best_loss = np.inf
n_epochs = 3000
# model.load_state_dict(torch.load('model.pytorch'))
# optimizer.load_state_dict(torch.load('optimizer.pytorch'))

for epoch in range(n_epochs):
    print('epoch', epoch)
    optimizer.zero_grad()
    output = model(Gs[0], train_features)
    loss = F.nll_loss(output[:n_training_docs], ally[:n_training_docs])
    loss.backward()
    optimizer.step()
    current_loss = loss.item()
    print(current_loss)

    if epoch % save_every == 0 and current_loss < best_loss:
        torch.save(model.state_dict(), 'model.pytorch')
        torch.save(optimizer.state_dict(), 'model.pytorch')
        best_loss = current_loss

with torch.no_grad():
    model.eval()
    output = model(Gs[1], test_features)
    pred_y = output.argmax(1)[-ty.shape[0]:]
    print((pred_y == ty).long().sum().float() / float(ty.size(0)))