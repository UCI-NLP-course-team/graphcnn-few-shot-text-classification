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

TRAIN_IDX = 7931
VALID_IDX = 12822
TEST_IDX = -6024

full_adj = np.load('data/fs.ind.20ng.adj', allow_pickle=True).tocsr()
print('data/fs.ind.20ng.adj', full_adj.shape)

tx = np.load('data/fs.ind.20ng.tx', allow_pickle=True)
print('data/fs.ind.20ng.tx', tx.shape)

ty = np.load('data/fs.ind.20ng.ty', allow_pickle=True)
print('data/fs.ind.20ng.ty', ty.shape)

allx = np.load('data/fs.ind.20ng.allx', allow_pickle=True).toarray()
print('data/fs.ind.20ng.allx', allx.shape)

ally = np.load('data/fs.ind.20ng.ally', allow_pickle=True).toarray()
print('data/fs.ind.20ng.ally', ally.shape)

train_labels = np.unique(np.argmax(ally[:TRAIN_IDX], axis=1))
print('train_labels', train_labels)
valid_labels = np.unique(np.argmax(ally[TRAIN_IDX:VALID_IDX], axis=1))
print('valid_labels', valid_labels)
test_labels = np.unique(np.argmax(ty, axis=1))
print('test_labels', test_labels)

n_training_docs = int(ally.sum())
print('n_training_docs', n_training_docs)
n_training_samples = allx.shape[0]
print('n_training_samples', n_training_samples)

assert (n_training_docs == ally[:n_training_docs + 1].sum())
assert (full_adj.shape[0] - n_training_samples == tx.shape[0])
assert (ally[n_training_docs:-tx.shape[0]].sum() == 0)

train_features = allx
print('train_features', train_features.shape)
test_features = np.concatenate([allx, tx], 0)
ty = np.concatenate([ally, ty], 0)
print('test_features', test_features.shape)
print('ty', ty.shape)

if os.path.isfile('fs.graph_20ng.bin'):
    Gs, labels = load_graphs('fs.graph_20ng.bin')
else:
    train_G = nx.from_scipy_sparse_matrix(full_adj[:n_training_samples][:, :n_training_samples])
    train_DGL = dgl.DGLGraph()
    train_DGL.from_networkx(train_G, edge_attrs=['weight'])
    # train_DGL.from_scipy_sparse_matrix(full_adj[:n_training_samples][:, :n_training_samples])
    assert (len(train_DGL) == train_features.shape[0])

    test_G = nx.from_scipy_sparse_matrix(full_adj)
    test_DGL = dgl.DGLGraph()
    test_DGL.from_networkx(test_G, edge_attrs=['weight'])
    # test_DGL.from_scipy_sparse_matrix(full_adj[n_training_docs:][:, n_training_docs:])
    assert (len(test_DGL) == test_features.shape[0])

    Gs = [train_DGL, test_DGL]
    save_graphs('fs.graph_20ng.bin', Gs)
print(Gs[0])
print('load graph done')


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats, n_hidden, n_embs, n_layers, activation, dropout, aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(
            SAGEConv(n_hidden, n_embs, aggregator_type, feat_drop=dropout, activation=F.relu))  # activation None
        self.fc1 = nn.Linear(401, 100)
        self.fc2 = nn.Linear(100, 1)

    def get_diff(self, vec1, vec2):
        x3 = vec1 - vec2
        x3 = x3 * x3
        x1_ = vec1 * vec1
        x2_ = vec2 * vec2
        x4 = x1_ - x2_
        x5 = torch.cosine_similarity(vec1, vec2)
        conc = torch.cat([x5.view(-1, 1), x4, x3], axis=1)
        return conc

    def forward(self, graph, features, y, num_pairs_each_label=None, test_sim=False, test_fewshots={}):
        h = features
        for layer in self.layers:
            h = layer(graph, h)

        if test_sim or test_fewshots:
            lbs = test_labels
            h = h[TEST_IDX:]
            y = y[TEST_IDX:]
        else:
            lbs = train_labels

        if test_fewshots:
            n_ways = test_fewshots['n_ways']
            n_shots = test_fewshots['n_shots']
            lbs = np.random.choice(lbs, n_ways)
            supports = torch.zeros((n_ways * n_shots, h.shape[1]))
            testp = torch.zeros((1, h.shape[1]))

            for i, lb in enumerate(lbs):
                embeddings = h[y == lb]
                from_ = i * n_shots
                to_ = (i + 1) * n_shots

                if i == 0:
                    embeddings = embeddings[np.random.randint(embeddings.shape[0], size=n_shots + 1), :]
                    testp[0] = embeddings[0]
                    supports[from_:to_] = embeddings[1:]
                else:
                    embeddings = embeddings[np.random.randint(embeddings.shape[0], size=n_shots), :]
                    supports[from_:to_] = embeddings

            conc = self.get_diff(testp, supports)
            h = F.relu(self.fc1(conc.to(device)))
            h = F.sigmoid(self.fc2(h.to(device)))

            return h

        else:
            doc0_same = torch.zeros((num_pairs_each_label * len(lbs), h.shape[1]))
            doc1_same = torch.zeros_like(doc0_same)
            doc2_diff = torch.zeros_like(doc0_same)

            for i, lb in enumerate(lbs):
                embeddings_same = h[y == lb]
                embeddings_diff = h[y != lb]

                embeddings_same = embeddings_same[np.random.randint(embeddings_same.shape[0],
                                                                    size=num_pairs_each_label * 2), :]
                diffs = np.random.randint(TRAIN_IDX, size=num_pairs_each_label)

                embeddings_diff = embeddings_diff[diffs, :]
                from_ = i * num_pairs_each_label
                to_ = (i + 1) * num_pairs_each_label
                doc0_same[from_:to_] = embeddings_same[0::2]
                doc1_same[from_:to_] = embeddings_same[1::2]
                doc2_diff[from_:to_] = embeddings_diff

            conc1 = self.get_diff(doc0_same, doc1_same)
            conc2 = self.get_diff(doc0_same, doc2_diff)

            h = torch.cat([conc1, conc2])
            h = F.relu(self.fc1(h.to(device)))
            h = F.sigmoid(self.fc2(h.to(device)))

            labels_same = torch.ones((doc0_same.shape[0], 1))
            labels_diff = torch.zeros((doc2_diff.shape[0], 1))

            labels_sim = torch.cat([labels_same, labels_diff]).to(device)

            return h, labels_sim


device = 'cuda:0'
model = GraphSAGE(allx.shape[1], 400, 200, 1, F.relu, 0.25, 'gcn').to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
train_features = torch.from_numpy(train_features).float().to(device)
test_features = torch.from_numpy(test_features).float().to(device)
ally = torch.from_numpy(ally).argmax(1).to(device)
ty = torch.from_numpy(ty).argmax(1).to(device)
# print(Gs[0].in_degrees().sum(), Gs[0].out_degrees().sum())
# exit()

Gs[0].edata['weight'].unsqueeze_(1)
Gs[0] = Gs[0].to(torch.device(device))
Gs[1].edata['weight'].unsqueeze_(1)
Gs[1] = Gs[1].to(torch.device(device))

model.train()
save_every = 10
n_epochs = 700
best_loss = 9

# model.load_state_dict(torch.load('model_fs_20ng.pytorch'))
# optimizer.load_state_dict(torch.load('optimizer_fs_20ng.pytorch'))

for epoch in range(n_epochs):
    print('epoch', epoch)
    optimizer.zero_grad()
    similarities, labels_sim = model(Gs[0], train_features, ally, num_pairs_each_label=20)

    loss = F.binary_cross_entropy(similarities, labels_sim)
    loss.backward()
    optimizer.step()
    current_loss = loss.item()
    print(current_loss)

    if epoch % save_every == 0 and current_loss < best_loss:
        torch.save(model.state_dict(), 'model_fs_20ng.pytorch')
        torch.save(optimizer.state_dict(), 'optimizer_fs_20ng.pytorch')
        best_loss = current_loss

num_tests = 1000
with torch.no_grad():
    model.eval()

    test_fewshots = {'n_ways': 5, 'n_shots': 1}
    trues = 0
    for i in range(num_tests):
        similarities = model(Gs[1], test_features, ty, test_fewshots=test_fewshots)
        similarities = torch.squeeze(similarities)
        lbi = similarities.argmax()

        print('lbi', lbi)
        if lbi in range(test_fewshots['n_shots']):
            trues += 1
            if i > 0 and i % 100 == 0:
                print(trues / i)

    print(trues / num_tests)

num_tests = 1000
with torch.no_grad():
    model.eval()

    test_fewshots = {'n_ways': 5, 'n_shots': 5}
    k = 10
    trues = 0
    for i in range(num_tests):
        similarities = model(Gs[1], test_features, ty, test_fewshots=test_fewshots)
        similarities = torch.squeeze(similarities).cpu().numpy()
        top_k = similarities.argsort()[-k:][::-1]
        lbi = np.bincount(top_k).argmax()

        print('lbi', lbi)
        if lbi == 0:
            trues += 1
            if i > 0 and i % 100 == 0:
                print(trues / i)

    print(trues / num_tests)
