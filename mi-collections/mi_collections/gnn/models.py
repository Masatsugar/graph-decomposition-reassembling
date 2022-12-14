import dgl
import dgl.function as fn
import numpy
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch.conv import GraphConv
from dgllife.utils import EarlyStopping, Meter
from torch.utils.data import DataLoader


# CustomDataset, collate_molgraphs, mol_to_graph
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class GCNModel(nn.Module):
    """
    H(l+1)=σ(D{−1/2} A~ {D}{−1/2} H(l) W(l))
    """

    def __init__(self, in_feats, hidden_dim, outputs, n_iter=1):
        super(GCNModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.outputs = outputs

        self.gcn = GraphConv(
            in_feats=in_feats, out_feats=hidden_dim, activation=F.relu
        )  # (Nodes, Hidden_dim)
        self.linear = nn.Linear(
            in_features=hidden_dim, out_features=outputs
        )  # (Hidden_dim, 1)

        # self.gcn2 = GraphConv(in_feats=hidden_dim, out_feats=outputs)
        # self.layers = nn.ModuleList()
        # # input layer
        # self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # # hidden layers
        # for i in range(n_layers - 1):
        #     self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # # output layer
        # self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, g, n_feat, e_feat=None):
        x = self.gcn(g, n_feat)
        out = self.linear(x)
        return out


# model = GCNModel(1433, 20, 7)


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args.device)
            prediction, _ = model.predict(args, model, bg)
            eval_meter.update(prediction, labels, masks)
        total_score = numpy.mean(eval_meter.compute_metric(args.metric_name))
    return total_score


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.to(args.device), masks.to(args.device)
        prediction, _ = model(model, bg)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
    total_score = numpy.mean(train_meter.compute_metric(args.metric_name))
    print(
        f"epoch {epoch + 1}/{args.num_epochs}, training {args.metric_name,} {total_score:.4f}"
    )


args = dotdict(
    {
        "layers": 32,
        "optimizer": dotdict(
            {
                "learning_rate": 0.01,
                "decay_steps": 10000,
                "decay_ratio": 0.5,
                "weight_decay": 5e-4,  # Weight for L2 loss
            }
        ),
        "device": "cpu",
        "metric_name": "mse",
        "num_epochs": 10,
        "patience": 50,
    }
)


def train(args, model, train_loader, val_loader=None):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.optimizer.learning_rate,
        weight_decay=args.optimizer.weight_decay,
    )
    stopper = EarlyStopping(args.patience)
    loss_fn = nn.MSELoss(reduction="none")
    for epoch in range(args.num_epochs):
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)
        if val_loader is not None:
            val_score = run_an_eval_epoch(args, model, val_loader)
            early_stop = stopper.step(val_score, model)
            print(
                f"epoch {epoch + 1}/{args.num_epochs}, validation {args.metric_name} {val_score:.4f}, "
                f"best validation {args.metric_name} {stopper.best_score:.4f}"
            )
            if early_stop:
                break
