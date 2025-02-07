import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

class ncGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0

    def forward(self,g,x):
        z = self.GNN(g,x)
        y = self.classifier(z)
        return y
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        ### update E, G
        self.optimizer_G.zero_grad()

        h = self.GNN(g,x)
        y = self.classifier(h)

        y_score = torch.sigmoid(y)

        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        
        self.cls_loss.backward()
        self.optimizer_G.step()
