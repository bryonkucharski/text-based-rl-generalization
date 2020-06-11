import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class RelationalGraphConvolution(torch.nn.Module):
    """
    Simple R-GCN layer, modified from theano/keras implementation from https://github.com/tkipf/relational-gcn
    We also consider relation representation here (relation labels matter)
    """

    def __init__(self, entity_input_dim, relation_input_dim, num_relations, out_dim, bias=True, num_bases=0):
        super(RelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        if self.num_bases > 0:
            self.bottleneck_layer = torch.nn.Linear((self.entity_input_dim + self.relation_input_dim) * self.num_relations, self.num_bases, bias = False)
            self.weight = torch.nn.Linear(self.num_bases, self.out_dim, bias=False)
        else:
            self.weight = torch.nn.Linear((self.entity_input_dim + self.relation_input_dim) * self.num_relations, self.out_dim, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, node_features, relation_features, adj):
        # node_features: batch x num_entity x in_dim
        # relation_features: batch x num_relation x in_dim
        # adj:   batch x num_relations x num_entity x num_entity
        supports = []
        for relation_idx in range(self.num_relations):
            _r_features = relation_features[:, relation_idx: relation_idx + 1]  # batch x 1 x in_dim
            _r_features = _r_features.repeat(1, node_features.size(1), 1)  # batch x num_entity x in_dim
            supports.append(torch.bmm(adj[:, relation_idx], torch.cat([node_features, _r_features], dim=-1)))  # batch x num_entity x in_dim+in_dim
        supports = torch.cat(supports, dim=-1)  # batch x num_entity x (in_dim+in_dim)*num_relations
        if self.num_bases > 0:
            supports = self.bottleneck_layer(supports.float())
        output = self.weight(supports)  # batch x num_entity x out_dim

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class StackedRelationalGraphConvolution(torch.nn.Module):
    '''
    input:  entity features:    batch x num_entity x input_dim
            relation features:  batch x num_relations x input_dim
            adjacency matrix:   batch x num_relations x num_entity x num_entity
    '''

    def __init__(self, entity_input_dim, relation_input_dim, num_relations, hidden_dims, num_bases, use_highway_connections=False, dropout_rate=0.0):
        super(StackedRelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.hidden_dims = hidden_dims
        self.num_relations = num_relations
        self.dropout_rate = dropout_rate
        self.num_bases = num_bases
        self.nlayers = len(self.hidden_dims)
        self.stack_gcns()
        self.use_highway_connections = use_highway_connections
        if self.use_highway_connections:
            self.stack_highway_connections()

    def stack_highway_connections(self):
        highways = [torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i]) for i in range(self.nlayers)]
        self.highways = torch.nn.ModuleList(highways)
        self.input_linear = torch.nn.Linear(self.entity_input_dim, self.hidden_dims[0])

    def stack_gcns(self):
        gcns = [RelationalGraphConvolution(self.entity_input_dim if i == 0 else self.hidden_dims[i - 1], self.relation_input_dim, self.num_relations, self.hidden_dims[i], num_bases=self.num_bases)
                for i in range(self.nlayers)]
        self.gcns = torch.nn.ModuleList(gcns)

    def forward(self, node_features, relation_features, adj):
        x = node_features
        for i in range(self.nlayers):
            if self.use_highway_connections:
                if i == 0:
                    prev = self.input_linear(x)
                else:
                    prev = x.clone()
            x = self.gcns[i](x, relation_features, adj)  # batch x num_nodes x hid
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
            if self.use_highway_connections:
                gate = torch.sigmoid(self.highways[i](x))
                x = gate * x + (1 - gate) * prev
        return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = torch.zeros_like(e)
        zero_vec = zero_vec.fill_(9e-15)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'