import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math


USE_CUDA = torch.cuda.is_available()


class CNN(nn.Module):
    """
    input: [batch_size, 3, 64, 128]
    output: [batch_size, 32, 256]
    """

    def __init__(self):
        super(CNN, self).__init__()
        channel_tmp = 3
        main = nn.Sequential()
        for i in range(5):
            main.add_module("ResBlk-{0}".format(i), ResBlk(channel_tmp, 32 * 2 ** min(i, 3)))
            channel_tmp = 32 * 2 ** min(i, 3)
            if i < 2:
                main.add_module("MAXPOOL-{0}".format(i), nn.MaxPool2d(kernel_size=2))
            elif i < 4:
                main.add_module("MAXPOOL-{0}".format(i), nn.MaxPool2d(kernel_size=(2, 1)))
            else:
                main.add_module("MAXPOOL-{0}".format(i), nn.MaxPool2d(kernel_size=(4, 1)))
            main.add_module("Dropout-{0}".format(i), nn.Dropout(0.1))
        self.main = main

    def forward(self, x):
        out = self.main(x).squeeze(2)
        out = out.transpose(1, 2)

        return out


class Encoder(nn.Module):
    """
    input: [batch_size, 32, 256]
    output: out [batch_size, 32, 256]
            hidden [2, batch_size, 128]
    """

    def __init__(self, num_rnn_layers=2, rnn_hidden_size=128, dropout=0.5):
        super(Encoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.gru = nn.GRU(256, rnn_hidden_size, num_rnn_layers,
                          batch_first=True,
                          dropout=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size))
        if USE_CUDA:
            h0 = h0.cuda()
        out, hidden = self.gru(x, h0)

        return out, hidden


class RNNAttnDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_rnn_layers=2, dropout=0.5):
        super(RNNAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(vocab_size + hidden_size, hidden_size,
                          num_rnn_layers, batch_first=True,
                          dropout=dropout)

        self.wc = nn.Linear(2 * hidden_size, hidden_size)  # ,bias=False)
        self.ws = nn.Linear(hidden_size, vocab_size)

        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        fix_embedding = torch.from_numpy(np.eye(vocab_size, vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, y, encoder_outputs, encoder_hidden, is_training):
        batch_size = y.size(0)
        max_len = y.size(1)

        last_hidden = encoder_hidden
        last_ht = Variable(torch.zeros(batch_size, self.hidden_size))
        outputs = []

        if USE_CUDA:
            last_ht = last_ht.cuda()

        if not is_training:
            input = y[:, 0]

        for di in range(max_len - 1):
            if is_training:
                input = y[:, di]
            output, last_ht, last_hidden, alpha = self.forward_step(input, last_ht, last_hidden, encoder_outputs)
            if not is_training:
                input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def forward_2(self, encoder_outputs, encoder_hidden, max_len):
        batch_size = encoder_outputs.size(0)

        last_hidden = encoder_hidden
        last_ht = Variable(torch.zeros(batch_size, self.hidden_size))
        outputs = []

        if USE_CUDA:
            last_ht = last_ht.cuda()

        input = torch.zeros([batch_size]).long()
        if USE_CUDA:
            input = input.cuda()

        for di in range(max_len - 1):
            output, last_ht, last_hidden, alpha = self.forward_step(input, last_ht, last_hidden, encoder_outputs)
            input = output.max(1)[1]
            outputs.append(output.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def forward_step(self, input, last_ht, last_hidden, encoder_outputs):
        embed_input = self.embedding(input)
        rnn_input = torch.cat((embed_input, last_ht), 1)
        output, hidden = self.gru(rnn_input.unsqueeze(1), last_hidden)
        output = output.squeeze(1)

        weighted_context, alpha = self.attn(output, encoder_outputs)
        ht = self.tanh(self.wc(torch.cat((output, weighted_context), 1)))
        output = self.ws(ht)
        return output, ht, hidden, alpha


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out

        return out


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        hidden_expanded = hidden.unsqueeze(2)

        energy = torch.bmm(encoder_outputs, hidden_expanded).squeeze(2)

        alpha = nn.functional.softmax(energy)
        weighted_context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)

        return weighted_context, alpha


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dot_number, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dot_number = dot_number
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.A_D = Parameter(torch.from_numpy(compute_matrix(dot_number)).float(), requires_grad=False)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.A_D.detach(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


def compute_matrix(dot_number):
    D = np.eye(dot_number, k=-1) + np.eye(dot_number, k=0) + np.eye(dot_number, k=1)
    return D


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiAttention(nn.Module):
    def __init__(self, dim, dim_2, n_head, mask):
        super(MultiAttention, self).__init__()
        self.n_head = n_head
        self.dim_2 = dim_2
        self.mask = mask
        self.W_Q = nn.Linear(dim, dim_2 * n_head, bias=False)
        self.W_K = nn.Linear(dim, dim_2 * n_head, bias=False)
        self.W_V = nn.Linear(dim, dim_2 * n_head, bias=False)
        self.fc = nn.Linear(dim_2 * n_head, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [batch_size, len, dim]
        batch_size = x.size(0)
        dot_number = x.size(1)

        Q = self.W_Q(x).view(batch_size, -1, self.n_head, self.dim_2).transpose(1, 2)
        K = self.W_K(x).view(batch_size, -1, self.n_head, self.dim_2).transpose(1, 2)
        V = self.W_V(x).view(batch_size, -1, self.n_head, self.dim_2).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dim_2)

        if self.mask:
            mask = torch.from_numpy(compute_matrix(dot_number)).float().detach()
            if USE_CUDA:
                mask = mask.cuda()
            scores = scores.masked_fill(mask == 0, -1e-9)

        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.dim_2)
        output = self.fc(context)
        output = self.norm(output + x)
        return output
