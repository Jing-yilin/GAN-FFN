import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector

        return attn_pool, alpha


class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction="mean", gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            logp = self.ce(pred * mask_, target) / torch.sum(mask)
            p = torch.exp(-logp)
        else:
            logp = self.ce(pred * mask_, target) / torch.sum(
                self.weight[target] * mask_.squeeze()
            )
            p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction="sum")

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(
                self.weight[target] * mask_.squeeze()
            )
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction="sum")

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) / torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, vector
        return attn_pool, alpha


class MatchingAttention(nn.Module):
    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type="general2"):
        super(MatchingAttention, self).__init__()
        assert att_type != "concat" or alpha_dim != None
        assert att_type != "dot" or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == "general":
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == "general2":
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            torch.nn.init.normal_(self.transform.weight, std=0.01)
        elif att_type == "concat":
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == "dot":
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == "general":
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == "general2":
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            mask_ = (
                mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            )  # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(
                1, 2
            )  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim=None,
        out_dim=None,
        n_head=4,
        score_function="scaled_dot_product",
        dropout=0.6,
    ):
        """Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        """
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == "mlp":
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == "bi_linear":
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == "dot_product":
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == "scaled_dot_product":
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == "mlp":
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == "bi_linear":
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError("invalid score_function")

        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(
            torch.split(output, mb_size, dim=0), dim=-1
        )  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


def Matching(matchatt, emotions, modal, umask):
    att_emotions = []
    alpha = []
    for t in modal:
        att_em, alpha_ = matchatt(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:, 0, :])
    att_emotions = torch.cat(att_emotions, dim=0)
    hidden = att_emotions + F.gelu(emotions)
    return hidden, alpha


class CNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_filter,
        filter_sizes,
        output_dim,
        dropout=0.2,
        pad_idx=0,
    ):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filter,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )
        # in_channels：输入的channel，文字都是1
        # out_channels：输出的channel维度
        # fs：每次滑动窗口计算用到几个单词,相当于n-gram中的n
        # for fs in filter_sizes用好几个卷积模型最后concate起来看效果。

        self.fc = nn.Linear(len(filter_sizes) * num_filter, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, qmask, umask):
        alpha, alpha_f, alpha_b = [], [], []
        text = text.permute(1, 0, 2)
        # embedded = self.dropout(self.embedding(text))  # [batch size, sent len, emb dim]
        text = text.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        # print(embedded.shape)
        # 升维是为了和nn.Conv2d的输入维度吻合，把channel列升维。
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        # print(conved[0].shape,conved[1].shape,conved[2].shape)
        # conved = [batch size, num_filter, sent len - filter_sizes+1]
        # 有几个filter_sizes就有几个conved
        print(conved[1].size())
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved
        ]  # [batch,num_filter]
        # print(pooled[0].shape,pooled[1].shape,pooled[2].shape)
        x_cat = torch.cat(pooled, dim=1)
        # print(x_cat.shape)
        cat = self.dropout(x_cat)
        # cat = [batch size, num_filter * len(filter_sizes)]
        # 把 len(filter_sizes)个卷积模型concate起来传到全连接层。
        log_prob = F.log_softmax(self.fc(cat), 2)
        return log_prob, alpha, alpha_f, alpha_b, x_cat


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.out_channel = 100
        self.conv3 = nn.Conv2d(1, 1, (3, 100))
        self.conv4 = nn.Conv2d(1, 1, (4, 100))
        self.conv5 = nn.Conv2d(1, 1, (5, 100))
        self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size - 3 + 1, 1))
        self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size - 5 + 1, 1))
        self.linear1 = nn.Linear(3, config.label_num)

    def forward(self, x):
        batch = x.shape[1]
        x = x.permute(1, 0, 2)
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.config.label_num)

        return x


class LSTMModel(nn.Module):
    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5, attention=True):
        super(LSTMModel, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm_1 = nn.LSTM(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_2 = nn.LSTM(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_3 = nn.LSTM(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )

        # if self.attention:
        #     self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')
        self.attention = Attention(600)

        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(600, n_classes)

    def forward(self, textf, acouf, visuf, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions_1, hidden_1 = self.lstm_1(textf)
        emotions_2, hidden_2 = self.lstm_2(acouf)
        emotions_3, hidden_3 = self.lstm_3(visuf[:, :, :100])
        alpha, alpha_f, alpha_b = [], [], []
        emotion = torch.cat((emotions_1, emotions_2, emotions_3), dim=-1)
        att, score = self.attention(emotion, emotion)
        emotion = F.gelu(emotion + att)

        # emotions = self.lstm(torch.cat(textf, acouf, visuf), dim=1)

        # if self.attention:
        #     # att_emotions = []
        #     alpha = []
        #     emotions = [emotions_1, emotions_2, emotions_3]
        #     hidden = 0.
        #     for i in emotions:
        #         for j in emotions:
        #             hid, alpha = Matching(self.matchatt, i, j, umask)
        #             hidden += hid

        # for t in emotions_1:
        #      att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
        #      att_emotions.append(att_em.unsqueeze(0))
        #      alpha.append(alpha_[:, 0, :])
        # att_emotions = torch.cat(att_emotions, dim=0)
        # hidden_1 = att_emotions + F.gelu(emotions)

        # att_em, alpha_ = self.attention(emotions, emotions)
        # # hidden = att_em + F.gelu(self.linear(emotions))
        # hidden = att_em + F.gelu(emotions)

        # else:
        #     hidden = F.relu(self.linear(emotions_1))

        hidden = emotion
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        # log_prob = F.softmax(self.smax_fc(hidden))
        return log_prob, alpha, alpha_f, alpha_b, hidden


class LSTMModel2(nn.Module):
    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5, attention=False):
        super(LSTMModel2, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )

        if self.attention:
            self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")

        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []

        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        # print("log_prob.size() = ", log_prob.size()) # log_prob.size() =  torch.Size([94, 32, 6])
        return log_prob, alpha, alpha_f, alpha_b


class MELDLSTMModel(nn.Module):
    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        super(MELDLSTMModel, self).__init__()

        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=D_m,
            hidden_size=D_e,
            num_layers=4,
            bidirectional=True,
            dropout=dropout,
        )
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, visuf=None, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        # U = torch.cat((U, visuf, acouf), dim=2)
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []

        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_em = F.hardswish(att_em)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.hardswish(emotions + att_emotions)
        else:
            hidden = F.gelu(self.linear(emotions))

        # hidden = F.relu(self.linear(emotions))
        # hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b


class FullyConnection(nn.Module):
    def __init__(self):
        super(FullyConnection, self).__init__()

        # self.fc1 = nn.Linear(512, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 128)
        # self.fc5 = nn.Linear(128, 100)
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 2048)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        x = self.fc5(x)
        return x


class Emoformer(nn.Module):
    def __init__(self, D_m, D_e, n_classes=7, dropout=0.5, attention=True):
        super(Emoformer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.attention = attention

        # self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_1 = nn.LSTM(
            input_size=2 * D_e,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )

        self.attention_1 = Attention(D_m)
        self.attention_2 = Attention(D_m)
        self.attention_3 = Attention(D_m + 412)

        # self.attention_4 = Attention(D_m + 412)
        # self.attention_5 = Attention(D_m + 412)
        # self.attention_6 = Attention(D_m + 412)
        self.attention_4 = Attention(D_m)
        self.attention_5 = Attention(D_m)
        self.attention_6 = Attention(D_m)

        self.norm_1 = nn.LayerNorm(D_m)
        self.norm_2 = nn.LayerNorm(D_m + 412)
        # self.transform = nn.Linear(D_m, D_m + 412)
        self.transform = nn.Linear(512, D_m)
        self.transform2 = nn.Linear(D_m, 2 * D_e)
        self.transform3 = nn.Linear(512, 100)
        self.transform4 = nn.Linear(100, 2048)

        self.fc1 = FullyConnection()
        self.fc2 = FullyConnection()
        self.fc3 = FullyConnection()

        if self.attention:
            self.matchatt = MatchingAttention(2048, 2048, att_type="general2")

        self.smax_fc = nn.Linear(2 * D_e, n_classes)

    def forward(self, textf, acouf, visuf, qmask, umask):
        alpha, alpha_f, alpha_b = [], [], []

        textf_u = textf
        output_t1, score_t1 = self.attention_1(textf, textf)
        output_a1, score_a1 = self.attention_2(acouf, acouf)
        output_v1, score_v1 = self.attention_3(visuf, visuf)
        textf_1 = self.norm_1(textf + output_t1)
        acouf_1 = self.norm_1(acouf + output_a1)
        visuf_1 = self.norm_2(visuf + output_v1)
        # textf_1 = F.gelu(self.transform(textf_1))
        # acouf_1 = F.gelu(self.transform(acouf_1))
        visuf_1 = F.relu(self.transform3(visuf_1))

        output_t2, score_t2 = self.attention_4(textf_1, textf_1)
        output_a2, score_a2 = self.attention_5(acouf_1, acouf_1)
        output_v2, score_v2 = self.attention_6(visuf_1, visuf_1)
        # textf_1 = self.norm_2(textf_1 + output_t2)
        # acouf_1 = self.norm_2(acouf_1 + output_a2)
        # visuf_1 = self.norm_2(visuf_1 + output_v2)
        textf_1 = self.norm_1(textf_1 + output_t2)
        acouf_1 = self.norm_1(acouf_1 + output_a2)
        visuf_1 = self.norm_1(visuf_1 + output_v2)
        # textf = self.fc1(self.transform(textf) + textf_1)
        # acouf = self.fc2(self.transform(acouf) + acouf_1)
        textf = self.fc1(textf + textf_1)
        acouf = self.fc2(acouf + acouf_1)
        visuf = self.fc3(self.transform(visuf) + visuf_1)

        if self.attention:
            alpha, alpha_f, alpha_b = [], [], []
            alpha = []
            emotions = [textf, acouf, visuf]
            output = 0.0
            for i in emotions:
                for j in emotions:
                    hid, alpha = Matching(self.matchatt, i, j, umask)
                    output += hid

            output, hidden = self.lstm_1(self.transform2(output + textf_u))
            # output = F.gelu(self.transform2(textf_u)) + output

        else:
            output = self.transform4(textf_u) + textf + acouf + visuf
            output, hidden = self.lstm(output)

        output = self.dropout(output)
        log_prob = F.log_softmax(self.smax_fc(output), 2)
        return log_prob, alpha, alpha_f, alpha_b, output


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout
    ):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim, out_channels=filters, kernel_size=K
                )
                for K in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(
            torch.from_numpy(pretrained_word_vectors).float()
        )
        # if is_static:
        self.embedding.weight.requires_grad = False

    def forward(self, x, umask):
        num_utt, batch, num_words = x.size()

        x = x.type(LongTensor)  # (num_utt, batch, num_words)
        x = x.view(
            -1, num_words
        )  # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(
            x
        )  # (num_utt * batch, num_words) -> (num_utt * batch, num_words, embedding_dim)
        emb = emb.transpose(
            -2, -1
        ).contiguous()  # (num_utt * batch, num_words, embedding_dim)  -> (num_utt * batch, embedding_dim, num_words)

        convoluted = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, 1)
        features = F.relu(
            self.fc(self.dropout(concated))
        )  # (num_utt * batch, embedding_dim//2) -> (num_utt * batch, output_size)
        features = features.view(
            num_utt, batch, -1
        )  # (num_utt * batch, output_size) -> (num_utt, batch, output_size)
        mask = umask.unsqueeze(-1).type(
            FloatTensor
        )  # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1)  # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(
            1, 1, self.feature_dim
        )  # (num_utt, batch, 1) -> (num_utt, batch, output_size)
        features = (
            features * mask
        )  # (num_utt, batch, output_size) -> (num_utt, batch, output_size)

        return features


class E2ELSTMModel(nn.Module):
    def __init__(
        self,
        D_e,
        D_h,
        vocab_size,
        embedding_dim=300,
        cnn_output_size=100,
        cnn_filters=50,
        cnn_kernel_sizes=(3, 4, 5),
        cnn_dropout=0.5,
        n_classes=7,
        dropout=0.5,
        attention=False,
    ):
        super(E2ELSTMModel, self).__init__()

        self.cnn_feat_extractor = CNNFeatureExtractor(
            vocab_size,
            embedding_dim,
            cnn_output_size,
            cnn_filters,
            cnn_kernel_sizes,
            cnn_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=D_e,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )

        if self.attention:
            self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")

        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def init_pretrained_embeddings(self, pretrained_word_vectors):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(
            pretrained_word_vectors
        )

    def forward(self, input_seq, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.cnn_feat_extractor(input_seq, umask)

        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []

        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b


class DialogueRNNCell(nn.Module):
    def __init__(
        self,
        D_m,
        D_g,
        D_p,
        D_e,
        listener_state=False,
        context_attention="simple",
        D_a=100,
        dropout=0.5,
    ):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m + D_p, D_g)
        self.p_cell = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m + D_p, D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention == "simple":
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(
            torch.cat([U, q0_sel], dim=1),
            torch.zeros(U.size()[0], self.D_g).type(U.type())
            if g_hist.size()[0] == 0
            else g_hist[-1],
        )
        g_ = self.dropout(g_)
        if g_hist.size()[0] == 0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)
        # c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0\
        #         else self.attention(g_hist,U)[0] # batch, D_g
        U_c_ = torch.cat([U, c_], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        qs_ = self.p_cell(
            U_c_.contiguous().view(-1, self.D_m + self.D_g), q0.view(-1, self.D_p)
        ).view(U.size()[0], -1, self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = (
                U.unsqueeze(1)
                .expand(-1, qmask.size()[1], -1)
                .contiguous()
                .view(-1, self.D_m)
            )
            ss_ = (
                self._select_parties(qs_, qm_idx)
                .unsqueeze(1)
                .expand(-1, qmask.size()[1], -1)
                .contiguous()
                .view(-1, self.D_p)
            )
            U_ss_ = torch.cat([U_, ss_], 1)
            ql_ = self.l_cell(U_ss_, q0.view(-1, self.D_p)).view(
                U.size()[0], -1, self.D_p
            )
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_ * (1 - qmask_) + qs_ * qmask_
        e0 = (
            torch.zeros(qmask.size()[0], self.D_e).type(U.type())
            if e0.size()[0] == 0
            else e0
        )
        e_ = self.e_cell(self._select_parties(q_, qm_idx), e0)
        e_ = self.dropout(e_)

        return g_, q_, e_, alpha


class DialogueRNN(nn.Module):
    def __init__(
        self,
        D_m,
        D_g,
        D_p,
        D_e,
        listener_state=False,
        context_attention="simple",
        D_a=100,
        dropout=0.5,
    ):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(
            D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout
        )

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type())  # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(
            U.type()
        )  # batch, party, D_p
        e_ = torch.zeros(0).type(U.type())  # batch, D_e
        e = e_

        alpha = []
        for u_, qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            e = torch.cat([e, e_.unsqueeze(0)], 0)
            if type(alpha_) != type(None):
                alpha.append(alpha_[:, 0, :])

        return e, alpha  # seq_len, batch, D_e


class BiModel(nn.Module):
    def __init__(
        self,
        D_m,
        D_g,
        D_p,
        D_e,
        D_h,
        n_classes=7,
        listener_state=False,
        context_attention="simple",
        D_a=100,
        dropout_rec=0.5,
        dropout=0.5,
    ):
        super(BiModel, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.dialog_rnn_f = DialogueRNN(
            D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec
        )
        self.dialog_rnn_r = DialogueRNN(
            D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec
        )
        self.linear = nn.Linear(2 * D_e, 2 * D_h)
        self.smax_fc = nn.Linear(2 * D_h, n_classes)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)  # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)  # seq_len, batch, n_classes
        if att2:
            return log_prob, alpha, alpha_f, alpha_b
        else:
            return log_prob, [], alpha_f, alpha_b


class BiE2EModel(nn.Module):
    def __init__(
        self,
        D_emb,
        D_m,
        D_g,
        D_p,
        D_e,
        D_h,
        word_embeddings,
        n_classes=7,
        listener_state=False,
        context_attention="simple",
        D_a=100,
        dropout_rec=0.5,
        dropout=0.5,
    ):
        super(BiE2EModel, self).__init__()

        self.D_emb = D_emb
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        # self.dropout_rec = nn.Dropout(0.2)
        self.dropout_rec = nn.Dropout(dropout)
        self.turn_rnn = nn.GRU(D_emb, D_m)
        self.dialog_rnn_f = DialogueRNN(
            D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec
        )
        self.dialog_rnn_r = DialogueRNN(
            D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec
        )
        self.linear1 = nn.Linear(2 * D_e, D_h)
        # self.linear2     = nn.Linear(D_h, D_h)
        # self.linear3     = nn.Linear(D_h, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        self.embedding = nn.Embedding(
            word_embeddings.shape[0], word_embeddings.shape[1]
        )
        self.embedding.weight.data.copy_(word_embeddings)
        self.embedding.weight.requires_grad = True
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, data, att2=False):
        # T1 = word_embeddings[data.turn1] # seq_len, batch, D_emb
        # T2 = word_embeddings[data.turn2] # seq_len, batch, D_emb
        # T3 = word_embeddings[data.turn3] # seq_len, batch, D_emb

        T1 = self.embedding(data.turn1)
        T2 = self.embedding(data.turn2)
        T3 = self.embedding(data.turn3)

        T1_, h_out1 = self.turn_rnn(
            T1, torch.zeros(1, T1.size(1), self.D_m).type(T1.type())
        )
        T2_, h_out2 = self.turn_rnn(
            T2, torch.zeros(1, T1.size(1), self.D_m).type(T1.type())
        )
        T3_, h_out3 = self.turn_rnn(
            T3, torch.zeros(1, T1.size(1), self.D_m).type(T1.type())
        )

        U = torch.cat([h_out1, h_out2, h_out3], 0)  # 3, batch, D_m

        qmask = torch.FloatTensor([[1, 0], [0, 1], [1, 0]]).type(T1.type())
        qmask = qmask.unsqueeze(1).expand(-1, T1.size(1), -1)

        umask = torch.FloatTensor([[1, 1, 1]]).type(T1.type())
        umask = umask.expand(T1.size(1), -1)

        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)  # seq_len, batch, D_e
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        # emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        # print(emotions)
        emotions = self.dropout_rec(emotions)

        # emotions = emotions.unsqueeze(1)
        if att2:
            att_emotion, _ = self.matchatt(emotions, emotions[-1])
            hidden = F.relu(self.linear1(att_emotion))
        else:
            hidden = F.relu(self.linear1(emotions[-1]))
        # hidden = F.relu(self.linear2(hidden))
        # hidden = F.relu(self.linear3(hidden))
        # hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), -1)  # batch, n_classes
        return log_prob


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 110):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AcousticGenerator(nn.Module):
    """
    acoustic : (seq_len, batch_size, 100)
    fusion : (seq_len, batch_size, D_h)
    acoustic -> fusion
    """

    def __init__(self, D_h, dropout=0.2):
        super(AcousticGenerator, self).__init__()
        self.position_encoding = PositionalEncoding(100)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=8
        )
        self.fc1 = nn.Linear(100, 512)  # 尝试一下100->512->100
        self.fc2 = nn.Linear(512, D_h)

        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic):
        acoustic_fusion = self.position_encoding(acoustic)
        acoustic_fusion_transformered = self.gelu(
            self.transformer_encoder(acoustic_fusion)
        )
        acoustic_fusion = self.dropout(acoustic_fusion_transformered)
        acoustic_fusion = self.gelu(self.dropout(self.fc1(acoustic_fusion)))
        acoustic_fusion = self.gelu(self.dropout(self.fc2(acoustic_fusion)))
        # acoustic_fusion += acoustic_fusion_transformered

        return acoustic_fusion


class VisualGenerator(nn.Module):
    """
    visual : (seq_len, batch_size, 512)
    fusion : (seq_len, batch_size, D_h)
    visual -> fusion
    """

    def __init__(self, D_h, dropout=0.2):
        super(VisualGenerator, self).__init__()
        self.position_encoding = PositionalEncoding(512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=8
        )
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, D_h)

        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic):
        visual_fusion = self.position_encoding(acoustic)
        visual_fusion_transformered = self.gelu(self.transformer_encoder(visual_fusion))
        visual_fusion = self.dropout(visual_fusion_transformered)
        visual_fusion = self.gelu(self.dropout(self.fc1(visual_fusion)))
        # visual_fusion += visual_fusion_transformered
        visual_fusion = self.gelu(self.dropout(self.fc2(visual_fusion)))

        return visual_fusion


class TextGenerator(nn.Module):
    """
    text : (seq_len, batch_size, 100)
    fusion : (seq_len, batch_size, D_h)
    text -> fusion
    """

    def __init__(self, D_h, dropout=0.2):
        super(TextGenerator, self).__init__()
        self.position_encoding = PositionalEncoding(100)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=8
        )
        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, D_h)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic):
        text_fusion = self.position_encoding(acoustic)
        text_fusion_transformered = self.gelu(self.transformer_encoder(text_fusion))
        text_fusion = self.dropout(text_fusion_transformered)
        text_fusion = self.gelu(self.dropout(self.fc1(text_fusion)))
        text_fusion = self.gelu(self.dropout(self.fc2(text_fusion)))
        # text_fusion += text_fusion_transformered

        return text_fusion


class AcousticDiscriminator(nn.Module):
    """
    fusion : (seq_len, batch_size, D_h)
    prob : (seq_len, batch_size)
    fusion (from text and visual) -> prob
    """

    def __init__(self, D_h, dropout=0.2):
        super(AcousticDiscriminator, self).__init__()
        self.position_encoding = PositionalEncoding(D_h)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=D_h, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=8
        )
        self.fc1 = nn.Linear(D_h, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        # self.norm = nn.BatchNorm1d(32) # batch_size = 32
        self.dropout = nn.Dropout(dropout)

    def forward(self, acoustic_fusion):
        prob = self.position_encoding(acoustic_fusion)
        prob = self.transformer_encoder(prob)
        prob = self.gelu(prob)
        prob = self.gelu(self.dropout(self.fc1(prob)))
        prob = self.gelu(self.dropout(self.fc2(prob)))
        prob = self.sigmoid(self.dropout(self.fc3(prob)))
        return prob  # (seq_len, batch_size, 1)


class VisualDiscriminator(nn.Module):
    """
    fusion : (seq_len, batch_size, D_h)
    prob : (seq_len, batch_size)
    fusion (from text and acoustic) -> prob
    """

    def __init__(self, D_h, dropout=0.2):
        super(VisualDiscriminator, self).__init__()
        self.position_encoding = PositionalEncoding(D_h)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=D_h, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=8
        )
        self.object = nn.Linear(512, 100)  # 用来处理real_visual 输入为512个维度
        self.fc1 = nn.Linear(D_h, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        # self.norm = nn.BatchNorm1d(32) # batch_size = 32
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_fusion):
        if visual_fusion.size(-1) == 512:
            visual_fusion = self.object(visual_fusion)
        prob = self.position_encoding(visual_fusion)
        # print("visual_fusion.shape = ", visual_fusion.shape) # torch.Size([94, 32, 512])
        prob = self.transformer_encoder(prob)
        prob = self.gelu(prob)
        prob = self.gelu(self.dropout(self.fc1(prob)))
        prob = self.gelu(self.dropout(self.fc2(prob)))
        prob = self.sigmoid(self.dropout(self.fc3(prob)))
        return prob  # (seq_len, batch_size, 1)


class TextDiscriminator(nn.Module):
    """
    fusion : (seq_len, batch_size, D_h)
    prob : (seq_len, batch_size)
    fusion (from visual and acoustic) -> prob
    """

    def __init__(self, D_h, dropout=0.2):
        super(TextDiscriminator, self).__init__()
        self.position_encoding = PositionalEncoding(D_h)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=D_h, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=8
        )
        self.fc1 = nn.Linear(D_h, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        # self.norm = nn.BatchNorm1d(32) # batch_size = 32
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_fusion):
        prob = self.position_encoding(text_fusion)
        prob = self.transformer_encoder(prob)
        prob = self.gelu(prob)
        prob = self.gelu(self.dropout(self.fc1(prob)))
        prob = self.gelu(self.dropout(self.fc2(prob)))
        prob = self.sigmoid(self.dropout(self.fc3(prob)))
        return prob  # (seq_len, batch_size, 1)


"""
这是我的基于GAN的特征融合网络(GAN-Feature Fusion Network)
"""


class GAN_FFN(nn.Module):
    """
    acoustic_generator, visual_generator, text_generator are trained in GAN
    """

    def __init__(
        self,
        acoustic_generator: AcousticGenerator,
        visual_generator: VisualGenerator,
        text_generator: TextGenerator,
        n_classes=6,
        dropout=0.2,
    ):
        super(GAN_FFN, self).__init__()
        self.n_classes = n_classes

        self.acoustic_generator = acoustic_generator
        self.visual_generator = visual_generator
        self.text_generator = text_generator

        self.lstm = nn.LSTM(100, n_classes, bidirectional=False)
        # self.attention = nn.MultiheadAttention(embed_dim=100, num_heads=4, dropout=0.2)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(32*2, n_classes)

        self.fc = nn.Linear(100, n_classes)

    def forward(self, acoustic, visual, text):
        alpha, alpha_f, alpha_b = [], [], []

        # print("acoustic.size() = ", acoustic.size()) # torch.Size([94, 32, 200])
        # print("visual.size() = ", visual.size()) # torch.Size([94, 32, 2])
        # print("text.size() = ", text.size()) # torch.Size([32, 94])
        acoustic_fusion = self.acoustic_generator(acoustic)  # (seq_len, batch_size, D_h)
        visual_fusion = self.visual_generator(visual)  # (seq_len, batch_size, D_h)
        text_fusion = self.text_generator(text)  # (seq_len, batch_size, D_h)

        fusion = acoustic_fusion + visual_fusion + text_fusion  # 如果只用这个有59.14

        # attention_out, _ = self.attention(fusion, fusion, fusion)
        # attention_out = self.dropout(attention_out)
        fc_out = self.fc(fusion)
        log_prob = F.log_softmax(fc_out, 2)

        # fc_out = self.dropout(self.fc(fusion))
        # lstm_out, _ = self.lstm(fusion)
        # lstm_out = self.dropout(lstm_out)
        
        # fc_log_prob = F.log_softmax(fc_out, 2)
        # lstm_log_prob = F.log_softmax(lstm_out, 2)
        # log_prob = (fc_log_prob + lstm_log_prob) / 2

        # log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        # print("log_prob.size() = ", log_prob.size()) # torch.Size([94, 32, 6])

        return log_prob, alpha, alpha_f, alpha_b  # (所有句子去掉填充的总长度, n_classes)


class GAN_FFN_DialogueRNN(nn.Module):
    """
    acoustic_generator, visual_generator, text_generator are trained in GAN
    """

    def __init__(
        self,
        acoustic_generator: AcousticGenerator,
        visual_generator: VisualGenerator,
        text_generator: TextGenerator,
        D_m,
        D_g,
        D_p,
        D_e,
        D_h,
        D_a,
        n_classes,
        listener_state,
        context_attention,
        dropout_rec,
        dropout
    ):
        super(GAN_FFN_DialogueRNN, self).__init__()
        self.n_classes = n_classes

        self.acoustic_generator = acoustic_generator
        self.visual_generator = visual_generator
        self.text_generator = text_generator

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.bi_model = BiModel(
            D_m=D_m,
            D_g=D_g,
            D_p=D_p,
            D_e=D_e,
            D_h=D_h,
            n_classes=n_classes,
            listener_state=listener_state,
            context_attention=context_attention,
            D_a=D_a,
            dropout_rec=dropout_rec,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(100, n_classes)

    def forward(self, acoustic, visual, text, qmask, umask):
        alpha, alpha_f, alpha_b = [], [], []

        acoustic_fusion = self.acoustic_generator(
            acoustic
        )  # (seq_len, batch_size, D_h)
        visual_fusion = self.visual_generator(visual)  # (seq_len, batch_size, D_h)
        text_fusion = self.text_generator(text)  # (seq_len, batch_size, D_h)


        fusion = acoustic_fusion + visual_fusion + text_fusion  # 如果只用这个有59.14

        log_prob, alpha, alpha_f, alpha_b = self.bi_model(fusion, qmask, umask)

        return log_prob, alpha, alpha_f, alpha_b  # (所有句子去掉填充的总长度, n_classes)
