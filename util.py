import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy
import torch.nn.functional as F


class Seq2SeqLoss(nn.Module):
    def __init__(self):
        super(Seq2SeqLoss, self).__init__()
        self.criterion = LabelSmoothingCrossEntropy()

    def forward(self, outputs, y):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        y: [batch_size, max_len]
        """
        max_len = y.size(1)
        return sum([self.criterion(outputs[:, i, :], y[:, i + 1]) for i in range(max_len - 1)]) / (max_len - 1)


class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs, y):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        y: [batch_size, max_len]
        """
        max_len = y.size(1)
        return sum([self.criterion(outputs[:, i, :], y[:, i + 1]) for i in range(max_len - 1)])


class ConsistentLoss(nn.Module):
    def __init__(self):
        super(ConsistentLoss, self).__init__()

    def forward(self, outputs, outputs_ema):
        """
        outputs: [batch_size, max_len-1, vocab_size]
        outputs_ema: [batch_size, max_len-1, vocab_size]
        """
        batch_size = outputs.size(0)
        max_len = outputs.size(1) + 1
        num_classes = outputs.size(2)

        loss = 0
        for i in range(max_len-1):
            input_logits = outputs[:, i, :]
            target_logits = outputs_ema[:, i, :]
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            loss += F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

        return loss / batch_size


def compute_seq_acc(outputs, y, batch_size, max_len):
    """
    outputs: [batch_size, max_len-1, vocab_size]
    y: [batch_size, max_len]
    """
    num_eq = (y[:, 1:].data == outputs.max(2)[1]).sum(dim=1)
    accuracy_clevel = num_eq.sum() / batch_size / (max_len - 1)
    accuracy_all = (num_eq == max_len - 1).sum() / batch_size
    return accuracy_clevel, accuracy_all
