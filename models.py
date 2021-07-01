import torch.nn as nn
from layers import CNN, Encoder, RNNAttnDecoder


class CNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super(CNNSeq2Seq, self).__init__()

        self.backbone = CNN()
        self.encoder = Encoder()
        self.decoder = RNNAttnDecoder(vocab_size=vocab_size)

    def forward(self, x, y, is_training):
        out = self.backbone(x)
        encoder_outputs, encoder_hidden = self.encoder(out)

        vocab_out = self.decoder(y, encoder_outputs, encoder_hidden, is_training)

        return vocab_out

    def forward_2(self, x, max_len):
        out = self.backbone(x)
        encoder_outputs, encoder_hidden = self.encoder(out)

        vocab_out = self.decoder.forward_2(encoder_outputs, encoder_hidden, max_len)

        return vocab_out
