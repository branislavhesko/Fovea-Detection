from math import sqrt

import einops
import torch
import torch.nn as nn


class CoarseTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 num_output_queries, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.positional_encoding = nn.parameter.Parameter(torch.randn(1, num_tokens, d_model))
        self.input_query = nn.Embedding(num_output_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.size = int(sqrt(num_output_queries - 2))

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        encoder_out = self.encoder(x + self.positional_encoding)
        decoder_out = self.decoder(self.input_query.weight.unsqueeze(0).repeat(encoder_out.shape[0], 1, 1), encoder_out)
        position_queries = decoder_out[:, :2, :]
        mask_queries = einops.rearrange(decoder_out[:, 2:, :], "b (h w) d -> b d h w", h=16, w=16)
        return position_queries, mask_queries



class FineTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_encoder_layers,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.positional_encoding = nn.parameter.Parameter(torch.randn(1, num_tokens, d_model))

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        encoder_out = self.encoder(x + self.positional_encoding)
        return einops.rearrange(encoder_out, "b (h w) c -> b c h w", h=17, w=17)


if __name__ == '__main__':
    model = CoarseTransformer(
        num_tokens=100,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_output_queries=258
    )
    x = torch.randn(10, 100, 512)
    y = model(x)
    print(y.shape)