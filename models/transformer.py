import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 num_output_queries, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.positional_encoding = nn.parameter.Parameter(torch.randn(1, num_tokens, d_model))
        self.input_query = nn.Embedding(num_output_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, x):
        encoder_out = self.encoder(x + self.positional_encoding)
        decoder_out = self.decoder(self.input_query.weight.unsqueeze(0).repeat(encoder_out.shape[0], 1, 1), encoder_out)
        return decoder_out

if __name__ == '__main__':
    model = Transformer(
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