"""Redes recurrentes y cabezas opcionales, inicializadas desde cero."""
import torch
from torch import nn


class PriceNetwork(nn.Module):
    def __init__(self, architecture, input_size, settings, output_size):
        super().__init__()
        self.pooling = settings["pooling"]
        recurrent_dropout = settings.get("recurrent_dropout", settings["dropout"]) if settings["num_layers"] > 1 else 0.
        self.recurrent = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[architecture](
            input_size=input_size, hidden_size=settings["hidden_size"],
            num_layers=settings["num_layers"], batch_first=True, dropout=recurrent_dropout)
        # Dropout explícito: el interno de PyTorch no actúa con una sola capa.
        self.dropout = nn.Dropout(settings["dropout"])
        state_size = settings["hidden_size"] * (2 if self.pooling == "last_mean" else 1)
        head_size = settings.get("head_size", 0)
        self.head = (nn.Sequential(nn.Linear(state_size, head_size), nn.ReLU(), nn.Dropout(settings["dropout"]))
                     if head_size else nn.Identity())
        self.output = nn.Linear(head_size or state_size, output_size)
        # Predicción residual: el punto de partida es el último Brent conocido.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values):
        sequence, _ = self.recurrent(values)
        if self.pooling == "last":
            state = sequence[:, -1]
        elif self.pooling == "max":
            state = torch.amax(sequence, dim=1)
        elif self.pooling == "last_mean":
            state = torch.cat([sequence[:, -1], sequence.mean(dim=1)], dim=1)
        else:
            raise ValueError(f"Agregación temporal desconocida: {self.pooling}")
        return self.output(self.head(self.dropout(state)))
