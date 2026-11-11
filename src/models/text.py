"""Text model: BERT -> LSTM(128) -> Dropout -> Dense(256)+ReLU -> Dropout -> Dense(101) (Table II).

In:  input_ids (B, L), attention_mask (B, L) from src/data.py
Out: logits (B, 101); extract_features gives the 128-d LSTM features for early fusion.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoModel, AutoTokenizer


class TextClassifier(nn.Module):
    """BERT token embeddings, read by an LSTM that skips padding, then a 2-layer MLP head."""

    def __init__(
        self,
        bert_name: str,
        num_classes: int,
        lstm_hidden: int,
        hidden_dim: int,
        dropout: float,
        freeze_bert: bool = True,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)       # bert-base-uncased: 12 layers, 768-d
        self.freeze_bert = freeze_bert
        if freeze_bert:
            self.bert.requires_grad_(False)

        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden, batch_first=True)  # 768 -> 128
        self.feature_dim = lstm_hidden                          # 128, used by fusion.py

        # --- Table II head ---
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, hidden_dim),                 # (B, 128) -> (B, 256)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),                 # (B, 256) -> (B, 101)
        )

    def train(self, mode: bool = True) -> "TextClassifier":
        """Like nn.Module.train, but a frozen BERT always stays in eval mode (its dropout off)."""
        super().train(mode)
        if self.freeze_bert:
            self.bert.eval()
        return self

    def _bert_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L), (B, L) -> BERT last_hidden_state (B, L, 768)."""
        # No graph is stored for a frozen BERT: saves memory and time.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_bert):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state                            # (B, L, 768)

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L), (B, L) -> LSTM hidden state at each title's last real token (B, 128)."""
        states = self._bert_states(input_ids, attention_mask)  # (B, L, 768)
        lengths = attention_mask.sum(dim=1).cpu()              # (B,) real tokens per title, incl. [CLS]/[SEP]
        packed = pack_padded_sequence(states, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)                        # h_n: (num_layers=1, B, 128)
        return h_n[-1]                                         # (B, 128), back in the input order

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """LSTM features (B, 128) -> logits (B, 101) through the Table II head."""
        return self.head(features)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L), (B, L) -> logits (B, 101)."""
        return self.classify(self.extract_features(input_ids, attention_mask))


def build_text_model(config: dict) -> TextClassifier:
    """Build the TextClassifier from data.yaml + text.yaml (merged, flat)."""
    return TextClassifier(
        bert_name=config["tokenizer_name"],
        num_classes=config["num_classes"],
        lstm_hidden=config["lstm_hidden"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        freeze_bert=config["freeze_bert"],
    )


def shape_check(config: dict) -> None:
    """Dummy batch through the model: print shapes, and check that padding does not change features."""
    model = build_text_model(config).eval()
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TextClassifier: {n_total / 1e6:.1f}M parameters, {n_train / 1e6:.2f}M trainable")

    titles = ["goat cheese & strawberry bruschetta | food republic", "apple pie"]
    batch = tokenizer(titles, padding="max_length", truncation=True,
                      max_length=config["max_length"], return_tensors="pt")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    with torch.no_grad():
        print("input_ids          ", tuple(ids.shape), "| real lengths", mask.sum(1).tolist())
        print("last_hidden_state  ", tuple(model._bert_states(ids, mask).shape))
        print("features           ", tuple(model.extract_features(ids, mask).shape))
        print("logits             ", tuple(model(ids, mask).shape))

        # Packing check: the same title padded to 64 tokens vs. not padded at all.
        short = tokenizer(titles[1:], return_tensors="pt")     # (1, 4): no padding
        f_padded = model.extract_features(ids[1:], mask[1:])
        f_short = model.extract_features(short["input_ids"], short["attention_mask"])
        diff = (f_padded - f_short).abs().max().item()
        print(f"padding check: max |features(padded to 64) - features(unpadded)| = {diff:.2e}")
