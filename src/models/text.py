"""Text model: BERT encoder followed by an LSTM and a classification head.

In:  token ids / attention mask (B, L)
Out: logits (B, 101); also exposes the pooled text features for early fusion.
"""
