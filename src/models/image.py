"""Image model: InceptionV3 (ImageNet-pretrained) fine-tuned on Food-101.

In:  images (B, 3, 299, 299)
Out: logits (B, 101); also exposes the penultimate features for early fusion.
"""
