"""Fusion models combining the image and text branches.

- Late fusion: combine the two models' class scores.
- Early fusion (stacking): concatenate image + text features and train a classifier on top.
"""
