"""Generic train/eval loop for any model (image / text / fusion), driven by a YAML config.

Uses the validation split for LR drops and early stopping; the test set is only used
for the final reported numbers. Sets seeds and logs config + metrics to results/.
"""
