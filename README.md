# UPMC Food-101: image + text fusion (PyTorch reproduction)

Reproduction of Gallo et al., IVCNZ 2020, "Image and Text fusion for UPMC Food-101 using BERT and CNNs".

```
src/
  data.py          # Dataset, transforms, tokenization
  models/
    image.py       # InceptionV3 classifier
    text.py        # BERT + LSTM classifier
    fusion.py      # late + early fusion heads
  train.py         # generic train/eval loop
  extract.py       # feature caching (phase 4)
modal_app.py       # Modal image, volumes, GPU entrypoints
configs/           # one YAML per experiment (lr, epochs, batch size…)
notebooks/         # EDA + error analysis only
results/           # metrics JSON/CSV, plots (small files only)
```
