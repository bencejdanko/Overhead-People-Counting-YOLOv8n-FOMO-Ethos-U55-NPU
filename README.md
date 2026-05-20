```
fomo-overhead-deploy/
├── .gitignore
├── requirements.txt
├── setup.py
├── fomo_core/
│   ├── __init__.py
│   ├── model.py          # Network definition & compilation (Keras)
│   ├── annotation.py     # Offline SAM mask/centroid extraction (PyTorch)
│   ├── pipeline.py       # tf.data dataset generation & augmentations
│   └── quantization.py   # TFLite INT8 conversion & representative dataset
└── configs/
    └── default_vela.ini  # Vela hardware configuration file
```