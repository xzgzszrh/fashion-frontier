"""Fashion Frontier -- end-to-end inference engineering on PYNQ-Z1.

Package layout mirrors the delivery pipeline:

    data      reproducible Fashion-MNIST splits
    models    teacher / TinyFashionCNN family / (optional) Brevitas QAT + BNN
    distill   soft targets and the two distillation losses
    engine    training loop, EMA, evaluation
    export    ONNX export with round-trip verification, static INT8 quantisation
    bench     host-side measurement and the ORT session factory
    board     PYNQ-Z1 board-side measurement (numpy + onnxruntime only)
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
