"""Mamba POC source package.

Layering rule (the seam): nothing in `data/`, `train/`, `serve/`, `eval/`, or
`conformance/` may import a hardware backend (`mlx`, `torch`/CUDA) directly.
All backend code lives behind `src.model.interface.ModelInterface`.
"""
