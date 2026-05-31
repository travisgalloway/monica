"""Model package: the hardware seam.

`interface` and `blocks` are portable (no backend imports). `mlx_backend`,
`mlx_train_step`, and `cuda_backend` are the only modules permitted to import a
hardware library.
"""
