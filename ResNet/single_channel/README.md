# Single channel

This folder contains the scripts to train a CTC system on one of the channels in TIMIT.

# How to run

**Note:** Copy and replace the following files in mentioned locations in pip's installation directory:

1. ``CRDNN.py`` in ``speechbrain/lobes/models``

``Run single_channel_prepare.py in google colab``

``After running the above script run this command python train_env.py hparams/train_env.yaml --data_folder /path/to/TIMIT_4_channels/TIMIT_LA2``

# Results

| train file   | hyperparams file | Val. PER | Test PER | Model link    | GPUs        |
| ------------ | ---------------- | -------- | -------- | ------------- | ----------- |
| train_env.py | train_env.yaml   | 26.84    | 28.31    | Not available | 1xP100 16GB |

# Training time

