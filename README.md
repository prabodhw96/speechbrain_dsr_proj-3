# Multi-microphone Signal Processing and Speech Recognition
## Overview
Multi-microphone speech recognition is one of the most challenging problems in spoken language understanding. Previously, speech scientists used to rely on traditional methodologies of
computing and preprocessing the features, and then feeding it to the recognition models. In this project, we did experiments with 3 models, namely, CRDNN (a combination of Convolutional, Recurrent, and Fully
connected networks), DenseNet and ResNet in combination with signal processing techniques
such as beamforming for preprocessing the multi-microphone signal.

 TIMIT Multi-mic dataset which is a modified version of the standard dataset called TIMIT Acoustic-Phonetic Continuous Speech Corpus was used for this project.
 
## Training the models
The instruction to run the experiments can be found in the README.md of the folders.

## Results
https://share.streamlit.io/prabodhw96/log_streamlit/app.py

Metric: Phoneme Error Rate (PER)


## About Speechbrain
* Website: https://speechbrain.github.io/
* Code: https://github.com/speechbrain/speechbrain/

## Citing Speechbrain
Please cite SpeechBrain if you use it for your research or business.
```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```