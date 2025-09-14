# HSE Basics of Deep Learning Antispoof Homework

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
</p>

## About

This repository contains the homework done during the HSE FCS 2025 summer mini-course dedicated to deep learning. The mini-course was based on LauzHack's summer deep learning bootcamp (https://github.com/LauzHack/deep-learning-bootcamp/tree/summer25). This homework is a voice antispoof detection neural network trained on ASVspoof 2019's Logical Access partition (https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset/data) [1]. Equal error rate (EER) was used as the model's performance metric. The goal was to drop the EER on the evaluation set below $5.3$. 

Link to the template used for this model's development: https://github.com/Blinorot/pytorch_project_template

## Methodology

The main architecture of this model is LightCNN (LCNN), described in paper [2].

The specific implementation of LCNN was taken from paper [3] with the addition of a dropout layer with $p = 0.1$
as in paper [5].

Before feeding an audiofile to the front-end, one of the 3 following augmentations was randomly applied to it upon being accessed from the dataset:
- The audiofile is unchanged.
- The volume is raised by $25%$.
- Random Gaussian noise is added to the audio.
These augmentations were taken from this notebook: https://colab.research.google.com/drive/1KgLPWtBVZVkPVsuaQjbwM-2phAUAjdD1#scrollTo=0ce54033-c77d-42f9-83e4-5b235f0fa78c

Transformation into a MEL spectrogram with parameters taken from paper [4] was used as the front-end:
- Sample rate: $16000$ Hz.
- Fast Fourier Transform size: $512$.
- Window length: $320$.
- Hop length: $160$.
- Number of mel filterbanks: $20$.

The resulting spectrogram was padded with zeros/trimmed to $600$ features after all the instance transforms, as described in paper [2]. After this, a small $\varepsilon$ value was added to the spectrogram and the logarithm of the spectrogram was taken.

The standard cross-entropy loss without modifications was used as the loss function.

The rest was taken from paper [4]:
- The model was trained on mini-batches of $64$ samples.
- Adam was used as the optimizer.
- The initial learning rate of $3 \cdot 10^{-4}$ was multiplied by $0.5$ every $10$ epochs.
- In total, the model was trained for $75$ epochs.
- The number $1$ was used as the random seed.

## Results

After 75 epochs, the EER equaled to 5.1424. As such, the homework was graded 10/10.

The Weights&Biases logs are available here: https://wandb.ai/roman_geek-hse-university/antispoof_homework/reports/Roman-Shamailov-s-antispoof-homework-logs--VmlldzoxMzkyMDcxMQ?accessToken=yrnjn2gcj0a9577o78hpw8r1v9wpfv63szhdjpj1e2ygg5oqq9i5judjqcljeevx


## References

[1] Yamagishi J. et al. Asvspoof 2019: Automatic speaker verification spoofing and
countermeasures challenge evaluation plan //ASV Spoof. – 2019. – Т. 13.

[2] Wu X. et al. A light CNN for deep face representation with noisy labels //IEEE transactions
on information forensics and security.
– 2018. – Т. 13. – №. 11. – С. 2884-2896.

[3] Lavrentyeva G. et al. STC antispoofing systems for the ASVspoof2019 challenge //arXiv
preprint arXiv:1904.05576. – 2019.

[4] Wang X., Yamagishi J. A comparative study on recent neural spoofing countermeasures for
synthetic speech detection //arXiv preprint arXiv:2103.11326. – 2021.

[5] Ma X. et al. Improved lightcnn with attention modules for asv spoofing detection //2021
IEEE International Conference on Multimedia and Expo (ICME). – IEEE Computer Society,
2021. – С. 1-6.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
