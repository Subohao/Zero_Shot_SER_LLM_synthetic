# Zero_Shot_SER_LLM_synthetic

<p align="center">
  <img src=https://github.com/user-attachments/assets/f2603652-d4b1-4466-82b1-cfbc203b2cae />
<p>

## Abstract

In generalized Speech Emotion Recognition (SER), traditional generalization techniques like transfer learning and domain adaptation rely on access to some amount of unlabeled target domain data. However, with increasing privacy concerns, building SER systems under **zero-shot** scenarios, where no target domain data is available, poses a significant challenge. In such cases, conventional methods become impractical without access to target samples or features. To leverage any available target information to bridge this gap, this work explores the potential of Large Language Models (LLMs), with their powerful generative capabilities, to generate target corpora based on documented scenario settings and published research, enabling SER under zero-shot conditions.
We assess the effectiveness of LLMs in SER tasks across both text and speech modalities under challenging zero-shot conditions, using IEMOCAP and MSP-PODCAST as unseen target corpora. To ensure a fair comparison, we validate the performance of the synthesized data against real source data from MELD and MSP-IMPROV. Our experimental results reveal that, on average, the synthetic data not only matches but often surpasses the performance of real data in both text and speech modalities.

## Prompting Engineering

- Prompt LLMs with viable information from released documents (i.e., papers, website descriptions) of target corpora

```
python3 gen_zero_shot_iemocap_json.py
```

## Neutral Speech Synthesis

- Convert generated content from LLMs to neutral speech through TTS

```
python3 coqui_TTS.py
```

## Emotion Injection

- Please refer to Chou's paper for a detailed implementation and model description

## Zero-shot Text Emotion Recognition

```
python3 ./text_modality/train_text.py
```

## Zero-shot Speech Emotion Recognition

- The overall zero-shot SER experiments are conducted under s3prl infrastructure, please well install it before you run the code
- s3prl installation reference: https://github.com/s3prl/s3prl
- create a `zero_shot_emotion` folder under `./s3prl/s3prl/downstream`
- clone all files in `speech_modality` to `./s3prl/s3prl/downstream/zero_shot_emotion`

```
cd ./s3prl/s3prl
python3 run_downstream.py -m train -n $EXP_NAME -u data2vec -d zero_shot_emotion -c downstream/zero_shot_emotion/config.yaml
```
