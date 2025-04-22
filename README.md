# Data-driven Marketing Project: Sentiment Analysis

## Deadline: 03/05/2025

## Requirements:
- Compare results between statistical models and LLMs
- Slides + Codes

## Goals:
- Compare performance between Lexicon-based tools, ML Models, DL Models and LLMs in 2 distinct datasets (book review and financial phrase)
- Compare performance of each model across 2 datasets to evaluate the impact of different data on the same model

## Setup:
- Download data from:
    + BookReview: [OneDrive Link](https://stneuedu-my.sharepoint.com/:f:/g/personal/11221106_st_neu_edu_vn/EjuHGwMvVxdFrNqojayLjhEBGbTNrFRfSgihvONdm91fZQ?e=lkD7qn)
    + FinancialPhrase: [OneDrive Link](https://stneuedu-my.sharepoint.com/:f:/g/personal/11221106_st_neu_edu_vn/EhkE1oHqgfJLp4GTxllBgbQBiEcv4bycqvS4-twJfHS_FA?e=bDjNM2)
- Clone the repo
- Paste datas into the folder `data/raw/*`
- In the main project folder, open Terminal in Admin mode:
    - Create a virtual environment: `python -m venv venv`
    - Activate the environment: `venv\Scripts\activate`
    - Install necessary libraries from requirements.txt file: `pip install -r requirements.txt`

## Models:
Feature Extractors are kept consistent across all ML models and DL models to ensure consistency when making comparision. Go to [feature_extractors.md](note/features_extractors.md) for more information.

### Lexicon-based tools:
- VADER
- SentiWordNet

### Machine Learning Models:
- TF-IDF (Unigrams + Bigrams) + **Naive Bayes** (generative classifier) (consider also using BoW)
- TF-IDF (Unigrams + Bigrams) + **Logistic Regression** (discriminative classifier)
- TF-IDF (Unigrams + Bigrams) + **Support Vector Machine** (linear classifier)
- TF-IDF (Unigrams + Bigrams) + **Random Forest** (tree-based classifier)
- TF-IDF (Unigrams + Bigrams) + **LightGBM** (improved tree-based classifier)

### Deep Learning Models:
From simple to complex combination of Feature Extractors and Models

#### Starter Pack
- **Averaged Static Embeddings (GloVe, Word2Vec, FastText) + Dense Network (MLP)** (Very simple DL. Loses all sequence information, often performs similarly to ML on aggregated embeddings. A small step up from ML.)

#### Mid level
- **Learned Embeddings + RNN** 
- **Learned Embeddings + 1D CNN** (Good at capturing local patterns. Often faster to train than RNNs. Different architectural concept)
- **Learned Embeddings + LSTM** (Introduces sequence modeling. Learns embeddings specific to your dataset, which can be good but requires sufficient data. Simple RNN architecture)
- **Learned Embeddings + BiLSTM** (Captures context from both directions, usually performs better than unidirectional LSTM/GRU. Slightly more complex)

#### Advanced Level
- **Pre-trained Static Embeddings + 1D CNN** (Leverages external knowledge and focuses on local patterns)
- **Pre-trained Static Embeddings + LSTM** (Leverages external knowledge from large corpora via embeddings. Often a strong baseline for basic DL. Requires handling OOV words if embeddings are fixed)
- **Pre-trained Static Embeddings + CNN-LSTM Hybrid** (Combines strengths of both architectures. More complex to implement)
    - *Sequential model*. 
    - 1D Convolutional layers are applied first to the sequence of word embeddings. The sequence of feature vectors output by the CNN layers is then fed into LSTM layers.

#### Hardcore vcl
- **Combine BiLSTM for Word Embeddings and CNN for Character Embeddings** (Feature fusion. Handles OOV words well. Significantly more complex architecture and input pipeline)
    - *Parallel model*
    - *Word Stream*: The sequence of word embeddings is fed into a BiLSTM (or GRU) to capture word-level semantic context and sequence dependencies.
    - *Character Stream*: The sequence of character embeddings (per word) is fed into a 1D CNN. The CNN identifies character n-gram patterns within each word (capturing prefixes, suffixes, morphology). The output of the CNN for each word is often pooled (e.g., MaxPooling) to get a fixed-size character-aware representation of that word. This results in a new sequence vector at the word level. (Sometimes another LSTM is applied after the character CNN).
    - *Fusion*: The final output vectors from the Word Stream (BiLSTM) and the Character Stream (CNN+Pooling) are concatenated.
    - *Classification*: This combined, richer vector is fed into Dense layers for classification.

### Pre-trained Transformer Models:
Use encoder-only model as decoder-only model (like GPT, Claude or Gemma) is more suitable for natural language generation, not understanding.

- **BERT/DistilBERT as Fixed Feature Extractor** (Uses the pre-trained model's knowledge without the complexity/cost of fine-tuning it. Often suboptimal performance compared to fine-tuning but conceptually simpler to integrate with ML pipelines)
- **DistilBERT Fine-tuning** (Standard entry point for transformer fine-tuning. Relatively lightweight)
- **BERT-base Fine-tuning** (Slightly larger and slower than DistilBERT, potentially slightly better performance. Still manageable)
- **RoBERTa-base / DeBERTa-v3-base Fine-tuning** (Often outperform BERT-base. Similar complexity level)
- **BERT-base / RoBERTa-base + LoRA Fine-tuning** (Introduces Parameter-Efficient Fine-Tuning (PEFT). Faster training, less memory required than full fine-tuning, often with comparable performance)
- **FinBERT** (Pre-trained model to analyze sentiment of financial text)