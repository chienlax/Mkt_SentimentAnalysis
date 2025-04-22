# Feature Extractors

## 1. Single Feature Extractors

**I. Feature Extractors Primarily for Classical Machine Learning (ML) Models**

*(Goal: Convert variable-length text into a fixed-size numerical vector)*

1.  **Bag-of-Words (BoW) Variants:**
    *   **Binary Presence/Absence:** Vector indicates if a word from the vocabulary is present (1) or absent (0) in the document. Ignores frequency.
    *   **Raw Counts (CountVectorizer):** Vector indicates how many times each vocabulary word appears in the document.
    *   **Term Frequency (TF):** Normalized raw counts (e.g., divided by total words in the document) to reduce bias towards longer documents.

2.  **TF-IDF (Term Frequency-Inverse Document Frequency):**
    *   **Concept:** Weights words based on their frequency in a document (TF) but down-weights words common across all documents (IDF). Highlights terms important *to that specific document*. Generally performs better than simple counts.

3.  **N-Grams (Word or Character):**
    *   **Concept:** Sequences of N consecutive items (words or characters). Captures local context/phrases.
    *   **Implementation:** Can be used *in combination* with BoW or TF-IDF. For example, you can create vectors containing TF-IDF scores for both unigrams (single words) and bigrams (pairs of words).
        *   *Example:* `TfidfVectorizer(ngram_range=(1, 2))` creates features for single words and pairs.
    *   **Character N-grams:** Can capture subword information, useful for typos or morphology, but increases feature space significantly.

4.  **Aggregated Word Embeddings:**
    *   **Concept:** Use pre-trained static word embeddings (Word2Vec, GloVe, FastText). Since ML models need a fixed-size vector per document, you aggregate the vectors of all words in the document.
    *   **Aggregation Methods:**
        *   Average Pooling: Calculate the element-wise average of all word vectors in the document.
        *   Max Pooling: Take the element-wise maximum value across all word vectors.
        *   Sum Pooling: Element-wise sum of vectors.
    *   *(Note:* Often loses significant information compared to using embeddings sequentially in DL).

5.  **Linguistic / Metadata Features:**
    *   **Concept:** Explicitly calculate features based on linguistic properties or text statistics.
    *   **Examples:**
        *   PoS Tag Counts: Number/proportion of adjectives, adverbs, nouns, verbs.
        *   Sentiment Lexicon Scores: Features derived from lexicons like VADER's compound score, counts of positive/negative words from SentiWordNet or other lists.
        *   Text Statistics: Document length, average sentence length, average word length, count of punctuation marks (esp. '!', '?'), count of capitalized words/letters.
        *   Readability Scores: Flesch-Kincaid, Gunning Fog index, etc. (Less common for pure sentiment).
    *   **Implementation:** Calculated separately and *appended* as additional columns to other feature vectors (like TF-IDF).

6.  **Topic Model Features:**
    *   **Concept:** Represent documents based on the distribution of latent topics discovered using models like LDA (Latent Dirichlet Allocation).
    *   **Implementation:** The vector represents the probability distribution of the document across K topics. (Less direct for sentiment but captures themes).

7.  **Hashing Vectorizer:**
    *   **Concept:** Uses a hashing function to map features (words, n-grams) to a fixed number of columns. Memory efficient, avoids building a vocabulary, but can have collisions (different features map to the same column).

**II. Feature Representations / Inputs Primarily for Deep Learning (DL) Models**

*(Goal: Represent text as sequences of vectors or leverage internal model representations)*

1.  **Word Embeddings (as Sequences):**
    *   **Pre-trained Static Embeddings (GloVe, Word2Vec, FastText):** Each word is mapped to its vector, creating a sequence of vectors (e.g., `sequence_length x embedding_dimension`). This sequence is the input to CNN, RNN layers (usually via an `Embedding` layer).
    *   **Learned Embeddings (Trained from Scratch):** An `Embedding` layer is initialized randomly and learns representations specific to the task and dataset during DL model training.

2.  **Character Embeddings / Character-level Models:**
    *   **Concept:** Represent text as sequences of characters, each with an embedding. Often processed by CNNs or RNNs.
    *   **Benefit:** Handles out-of-vocabulary words, typos, and morphology naturally.

3.  **Subword Tokenization Embeddings (from Transformers):**
    *   **Concept:** Used by models like BERT, GPT, Phi-2. Text is tokenized into subword units (e.g., "embedding" -> "embed", "##ding") using WordPiece, BPE, or SentencePiece. Each subword token has an embedding.
    *   **Implementation:** The tokenizer converts text to input IDs, which the Transformer model uses to look up initial embeddings before processing through attention layers.

4.  **Contextual Embeddings (as Features):**
    *   **Concept:** Output representations from layers within pre-trained models like BERT or ELMo. These embeddings capture context.
    *   **Implementation (Less common now than fine-tuning):** Freeze the pre-trained model, pass text through it, extract hidden states from one or more layers, and use these (potentially aggregated) as input features for another model (could be ML or another DL model).

5.  **Fine-tuning Pre-trained Models (End-to-End):**
    *   **Concept:** The pre-trained Transformer model (e.g., BERT, Phi-2) itself acts as the feature extractor and classifier (with a classification head added). The model's internal layers learn task-specific features during fine-tuning. Raw text (tokenized appropriately) is the input. This is the dominant approach for state-of-the-art results.

## 2. Feature Fusion

The motivation is that different feature types capture different aspects of the text:

*   **Word Embeddings (GloVe, Word2Vec, FastText):** Capture semantic meaning and relationships between words based on large corpora.
*   **Contextual Embeddings (BERT, Phi-2):** Capture word meaning *in context*.
*   **TF-IDF/BoW:** Capture statistical importance of words within the specific dataset/document.
*   **Linguistic Features (PoS, Lexicon Scores):** Capture explicit grammatical information or pre-defined sentiment scores.
*   **Character Embeddings:** Capture subword information, helpful for morphology, typos, and OOV words.

Combining these can provide a richer, more comprehensive signal to the DL model, potentially leading to better performance, especially if one feature type compensates for the weaknesses of another.

**Common Combination Strategies:**

1.  **Concatenation:** The most common method. Features from different sources are computed and then concatenated into a single, larger feature vector before being fed into subsequent layers.
2.  **Separate Processing Streams:** Different feature types are processed through separate initial layers (e.g., separate LSTMs or CNNs), and their outputs are then combined (e.g., concatenated or added) before final classification.
3.  **Attention Mechanisms:** Use attention to allow the model to dynamically weigh the importance of different feature sources at different points in the sequence or before the final classification.

**Recommended Combinations for Basic DL Models (CNN/LSTM/GRU):**

1.  **Word Embeddings + Character Embeddings:**
    *   **Why:** Combines word-level semantics with subword information (robust to OOV words, typos).
    *   **How:** Typically uses two parallel input streams. One stream takes word embeddings (e.g., GloVe) into an LSTM/GRU. Another stream takes character embeddings into a CNN (to create a character-level word representation) or another LSTM/GRU. The outputs of these two streams are then concatenated and fed into dense layers for classification.
    *   **Tools:** Keras/PyTorch allows defining models with multiple inputs or parallel branches that merge later.

2.  **Word Embeddings + Explicit Linguistic Features (e.g., PoS Embeddings):**
    *   **Why:** Augments semantic word meaning with explicit grammatical information.
    *   **How (Word Level):** Create embeddings for PoS tags (e.g., map "NN", "JJ", "VB" to dense vectors). At each time step in the sequence, concatenate the word embedding (e.g., GloVe) with its corresponding PoS tag embedding. Feed this combined sequence into the LSTM/GRU/CNN.
    *   **How (Document Level - Simpler):** Calculate document-level linguistic features (e.g., count of adjectives, average sentence length). Process the word embedding sequence through your main LSTM/CNN. Concatenate the final output vector (e.g., the last hidden state or pooled output) with the separately calculated document-level linguistic features before the final dense classification layer.

3.  **Word Embeddings + Sentiment Lexicon Scores:**
    *   **Why:** Inject prior knowledge about word sentiment directly.
    *   **How (Word Level):** For each word, get its embedding (e.g., GloVe) and also look up its sentiment score(s) from a lexicon (e.g., VADER scores, SentiWordNet scores). Concatenate the word embedding with these numerical scores at each time step. Feed the combined sequence to the LSTM/GRU/CNN. *(Need to handle words not in the lexicon)*.
    *   **How (Document Level):** Similar to document-level linguistic features. Calculate an overall sentiment score for the document using a lexicon (e.g., average VADER compound score). Concatenate this score with the final output vector from the embedding-based sequence model before classification.

**Combinations Involving TF-IDF/BoW with DL:**

This is less common than combining different types of embeddings but possible:

4.  **Sequence Model Output + TF-IDF/BoW Document Vector:**
    *   **Why:** Combine the sequence understanding of LSTMs/CNNs with the global term importance statistics from TF-IDF.
    *   **How:** Process the word embedding sequence through your LSTM/CNN to get a final document representation (e.g., pooled output or last hidden state). Separately, calculate the TF-IDF vector for the same document. Concatenate these two vectors (one dense from DL, one sparse or dense from TF-IDF) before feeding them into the final dense classification layers.

**Combinations Involving Pre-trained Transformers (BERT, Phi-2, etc.):**

Fine-tuning the transformer is often the most powerful approach. However, you *can* augment its output:

5.  **Transformer Output Embedding + Document-Level Features:**
    *   **Why:** Leverage the powerful contextual embeddings from the transformer but add global document statistics or other explicit features.
    *   **How:** Obtain the final representation from the fine-tuned transformer (e.g., the [CLS] token embedding or pooled output of the last layer). Concatenate this vector with separately calculated document-level features like TF-IDF scores (perhaps top N scores), document length, or overall lexicon scores before the final classification head.

## List for Deep Learning

This progression considers factors like:

*   Model size and complexity
*   Reliance on pre-training
*   Need for fine-tuning
*   Feature representation complexity

**I. Basic Deep Learning Models (Often trained from scratch or using static embeddings)**

These models typically require you to define the architecture and handle embeddings explicitly.

1.  **Averaged Static Embeddings + Dense Network:**
    *   **Input:** Pre-trained static embeddings (GloVe, Word2Vec, FastText). Average the vectors of all words in a document/sentence.
    *   **Model:** Simple Multi-Layer Perceptron (MLP) / Feed-forward Dense Network.
    *   **Complexity/Why:** Very simple DL. Loses all sequence information, often performs similarly to ML on aggregated embeddings. A small step up from ML.

2.  **Learned Embeddings + LSTM / GRU:**
    *   **Input:** Raw text -> Tokenizer -> Embedding Layer (trained from scratch).
    *   **Model:** A single-layer Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU).
    *   **Complexity/Why:** Introduces sequence modeling. Learns embeddings specific to your dataset, which can be good but requires sufficient data. Simple RNN architecture.

3.  **Learned Embeddings + BiLSTM / BiGRU:**
    *   **Input:** Raw text -> Tokenizer -> Embedding Layer (trained from scratch).
    *   **Model:** Bidirectional LSTM or GRU (processes sequence forward and backward).
    *   **Complexity/Why:** Captures context from both directions, usually performs better than unidirectional LSTM/GRU. Slightly more complex.

4.  **Learned Embeddings + 1D CNN:**
    *   **Input:** Raw text -> Tokenizer -> Embedding Layer (trained from scratch).
    *   **Model:** 1D Convolutional layers (acting as n-gram detectors) followed by Pooling (e.g., MaxPooling).
    *   **Complexity/Why:** Good at capturing local patterns. Often faster to train than RNNs. Different architectural concept.

5.  **Pre-trained Static Embeddings + LSTM / GRU:**
    *   **Input:** Raw text -> Tokenizer -> Map tokens to pre-trained GloVe/Word2Vec/FastText vectors (often via an Embedding layer initialized with these weights).
    *   **Model:** LSTM or GRU.
    *   **Complexity/Why:** Leverages external knowledge from large corpora via embeddings. Often a strong baseline for basic DL. Requires handling OOV words if embeddings are fixed.

6.  **Pre-trained Static Embeddings + BiLSTM / BiGRU:**
    *   **Input:** Pre-trained GloVe/Word2Vec/FastText vectors.
    *   **Model:** Bidirectional LSTM or GRU.
    *   **Complexity/Why:** Combines external knowledge with bidirectional context. Often a very solid basic DL performer.

7.  **Pre-trained Static Embeddings + 1D CNN:**
    *   **Input:** Pre-trained GloVe/Word2Vec/FastText vectors.
    *   **Model:** 1D CNN + Pooling.
    *   **Complexity/Why:** Leverages external knowledge and focuses on local patterns.

8.  **Pre-trained Static Embeddings + CNN-LSTM Hybrid:**
    *   **Input:** Pre-trained GloVe/Word2Vec/FastText vectors.
    *   **Model:** CNN layers first (extract local features) followed by LSTM/GRU layers (model sequence of features).
    *   **Complexity/Why:** Combines strengths of both architectures. More complex to implement.

9.  **(Word Embeddings + Character Embeddings) + BiLSTM/CNN:**
    *   **Input:** Parallel inputs: Word embeddings (e.g., GloVe) and Character embeddings (learned or pre-trained).
    *   **Model:** Process each stream (e.g., BiLSTM for words, CNN for chars), concatenate outputs, then classify.
    *   **Complexity/Why:** Feature fusion. Handles OOV words well. Significantly more complex architecture and input pipeline.

**II. Pre-trained Transformer Models (Leveraging large models and fine-tuning)**

These models shift the focus from designing architectures from scratch to adapting large, pre-existing models.

10. **BERT/DistilBERT as Fixed Feature Extractor:**
    *   **Input:** Raw text -> BERT/DistilBERT Tokenizer -> Pre-trained BERT/DistilBERT model (frozen weights).
    *   **Model:** Extract fixed-size embeddings (e.g., the [CLS] token's final hidden state, or averaged hidden states) from the frozen BERT model. Feed these embeddings into a separate, simple classifier (e.g., Logistic Regression, Dense Network).
    *   **Complexity/Why:** Uses the pre-trained model's knowledge without the complexity/cost of fine-tuning it. Often suboptimal performance compared to fine-tuning but conceptually simpler to integrate with ML pipelines.

11. **DistilBERT Fine-tuning:**
    *   **Input:** Raw text -> DistilBERT Tokenizer -> DistilBERT model with a classification head.
    *   **Model:** DistilBERT (smaller, faster version of BERT).
    *   **Training:** Update (fine-tune) the weights of the pre-trained DistilBERT model on your specific sentiment dataset.
    *   **Complexity/Why:** Standard entry point for transformer fine-tuning. Relatively lightweight.

12. **BERT-base Fine-tuning:**
    *   **Input:** Raw text -> BERT Tokenizer -> BERT-base model with a classification head.
    *   **Model:** BERT-base (standard size).
    *   **Training:** Fine-tune the pre-trained BERT-base weights.
    *   **Complexity/Why:** Slightly larger and slower than DistilBERT, potentially slightly better performance. Still manageable.

13. **RoBERTa-base / DeBERTa-v3-base Fine-tuning:**
    *   **Input:** Raw text -> Respective Tokenizer -> Respective model with a classification head.
    *   **Model:** Improved variants of BERT.
    *   **Training:** Fine-tune the pre-trained weights.
    *   **Complexity/Why:** Often outperform BERT-base. Similar complexity level.

14. **BERT-base / RoBERTa-base + LoRA Fine-tuning:**
    *   **Input:** Raw text -> Respective Tokenizer -> Respective model configured with LoRA adapters.
    *   **Model:** Base model (BERT/RoBERTa) with Low-Rank Adaptation.
    *   **Training:** Freeze most pre-trained weights, only train the small LoRA adapter weights.
    *   **Complexity/Why:** Introduces Parameter-Efficient Fine-Tuning (PEFT). Faster training, less memory required than full fine-tuning, often with comparable performance.

15. **Phi-2 / OPT-2.7B + LoRA Fine-tuning:**
    *   **Input:** Raw text -> Respective Tokenizer -> Respective model (~3B params) configured with LoRA adapters.
    *   **Model:** Larger base model + LoRA.
    *   **Training:** PEFT on a more powerful base model.
    *   **Complexity/Why:** Steps up model size significantly. LoRA makes it feasible on moderate hardware (like your 12GB VRAM). Potential for much better performance than base models.

16. **Phi-2 / OPT-2.7B + QLoRA Fine-tuning:**
    *   **Input:** Raw text -> Respective Tokenizer -> Respective model loaded in 4-bit precision with LoRA adapters.
    *   **Model:** Larger base model + Quantization + LoRA.
    *   **Training:** PEFT with reduced memory footprint due to 4-bit loading.
    *   **Complexity/Why:** State-of-the-art efficient fine-tuning for larger models on consumer hardware. The standard way to fine-tune models of this size (and larger) on limited VRAM.

17. **Mistral-7B / Llama-2-7B + QLoRA Fine-tuning:**
    *   **Input:** Raw text -> Respective Tokenizer -> Respective model (~7B params) loaded in 4-bit with LoRA adapters.
    *   **Model:** Even larger base model + Quantization + LoRA.
    *   **Training:** Pushing the limits of consumer hardware fine-tuning.
    *   **Complexity/Why:** Represents the current upper end of what's often feasible without high-end enterprise GPUs. Potentially state-of-the-art performance. Requires careful setup and resource management.

18. **Fine-tuned Transformer Output + Other Features:**
    *   **Input:** Raw text processed by fine-tuned Transformer + separately calculated features (e.g., TF-IDF scores, document length).
    *   **Model:** Concatenate the Transformer's output embedding (e.g., [CLS] token) with the extra features before the final classification layer.
    *   **Complexity/Why:** Advanced feature fusion. Tries to combine learned contextual representations with explicit statistical or linguistic features. Adds complexity; effectiveness varies.