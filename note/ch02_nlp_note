## 2.4 Simple Unix Tools for Word Tokenization
- Before almost any natural language processing of a text, the text has to be normalized, a task called text normalization.
    + Tokenizing (segmenting) words
    + Normalizing word formats
    + Segmenting sentences
- If somewhat naive version of word tokenization and normalization (and frequency computation) that can be accomplished for English solely in a single Unix command-line. Unix tools of this sort can be very handy in building quick word count statistics for any corpus in English.

## 2.5 Word and Subword Tokenization
- There are roughly two classes of tokenization algorithms. In top-down tokenization, we define a standard and implement rules to implement that kind of tokenization. But more commonly instead of using words as the input to NLP algorithms we break up words into subword tokens, which can be words or parts of words or even individual letters. These are derived via bottom-up tokenization, in which we use simple statistics of letter sequences to come up with the vocabulary of subword tokens, and break up the input into those subwords.

### 2.5.1 Top-down (rule-based) tokenization
- Defining a Standard: You first decide on a set of rules for what constitutes a token (e.g., separate punctuation, keep hyphens in certain words, handle contractions). E.g: Penn Treebank tokenization standard as a well-known example of such a standard.
- Implementing Rules: You then write and implement a set of explicit rules (often using tools like regular expressions) to convert the raw text according to your predefined standard.


### 2.5.2 Byte-PairEncoding: A Bottom-up Tokenization Algorithm
- Instead of defining tokens as words (whether delimited by spaces or more complex algorithms), or as characters (as in Chinese), we can use our data to automatically tell us what the tokens should be. This is especially useful in dealing with unknown words, an important problem in language processing.
- NLP algorithms often learn some facts about language from one corpus (a training corpus) and then use these facts to make decisions about a separate test corpus and its language.
- To deal with unknown word problem, modern tokenizers automatically induce sets of tokens that include tokens smaller than words, called subwords. Sub
words can be arbitrary substrings, or they can be meaning-bearing units like the morphemes-est or-er. (A morpheme is the smallest meaning-bearing unit of a language; for example the word unwashable has the morphemes un-, wash, and-able.) In modern tokenization schemes, most tokens are words, but some tokens are frequently occurring morphemes or other subwords like-er. Every unseen word like lower can thus be represented by some sequence of known subword units, such as *low* and *er*, or even as a sequence of individual letters if necessary.
- Most tokenization schemes have two parts: a token learner, and a token segmenter. 
    + The token learner takes a raw training corpus (sometimes roughly pre-separated into words, for example by whitespace) and induces a vocabulary, a set of tokens. 
    + The token segmenter takes a raw test sentence and segments it into the tokens in the vocabulary.
- Two algorithms are widely used: byte-pair encoding, and unigram language modeling. There is also a SentencePiece library that includes implementations of both of these, and people often use the name SentencePiece to simply mean unigram language modeling tokenization.

#### Byte-Pair encoding (BPE algorithm) 

```
function BYTE-PAIR ENCODING(strings C, number of merges k) returns vocab V

    V <- all unique characters in C         # initial set of tokens is characters
    for i = 1 to k do                       # merge tokens k times
        tL, tR <- Most frequent pair of adjacent tokens in C
        tNEW <- tL + tR                     # make new token by concatenating
        V <- V + tNEW                       # update the vocabulary
        Replace each occurrence of tL, tR in C with tNEW # and update the corpus
    return V
```

## 2.6 Word Normalization

*   **Definition:** The overall task of putting words/tokens into a standard format.
*   **Case Folding:**
    *   The simplest form of normalization.
    *   Involves converting all text to a single case (usually lowercase).
    *   **Benefit:** Helps treat words like `Woodchuck` and `woodchuck` identically, useful for generalization in tasks like Information Retrieval or Speech Recognition.
    *   **Drawback:** Can lose important information (e.g., distinguishing `US` the country from `us` the pronoun). Often *not* done for tasks like Sentiment Analysis, Information Extraction, or Machine Translation where case can be meaningful.
    *   Modern tokenization (like BPE) might handle some normalization implicitly, but explicit steps are often still considered.

### 2.6.1 Lemmatization

*   **Definition:** The task of grouping together different inflected forms of a word so they can be analyzed as a single root form, known as the **lemma**.
    *   Examples: `am`, `are`, `is` have the lemma `be`; `cat` and `cats` have the lemma `cat`.
    *   The full inflected form (`cats`) is the **wordform**.
*   **Purpose:** Essential for understanding that different wordforms share the same core meaning. Particularly important for morphologically complex languages (like Polish).
*   **Method:** Sophisticated methods involve **morphological parsing** – breaking a word into its **stem** (the core meaning-bearing unit) and **affixes** (additional meaning units).

### Stemming

*   **Definition:** A simpler, often cruder, variant of lemmatization.
*   **Method:** Primarily involves chopping off word-final affixes (suffixes) using heuristic rules.
*   **Example:** The **Porter Stemmer** is a classic example, which applies rules iteratively to strip common suffixes.
*   **Limitations:**
    *   Less accurate than full lemmatization.
    *   Can **over-generalize** (e.g., mapping `policy` to `polic`) or **under-generalize** (e.g., not mapping `European` to `Europe`).
    *   Less common in modern NLP systems but can still be useful in specific scenarios where speed/simplicity are prioritized over accuracy.

## 2.7 Sentence Segmentation

*   **Task:** Sentence segmentation is the process of dividing a text into its individual sentences.
*   **Importance:** It's a fundamental preliminary step for many NLP tasks.
*   **Cues:** Primarily relies on punctuation like periods (`.`), question marks (`?`), and exclamation points (`!`).
    *   `?` and `!` are relatively unambiguous sentence boundaries.
*   **Challenge: Period Ambiguity:**
    *   The period (`.`) is ambiguous. It can mark the end of a sentence *or* be part of a word (like an abbreviation, e.g., `Mr.`, `Inc.`).
    *   A single period can even perform both functions simultaneously (e.g., `Inc.` at the end of a sentence).
*   **Methods:**
    *   Algorithms must decide whether a period belongs to a word or marks a sentence boundary.
    *   This often involves using dictionaries of abbreviations (hand-built or learned).
    *   Machine learning classifiers or rule-based systems can be used.
    *   Sometimes performed jointly with word tokenization.
    *   The Stanford CoreNLP toolkit uses a rule-based, deterministic approach tied to its tokenization process.

## 2.8 Minimum Edit Distance

*   **Purpose:** Measures the similarity between two strings by quantifying how different they are.
*   **Applications:** Crucial in various NLP tasks:
    *   **Spelling Correction:** Finding dictionary words similar to a misspelled word.
    *   **Coreference Resolution:** Comparing name strings to see if they might refer to the same entity.
    *   **Speech Recognition Evaluation:** Measuring the difference between a system's transcription hypothesis and a correct reference transcript (Word Error Rate).
*   **Definition:** The minimum number of basic **edit operations** (insertion, deletion, substitution) required to transform one string into the other.
*   **Levenshtein Distance:**
    *   The most common weighting scheme.
    *   Assigns a cost of `1` to each operation (insertion, deletion, substitution).
    *   Substitution of a character for itself has a cost of `0`.
    *   An alternative version assigns a cost of `2` to substitution (treating it as one deletion and one insertion).
*   **Alignment:**
    *   A key way to visualize the edit process.
    *   Shows a correspondence between characters/substrings of the two strings.
    *   An **operation list** (using symbols like 'd', 's', 'i') describes the sequence of edits.
*   **Algorithm (Minimum Edit Distance Algorithm):**
    *   The space of all possible edit sequences is too large to search naively.
    *   **Dynamic Programming** is used to efficiently compute the minimum distance.
    *   It works by building up a table (distance matrix) storing the minimum edit distance between prefixes of the two strings.
    *   Each cell `D[i, j]` is computed based on the minimum cost path from three neighboring cells, corresponding to the three possible operations (insertion, deletion, substitution).
*   **Computing Alignments:**
    *   By storing **backpointers** in the dynamic programming table that indicate which path was taken to reach the minimum cost for each cell, the actual minimum cost alignment (sequence of edits) can be reconstructed.