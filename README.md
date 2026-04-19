# NLP Author Identification System 📚🤖

An end-to-end Natural Language Processing pipeline built from scratch to classify 19th-century literature. This project compares traditional statistical NLP techniques against Deep Learning architectures to predict the author of a given text.

## 🧠 System Architecture

This project strictly avoids high-level machine learning libraries (like `scikit-learn`) for core algorithm implementation to demonstrate a deep mathematical understanding of vector spaces and probability models.

1. **Data Pipeline:** Automated fetching, tokenization, and stratified splitting of Gutenberg Project books (Mark Twain, Jane Austen, Arthur Conan Doyle, H.G. Wells) handling over 800+ text samples.
2. **Model 1: Vector Space Model (VSM):** A custom TF-IDF implementation utilizing Cosine Similarity and Centroid calculation.
3. **Model 2: N-Gram Language Model:** A probabilistic Bigram model implementing Laplace (Add-1) Smoothing to handle unseen vocabulary sequences.
4. **Model 3: Multi-Layer Perceptron (MLP):** A deep neural network built with PyTorch, illustrating the effects of overfitting on small datasets compared to statistical methods.

## 📊 Performance & Results

The models were evaluated on a perfectly balanced test set (160 documents).

| Model | Technique | Accuracy |
| :--- | :--- | :--- |
| **N-Gram Model** | Probabilistic (Bigram) | **98.75%** 🏆 |
| **VSM** | Mathematical (TF-IDF / Cosine) | **96.88%** |
| **MLP (PyTorch)** | Deep Learning (Feed-Forward) | **79.38%** |

*Insight:* The Deep Learning model (MLP) suffered from overfitting due to the small corpus size (Training Loss dropped to 0.000), proving that for constrained text datasets, structural probabilistic models (N-Grams) significantly outperform standard neural networks.

## 🛠️ Tech Stack
* **Python 3.x**
* **PyTorch** (For Deep Learning implementation)
* **NLTK** (For initial text tokenization only)
* **NumPy** (For matrix operations)
