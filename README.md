# DLNLP_24_SN23187901

This project implements a sentiment analysis application using the BERT (Bidirectional Encoder Representations from Transformers) model to classify movie reviews from the IMDB dataset as positive or negative.

## Project Overview

The application uses the PyTorch library along with Hugging Face's Transformers to fine-tune a pre-trained BERT model for the task of sentiment analysis. It demonstrates the process of loading, cleaning, tokenizing data, and training the model using best practices in modern NLP.

## Features

- Load and preprocess the IMDB movie review dataset.
- Use BERT tokenizer for data preparation.
- Train and evaluate the BERT model for sequence classification.
- Utilize GPU acceleration for efficient training.
- Plot results and analyze the model's performance.

## Installation

To set up your development environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/penggang0719/DLNLP_24_SN23187901.git

2. Install the required dependencies:
   ```bash
   conda env create -f environment.yml

## Usage
To run the sentiment analysis:
python main.py

## Structure
- Datasets/: Directory for the IMDB dataset.
- A/fine_tuned_bert: Saved fine-tuned BERT models.
- A/Preprocessed_Data: Saved preprocessed data.
- A/imdb.py: Saved funtions for data preprocessing, traing and analyzing.
- main.py/: Python scripts for running the model.
- train.ipynb/: Jupyter notebooks for exploratory data analysis and results visualization.

## Acknowledgments

- Thanks to Hugging Face for providing the Transformers library.
   ```bash
   @article{DBLP:journals/corr/abs-1810-04805,
     author    = {Jacob Devlin and
                  Ming{-}Wei Chang and
                  Kenton Lee and
                  Kristina Toutanova},
     title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
                  Understanding},
     journal   = {CoRR},
     volume    = {abs/1810.04805},
     year      = {2018},
     url       = {http://arxiv.org/abs/1810.04805},
     archivePrefix = {arXiv},
     eprint    = {1810.04805},
     timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
     biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
     bibsource = {dblp computer science bibliography, https://dblp.org}
   }

- Thanks to IMDB datasets
    ```bash
   @InProceedings{maas-EtAl:2011:ACL-HLT2011,
     author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
     title     = {Learning Word Vectors for Sentiment Analysis},
     booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
     month     = {June},
     year      = {2011},
     address   = {Portland, Oregon, USA},
     publisher = {Association for Computational Linguistics},
     pages     = {142--150},
     url       = {http://www.aclweb.org/anthology/P11-1015}
   }
