# Project Structure - AspectVision

This document describes the directory structure of the AspectVision project.

```graphql
aspect-sentiment-analysis/
│── datasets/               # Dataset directory
│   ├── raw/                # Raw dataset (SemEval-2014, SemEval-2015, SemEval-2016, CR)
│   ├── processed/          # Preprocessed dataset (tokenized, cleaned)
│   ├── dataset_loader.py   # Script for loading and preprocessing data
│
│── models/                 # Model definitions
│   ├── ate_model.py        # Aspect Term Extraction model
│   ├── atsc_model.py       # Aspect Term Sentiment Classification model
│   ├── train.py            # Training script
│   ├── evaluate.py         # Model evaluation script
│
│── experiments/            # Experiment results and logs
│   ├── logs/               # Training logs
│   ├── results/            # Model performance results
│
│── notebooks/              # Jupyter Notebooks for EDA & experiments
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── train_ate.ipynb     # ATE model training notebook
│   ├── train_atsc.ipynb    # ATSC model training notebook
│
│── deployment/             # Deployment scripts
│   ├── app.py              # Streamlit web application
│   ├── requirements.txt    # Dependencies for deployment
│   ├── Dockerfile          # Docker configuration (if needed)
│
│── utils/                  # Utility functions
│   ├── preprocessing.py    # Text preprocessing functions
│   ├── metrics.py          # Evaluation metrics
│
│── config/                 # Configuration files
│   ├── train_config.json   # Training parameters
│   ├── model_config.json   # Model hyperparameters
│
│── tests/                  # Unit tests
│   ├── test_data.py        # Tests for data loading
│   ├── test_model.py       # Tests for model inference
│
│── README.md               # Project documentation
│── requirements.txt        # Python dependencies
│── setup.py                # Installation script
│── .gitignore              # Ignore unnecessary files
```
