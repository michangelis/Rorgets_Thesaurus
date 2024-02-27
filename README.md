## Roget's Thesaurus Classification Analysis

### Overview
This project explores the application of machine learning techniques to classify words based on Roget's Thesaurus. It includes unsupervised clustering to evaluate the natural grouping of words into classes and sections, as well as supervised learning to predict the hierarchical category of a given word. The project employs word embeddings, dimensionality reduction, and classification algorithms to replicate and examine Roget's categorization.

### Setup Instructions

To set up this project, ensure that you have Python installed on your system. The project uses Poetry for dependency management.

1. Clone the repository to your local machine:

```
git clone https://github.com/michangelis/Rorgets_Thesaurus.git
```

2. Install Poetry using pip (if it's not already installed):

```
pip install poetry
```

3. Install the project dependencies using Poetry:

```
poetry install
```

### Running the Project

With all dependencies installed, activate the virtual environment:

```
poetry shell
```

### Important Notice
Please download the pre-computed embeddings file <a href="https://drive.google.com/drive/folders/1PM4JO7nDptahiYJIGhpO-ymdzaTe17mE?usp=sharing">here</a> before running the clustering and prediction scripts. This file is essential for the analysis, as it contains the word embeddings required for the machine learning models to function correctly.



