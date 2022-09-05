# gradient-ascent-ai
Repo containing the code used to develop the "Business Names" EDA challenge.

# Requirements

- Python 3.9+
- [Pipenv](https://pypi.org/project/pipenv/)

# Setup

1. From the project directory, initialize the Pipenv virtual environment
```
pipenv shell
```

2. Install the jupyter notebooks kernel
```
pipenv run ipython kernel install --user --name=<KERNEL_NAME>
```
3. Download the required SpaCy pipelines

```
pipenv run python -m spacy download en_core_web_trf && pipenv run python -m spacy download en_core_web_sm
```
4. Launch the Jupyter server
```
pipenv jupyter notebook
```
5. From the browser window, double click on the `business_names_eda.ipynb` notebook located in the `src/notebooks` directory 