There are several .ipynb files:

Files beginning with TOOLS_ mean that these files are used for preprocessing purposes, whether it's used for data acquisition or data preprocessing. 
Other than that, it explains what method is used for that ipynb files:
a) EXTRATARGET_ files attempts to also predict realized volatility for all indices/commodity.
b) FE_ files attempt to use more Feature Engineering like momentum and real volatility for all indicies/commodity. 
c) NEW_ files attempt to use 1996-2025 dataset as a whole, still with train-val split.
d) OLD_ files attempt to use 1996-2020 dataset only, still with train-val split.
e) FULL_ files attempt to use 1996-2025 dataset as only training data, with cross-validation as a validation method.

eval files are the evaluation result of their respective file/methods

fed_speech_embeddings.npy is the result of FinBERT's inference on all of fed's speeches

folders beginning with models_ are the models saved by their respective files' names.

AS OF NOW, THE BEST METHOD IS BY USING THE FE_ FILES (FE_main.ipynb) AND FE_ MODELS (models_fe). ITS EVALUATION RESULT IS (FE_eval.txt). YOU SHOULD USE THESE. 
* don't forget to perform `pip install requirements.txt` for your venv (python 3.13.5 recommended).