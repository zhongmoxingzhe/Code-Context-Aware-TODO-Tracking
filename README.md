## Environment Dependencies
```
beautifulsoup4==4.11.1
GitPython==3.1.37
nbformat==5.9.2
nltk==3.6.7
numpy==1.21.5
openai==0.27.8
openpyxl==3.0.10
pandas==1.5.0
Pygments==2.13.0
Requests==2.31.0
scikit_learn==1.0.2
scipy==1.5.3
timeout_decorator==0.5.0
torch==1.12.1+cu116
tqdm==4.64.1
transformers==4.25.1
```
This project uses the dataset provided in the project available at https://doi.org/10.5281/zenodo.5402956
Datasets Publicly Available in folder **top_repos_10000**:
- **new_java** and **new_python**: Intra-project Datasets based on Java and Python languages, respectively.
- **sbp_java** and **sbp_python**: Inter-project Datasets based on Java and Python languages, respectively.

- You need to create a **Model** folder yourself to store the pre-trained CodeBERT model.

- #### Train
1.To train TODO_Checker on Intra-project Datasets:
```bash
python3 codebert_train.py
```
If you want to train TODO_Checker on different program language dataset, you need to change the `dir_path` in the codebert_train.py file.

2.To train TODO_Checker on Inter-project Datasets:
```bash
python3 codebert_train_sbp.py
```
If you want to train TODO_Checker on different program language dataset, you need to change the `dir_path` in the codebert_train_sbp.py file.

#### Inference
```bash
python3 codebert_eval.py
```
or 
```bash
python3 codebert_eval_sbp.py
```
You'll need to modify the `dir_path` in the script to specify a different dataset.

To load a different trained model, change the value of `model_state_path` in the script accordingly.

