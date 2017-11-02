# CaDRReS
---

**Ca**ncer **D**rug **R**esponse prediction using a **Re**commender **S**ystem (**CaDRReS**) is based on the matrix factorization approach to predict which drugs are sensitive for an unseen cell line. CaDRReS can also be used for studying drug response mechanisms including classes of drugs, subtypes of cell lines based on drug response profile, and drug-pathway associations.

## How to run CaDRReS?

CaDDReS is based on Python 2.7
##### Required libraries
- Pandas
- Numpy
- Scipy
- Argparse

```sh
pip install -r requirements.txt
```

##### NOTE

Users have two options to run CaDRReS. The first option is to run `CaDRReS_test.py` for applying the pre-trained model based on GDSC dataset to predict drug response of input samples. The second option is to run `CaDRReS_train_and_test.py` for training and testing the model as well as optaining the pharmacogenomic space of input samples. We provide examples of both options below:

## Predicting drug responses from an existing model

Here we provides a model trained on GDSC dataset using 10 dimensions of the pharmacogenomic space.

__Input files__
- `CaDRReS_model.pickle` is a file containing existing model
- `drug_response_ic50_test.csv` contains an empty matrix where rows are cell lines and columns are features
- `cell_line_features.csv` contains a feature matrix where rows are testing cell lines and columns are features. The features have to match with the features used for training the model.

__Output file__
- Drug response prediction of testing cell lines `{out_dir}/CaDRReS_pred.csv`
- Matrices P and Q and biases terms `{out_dir}/CaDRReS_pred.pickle`

__Command__
```sh
python CaDRReS_test.py CaDRReS_model.pickle ../input/ccle_all_abs_ic50_bayesian_sigmoid.csv ../input/ccle_cellline_pcor_ess_genes.csv {out_dir}
```

An example command for predicting drug responses based on the provided model:
```sh
$ cd scripts
$ python CaDRReS_test.py ../output/10D/seed0/lr0-01/CaDRReS_model.pickle ../input/ccle_all_abs_ic50_bayesian_sigmoid.csv ../input/ccle_cellline_pcor_ess_genes.csv ../output
```

## Training and testing a model

__Input files__
- `drug_response_ic50_train.csv` contains a matrix of IC50s where rows are cell lines and columns are features
- `ug_response_ic50_test.csv` contains an empty matrix where rows are cell lines and columns are features
- `cell_line_features.csv` contains a feature matrix where rows are both testing and training cell lines and columns are features
- `drug_list.txt` is a text file contains drugs of interest 

__Output files__
- Drug response prediction of testing cell lines `{out_dir}/{f}D/seed{seed}/lr{l_rate}/CaDRReS_pred_end.csv`
- Drug response prediction of training cell lines `{out_dir}/{f}D/seed{seed}/lr{l_rate}/CaDRReS_pred_end_train.csv`
- A pickle file contains the model `{out_name}/{f}D/seed{seed}/lr{l_rate}/CaDRReS_model.pickle`
- Cell line (P) and drug (Q) matrices `{out_name}/{f}D/seed{seed}/lr{l_rate}/CaDRReS_P.csv` and `{out_name}/{f}D/seed{seed}/lr{l_rate}/CaDDReS_Q.csv`

__Parameters__

| Description | Variable | Example |
| ------ | ------ | ------ | 
| Number of dimensions | f | 10 |
| Output directory | out_dir | output/ |
| Maximum iterations | max_iterations | 50000 |
| Learning rate | l_rate | 0.01 |
| Random seed | seed | 0 |

__Command__

```sh
python CaDRReS_train_and_test.py drug_response_ic50_train.csv drug_response_ic50_test.csv  cell_line_features.csv drug_list.txt {out_dir} {f} {max_iterations} {l_rate} {seed}
```

Note that CaDRReS also saves checkpoints (parameters and predictions) of the model for every 1000 iterations.
An example command to train a model for CCLE dataset:
```sh
$ cd scripts
$ python CaDRReS_train_and_test.py ../input/ccle_all_abs_ic50_bayesian_sigmoid.csv ../input/ccle_all_abs_ic50_bayesian_sigmoid.csv ../input/ccle_cellline_pcor_ess_genes.csv ../misc/ccle_drugMedianGE0.txt ../output 10 100 0.01 0
```


## Contact

Please direct any questions or feedback to Chayaporn Suphavilai (suphavilaic99@gis.a-star.edu.sg) and Niranjan Nagarajan (nagarajann@gis.a-star.edu.sg).

