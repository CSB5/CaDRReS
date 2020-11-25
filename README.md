# CaDRReS

**Ca**ncer **D**rug **R**esponse prediction using a **Re**commender **S**ystem (**CaDRReS**) is based on the matrix factorization approach to predict which drugs are sensitive for an unseen cell line. CaDRReS can also be used for studying drug response mechanisms including classes of drugs, subtypes of cell lines based on drug response profile, and drug-pathway associations.

*Chayaporn Suphavilai, Denis Bertrand, Niranjan Nagarajan; Predicting Cancer Drug Response using a Recommender System, Bioinformatics, Volume 34, Issue 22, 15 November 2018, Pages 3907–3914, https://doi.org/10.1093/bioinformatics/bty452*

> A newer TensorFlow implementation of CaDRReS, with additional features to support single-cell RNA-seq data, as well as Jupyter notebooks for model training and testing, can be found here: https://github.com/CSB5/CaDRReS-SC. 

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

Users have two options to run CaDRReS. The first option is to run `CaDRReS_test.py` for applying the pre-trained model based on the GDSC dataset to predict drug response of input samples. The second option is to run `CaDRReS_train_and_test.py` for training and testing the model as well as obtaining the pharmacogenomic space for input samples. We provide examples for both options below:

## Predicting drug responses from an existing model

Here we provides a model trained on the GDSC dataset using 10 dimensions for the pharmacogenomic space.

__Input files__
- `CaDRReS_model.pickle` is a file containing an existing model
- `drug_response_ic50_test.csv` contains an empty matrix where rows are cell lines and columns are features
- `cell_line_features.csv` contains a feature matrix where rows are the cell lines to be analyzed and columns are features. The features have to match with the features used for training the model.

__Output file__
- Drug response prediction for input cell lines `{out_dir}/CaDRReS_pred.csv`
- Matrices P and Q and bias terms `{out_dir}/CaDRReS_pred.pickle`

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
- `drug_response_ic50_train.csv` contains a matrix of IC50s where rows are cell lines and columns are drugs
- `drug_response_ic50_test.csv` contains an empty matrix where rows are cell lines and columns are drugs
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
An example command to train a model for the CCLE dataset:
```sh
$ cd scripts
$ python CaDRReS_train_and_test.py ../input/ccle_all_abs_ic50_bayesian_sigmoid.csv ../input/ccle_all_abs_ic50_bayesian_sigmoid.csv ../input/ccle_cellline_pcor_ess_genes.csv ../misc/ccle_drugMedianGE0.txt ../output 10 50000 0.01 0
```

## Bayesian Sigmoid Curve Fitting

For calculating dose-response curves for CCLE and GDSC, please visit [this page.](https://github.com/CSB5/CaDRReS/tree/master/scripts/bayesian_sigmoid_curve_fitting)

## Contact

Please direct any questions or feedback to Chayaporn Suphavilai (suphavilaic@gis.a-star.edu.sg) and Niranjan Nagarajan (nagarajann@gis.a-star.edu.sg).

