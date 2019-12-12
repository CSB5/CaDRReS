# Bayesian Sigmoid Curve Fitting

Here we provide scripts for Bayesian sigmoid curve fitting of dose-response values from CCLE and GDSC. The dose-response curve is based on [logistic function](https://en.wikipedia.org/wiki/Logistic_function). For more detail, please check the supplementary of CaDRReS paper.

*Chayaporn Suphavilai, Denis Bertrand, Niranjan Nagarajan; Predicting Cancer Drug Response using a Recommender System, Bioinformatics, Volume 34, Issue 22, 15 November 2018, Pages 3907–3914, https://doi.org/10.1093/bioinformatics/bty452*

- GDSC_dose_response_scores.tsv [Download](https://www.dropbox.com/s/2z6b38au9kq89nq/GDSC_dose_response_scores.tsv?dl=0) 
*This version is slightly different from the published version. Drug-sample pairs tested for five dosages were removed as the previous dosage values are incorrect. We are preparing to release a new version soon.*

- CCLE_dose_response_scores.tsv [Download](https://www.dropbox.com/s/glk6sttf1b1wx66/CCLE_dose_response_scores.tsv?dl=0)

### Additional Required Libraries
CaDDReS is based on Python 2.7
- pyjags
- pymc

__Example command__
```sh
python bayesian_sigmoid_curve_fitting_CCLE.py CCLE_dose_response_scores.tsv {out_dir}
```
