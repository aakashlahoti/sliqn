

# Sharpened Lazy Incremental Quasi Newton Method

This repository contains the official implementations and experiments for the Sharpened Lazy Incremental Quasi Newton (SLIQN) Method. The SLIQN Method is the first incremental Quasi-Newton method with an explicit superlinear rate of convergence and a per-iteration \$O(d^2)\$ cost.

## Setup

The following packages are required to run the code:
- MATLAB R2021a or later.
- LIBSVM installed in MATLAB.
- Python 3.10 or later.
- Python packages: Numpy, Matplotlib.
- Download and place the [Data.zip](https://www.dropbox.com/scl/fi/1cvpdbk6gcpva2a19asjk/Data.zip?rlkey=5gt1whw78u0kur421s1wtjuk7&dl=0) file in the main folder.

Create the directories:
- `data_store/a9a/epochs`
- `data_store/a9a/all_vars`
- `data_store/protein/epochs`
- `data_store/protein/all_vars`
- `data_store/w8a/epochs`
- `data_store/w8a/all_vars`
- `Matlab_plots`
- `Python_plots`

## Logistic Regression Experiments

The code must be run from within the `Regularized_logisitc_regression` directory. The plots are saved in the `Matlab_plots` directory. We support the following LIBSVM datasets:

1. **A9A**: `matlab -r a9a_p_21`
2. **Breast Cancer**: `matlab -r breast_cancer_p_21`
3. **German Numer**: `matlab -r german_numer_p_21`
4. **Mushrooms**: `matlab -r mushrooms_p_21`
5. **Phising**: `matlab -r phising_p_21`
6. **Protein**: `matlab -r protein_p_21`
7. **Segment**: `matlab -r segment_p_21`
8. **SVM Guide 3**: `matlab -r svmguide3_p_21`
9. **W8A**: `matlab -r w8a_p_21`

## Quadratic Minimization Experiments

The `Quadratic_minimization` directory contains easy-to-follow Jupyter notebooksfor the following experiments:

1. **Training Curves for High and Low Condition Numbers:** `Quadratic_Minimization_Exp.ipynb`
2. **Training Loss vs. Condition Number:** `Error_vs_Condition_No_Exp.ipynb`
3. **Training Loss vs. Problem Dimension:**`Error_vs_Dimension_Exp.ipynb`

The output plots from these experiments are saved in the `Python_plots` directory.
