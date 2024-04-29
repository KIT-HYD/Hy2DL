# Hy<sup>2</sup>DL: Hybrid Hydrological modeling using Deep Learning methods

![HybridModel](LSTM_SHM.png)

Hy<sup>2</sup>DL is a python library to create Hybrid Hydrological Models for rainfall-runoff prediction, by combining DL methods with process based models. Specifically, the DL methods act as dynamic parameterizations for process-based conceptual models.

The idea of the repository is to have 'easy' to understand codes and to introduce new users to hybrid models. Because of this, we decided to sacrifice some code modularity to gain interpretability. The main codes are presented as jupyter notebooks.

## Structure of the repository:

- **data**: Information necessary to run the codes. The CAMELS-GB and/or CAMELS-US datasets should be added here.
- **aux_functions**: auxiliary functions to run the codes
- **benchmarks**: Information from other studies that was used to benchmark our models
- **conceptual_models**: Present the codes to calibrate basin-wise process-based hydrological model. The calibration routines are based on the SPOTPY library (https://spotpy.readthedocs.io/en/latest/). The process-based models are used as baselines to compare the performance of the hybrid models.
- **datasetzoo**: codes to process the CAMELS-GB and CAMELS-US dataset and incorporate them to the models.
- **experiments**: JupyterNotebooks to run the experiments
- **modelzoo**: codes of the different models that can be used
- **results**: Folder where the results generated by all the codes will be stored.

## Dependencies
The packages used to run the codes are indicated at the beginning of each notebook. It must be considered that the codes for the data-driven models run better in GPU, therefore a PyTorch version that supports GPU should be installed!

## Citation:
This code is part of our study 

```
Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket? Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-1980, 2023.
```

If you want to reproduce the experiments of this paper, run the scripts: Hybrid_LSTM_SHM.ipynb, Hybrid_LSTM_Bucket.ipynb, Hybrid_LSTM_NonSense.ipynb, LSTM_CAMELS_GB.ipynb and LSTM_CAMELS_US.ipynb located in the path LSTM_CAMELS_US.ipynb.

## Authors:
 - Eduardo Acuña Espinoza (eduardo.espinoza@kit.edu)
 - Ralf Loritz (ralf.loritz@kit.edu)
 - Manuel Álvarez Cháves (manuel.alvarez-chaves@simtech.uni-stuttgart.de)

 ## Disclaimer:
 No warranty is expressed or implied regarding the usefulness or completeness of the information and documentation provided. References to commercial products do not imply endorsement by the Authors. The concepts, materials, and methods used in the algorithms and described in the documentation are for informational purposes only. The Authors has made substantial effort to ensure the accuracy of the algorithms and the documentation, but the Authors shall not be held liable, nor his employer or funding sponsors, for calculations and/or decisions made on the basis of application of the scripts and documentation. The information is provided "as is" and anyone who chooses to use the information is responsible for her or his own choices as to what to do with the data. The individual is responsible for the results that follow from their decisions.

This web site contains external links to other, external web sites and information provided by third parties. There may be technical inaccuracies, typographical or other errors, programming bugs or computer viruses contained within the web site or its contents. Users may use the information and links at their own risk. The Authors of this web site excludes all warranties whether express, implied, statutory or otherwise, relating in any way to this web site or use of this web site; and liability (including for negligence) to users in respect of any loss or damage (including special, indirect or consequential loss or damage such as loss of revenue, unavailability of systems or loss of data) arising from or in connection with any use of the information on or access through this web site for any reason whatsoever (including negligence).
