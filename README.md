# Using Slisemap to Interpret Physical Data

These are the auxiliary files for the "Using Slisemap to Interpret Physical Data" paper.

The files are structured as follows:

- The (preprocessed) data files are in the `data` and `GeckoQ` directories.
- Some pretrained Slisemap models are in `models`.
- The source code for the performance measures are in `experiments`.


## Citation

>  __Seppäläinen L, Björklund A, Besel V, Puolamäki K__ (2024)  
>  *Using slisemap to interpret physical data*.  
>  PLoS ONE 19(1): e0297714. DOI: [10.1371/journal.pone.0297714](https://doi.org/10.1371/journal.pone.0297714)


## Datasets

### Volatile organic compounds: GeckoQ

The directory `GeckoQ/` contains four files:

- `Dataframe.csv` contains the GeckoQ dataset, including all explainable features.
- `y_norm.jl` contains the normalized log10 targets.
- `ExpF_norm_.jl` contains the normalized explainable features.
- `indices.txt` contains the indices of the used molecules in the GeckoQ dataset.

The not pre-processed target and explainable features can be extracted from the GeckoQ dataset with the help of `indices.txt`. The targets are in the column "pSat_mbar" and the explainable features are the columns `['NumOfAtoms', 'NumOfC', 'C=C (non-aromatic)', 'hydroxyl (alkyl)', 'aldehyde', 'ketone', 'carboxylic acid', 'ester', 'ether (alicyclic)', 'nitrate', 'nitro', 'carbonylperoxynitrate', 'peroxide', 'hydroperoxide', 'carbonylperoxyacid']`

### Elementary particle jets

The `data` directory contains a `jets.feather` file with the data and a `jets_rf.feather` file with the predictions from a random forest that has been trained using `jets.py`. The data has been extracted from `*.root` files using https://github.com/edahelsinki/slise/blob/explanation_experiments/experiments/data/RootParser.cpp.

### Small organic molecules: QM9

The `data` directory contains a `qm9_interpretable.feather` file with the interpretable features, a `qm9_labels.feather` file with the targets and SMILE strings, and a `qm9_nn.feather` file predictions from a neural network. The interpretable features, the SMILES strings, and the neural network has been created with the `qm9.py` script.

## Models

The `models` directory contains three pre-trained Slisemap models (`*.sm`) that were used for the main Slisemap plots in the paper. The `train.py` script will recreate (retrain) these models, if they are removed.

## Experiments

Running the experiments is done in three steps:
1. Training a cohort of Slisemap models at various subsampling levels
2. Generating results from the Slisemap models
3. Generating the final plot

Tasks 1 and 2 are designed to be run on a Slurm-based environment and may require slight modifications to be run in other configurations.

### Training subsampled models

To train a cohort of models, either run
```
python3 SI/experiments/train_models_for_experiments.py [array id] [dataset name]
```
or, when in a Slurm environment,
```
sbatch SI/experiments/jobs/train_models_for_experiments.job
```
The arguments in the python command are
- `array id`: an index starting from 1, denoting the cohort array index
- `dataset name`: either `qm9`, `jets` or `gecko`

The trained models are saved to SI/experiments/models/`dataset name`/`current date` by default.

### Generating results

After training a cohort of models, we need to generate the various stability measures described further in the associated article. To generate results, either run
```
python3 SI/experiments/get_results.py [array id] [model directory] [results directory] [measure name] [comparison style]
```
or, when in a Slurm environment,
```
sbatch SI/experiments/jobs/get_results.job
```
with the appropriate modifications.
The arguments in the python command are:
- `array id`: similar as above
- `model directory`: a directory containing a cohort of models (such as one generated in the previous step)
- `results directory`: output directory for the results
- `measure name`: used to pick the measure in question. one of `permutation_loss`, `local_model_distance` or `neighbourhood_distance`
- `comparison style`: used to pick how different types of models are compared. Possible values are:
    - `versus`: compare normally trained model against one another in a round-robin fashion
    - `permutation`: compare each model against a permuted counterpart with the
    same index in the filename.
    - `half`: compare each model against one with 50% shared points with the same
    index in the filename.
    - `half_perm`: compare each model against one with permuted labels and 50% shared points

### Plotting results

Finally, to generate the plot, run
```
python3 SI/experiments/sm_stability_plot.py
```
after having modified in the appropriate results directories in the same script.
The generated plot can be found at `SI/experiments/figures/slisemap_stability.pdf`.

### Producing a DR comparison table
To recreate Table 1 in the paper, we use to following procedure.
First, we must have trained SLISEMAP models for each of the three datasets (such as the ones in the models folder).
Then:
1. We train the alternative DR models. To do this, we either run 
```
python3 SI/experiments/train_alternative_DR.py [array id] [dataset name]
```
or, when in a Slurm environment,
```
sbatch SI/experiments/jobs/train_alternative_DR.job
```
2. After training the alternative models, we calculate the DR comparison metrics outlined in the paper. To do this, we either run
```
python3 SI/experiments/slisemap_DR_comparison.py [array id] [model directory containing the pretrained DR models]
```
or, when in a Slurm environment,
```
sbatch SI/experiments/jobs/DR_comparison.job
```
3. Finally, we collect the results to a Latex table. To do this, edit the result directories to SI/experiments/produce_DR_comparison_table.py and run it with the command
```
python3 SI/experiments/produce_DR_comparison_table.py
```
