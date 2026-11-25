# MMOE-AMP

This repository provides datasets, feature-generation code, the pretrained MMOE-AMP model, and downstream MIC classification models.

## Repository layout

- `datatset/`: AMP/non-AMP train, validation and test FASTA files, MIC datasets, and the optional `example_test.fasta` sample. 
- `feature_generation/Feature_generation.py`: ESM-C 300M embeddings, MACREL features and TextCNN sequence encodings.
- `AMP_predict/model_multi_moecl_datav3.pth`: pretrained AMP model. Due to GitHub's file size limitations, the model weights are hosted externally and can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1aPWS7Ooz5gUdUxbnEp_PMQ?pwd=e10n) (access code: `e10n`). After downloading, place the model file in the `AMP_predict/` directory.
- `AMP_predict/load_model_test.py`: original test-set evaluation code.
- `AMP_predict/proteinmodel_classifier2l_SUPCL.py`: historical training reference, not required for pretrained-model inference.
- `AMP_predict/289211_smorfs_predict.py`: historical large-scale prediction reference; its fixed batching and FASTA header assumptions are retained.
- `MIC_predict/load_model_MIC_predict.py`: original MIC training, latent-representation extraction and prediction reference.


## Environment

The software environment and dependency configurations for MMoE-AMP are currently being organized. Detailed installation instructions and environment configuration files will be provided in a subsequent update.

## Feature generation

Run each script from its containing directory. Starting from the repository root:

```sh
cd feature_generation
python Feature_generation.py
cd ..
```

The script processes the six AMPBASE/nonAMPBASE train, validation and test FASTA files in `datatset/`. It creates `esm_embedding/`, `macrel_embedding/`, and `textcnn_convert/` inside `feature_generation/`, using the input FASTA stem for each output file.

`datatset/example_test.fasta` is retained as an optional small sample and is not required by the main workflow. To extract its features, add `example_test` to all three input lists in `Feature_generation.py`. 

## Reproduction workflow

This release preserves the original training, evaluation and screening code as research references. Run scripts from their containing directories, after providing the required features and dependencies.

### AMP test-set evaluation

After generating the six dataset feature sets:

```sh
cd AMP_predict
python load_model_test.py
cd ..
```

This script loads `model_multi_moecl_datav3.pth` and evaluates the labelled AMP/non-AMP test splits. `proteinmodel_classifier2l_SUPCL.py` documents the original training procedure and is not needed for evaluation of the supplied checkpoint. 

### AMP prediction for new sequences

Use `AMP_predict/289211_smorfs_predict.py` as the reference for batch prediction. Its original screening input is `datatset/289211_smorfs_clean_merge_ids_noX_nostar.fasta`; corresponding features are expected under the three `feature_generation/` output directories with the same filename stem. 

For a new dataset, generate its three feature representations in the same sequence order, then update the FASTA and feature paths in the batch script. The original loop uses 1,060 batches of 10,000 and assumes a specific FASTA header format with an additional `>` separator. Adapt the batch bounds and header handling to the new dataset before running. Create `AMP_predict/Predicted_results_datav3/` for its output CSV files. The original batching and parsing logic are preserved in this release.

### MIC analysis and prediction

`MIC_predict/load_model_MIC_predict.py` loads `../AMP_predict/model_multi_moecl_datav3.pth` to extract latent representations. It preserves the original MIC training and cross-validation steps, followed by random-forest prediction. The candidate prediction input is `datatset/potential_amp.fasta`, with matching `potential_amp.h5` / `.csv` features under `feature_generation/`. These candidate inputs must be supplied separately. The training portion also needs features for the MIC CSV datasets; the default FASTA feature-generation lists do not produce them. Set `bac` for the organism being analysed (the current value is `aeruginosa`). The final intersection analysis requires prediction CSV files for all three organisms. Run the relevant sections separately when reproducing individual-organism analyses.

### Shared preprocessing dependency

The original evaluation and prediction code reads training features to refit MACREL standardization. A new FASTA and the checkpoint alone are insufficient. Generate or supply the required training features and preserve their feature-column ordering. 

## Data availability

Due to file size, precomputed training, validation and test feature files and the sORF sequence files used for large-scale screening have not been included in this release. These files will be uploaded progressively after the paper becomes public. Full screening reproduction requires the original sORF inputs and corresponding features.