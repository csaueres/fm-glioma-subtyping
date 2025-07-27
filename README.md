# Repository for Glioma Classification using CompPath FMs.
This repository accompanies *From Histology to Diagnosis: Leveraging Foundation Models for Glioma Classification* (under review).


## Setup

### Installation
Required packages are listed in env.yml. An conda environment can be created by running the following command:
```
conda env create -n fmgs -f env.yml
```

### Download FMs
Please request access to the foundation models from their respective sources.
    [UNI](https://huggingface.co/MahmoodLab/UNI)
    [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath)
    [Virchow](https://huggingface.co/paige-ai/Virchow)
Copy your HuggingFace Token into the embedder/TOKENS.py file to automatically download model weights. 

### Download Data
TCGA data can be downloaded from https://portal.gdc.cancer.gov.
Alternatively, the provided model checkpoints can be used to predict glioma subtype on another dataset.

### Install CLAM
Clone this repository: https://github.com/mahmoodlab/CLAM.
The script create_patches_fp.py is required for generating image patches from WSIs. The environment provided in this repository can be used to perform patching.

## Example Pipeline

Our pipeline follows the general structure of 

### Generate Patches
Using the CLAM respository, patch all WSIs. Please see [here](https://github.com/mahmoodlab/CLAM) for more details.
We use 256x256 patches at a magnification of 20x for training our models. At 40x magnification this would correspond to patch sizes of 512x512.

```
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset tcga.csv --seg --patch --stitch
```

### (Optional) Clean Patches
This step removes patches which are homogenous in color. In particular this includes background patches, smudges, and pen marks, which have been included in the initial patching.
Removing these is recommended for biopsy slides which have been extensively written on, as in these cases the number of non-relevant patches often outnumbers the number of patches
which contain actual tissue.

```
python clean_patches.py --in_patches PATCH_IN_DIR --out_patches PATCH_OUT_DIR --data_slide_dir RAW_DATA_DIR --slide_ext .svs --csv_path METADATA.csv
```

### Create FM Embeddings
After patching, we can use the FMs to generate an embedding for each patch. This step should be repeated for each desired FM and augmentation.
Embeddings will be saved according to the following structure OUT_DIR/embedder/augment_method/h5_files/slide_xxx.h5. For training and evaluating models, the embedding root directory should only be EMBEDDING_DIR.

```
python create_embeddings.py --embedder <uni_v1,gigapath,virchow> --feat_dir EMBEDDING_DIR --augment_method <no-augment,rc-mix,rc-pure,macenko-norm> --data_patches_dir PATCHES_DIR --data_slide_dir RAW_DATA_DIR --slide_ext .svs --csv_path METADATA.csv --batch_size 256
```

### Train Classifier
To train a model using the FM embeddings generated in the previous step use the script below. We provide implementations of CLAM, MambaMIL, and a Linear classifier.
A classifier can be trained on multiple augmentation strategies by adding multiple arguments after the --augments flag.

```
python train_classifier.py --drop_out 0.5 --reg 0.2 --lr 1e-4 --patch_frac 0.5 --augments AUG1 [AUG2] --model_type <mamba,clam_sb,linear> --n_block 2 --embedder <uni_v1,gigapath,virchow> --exp_code NAME --task idh_1p19q_class --data_dir EMBEDDING_DIR --results_dir OUT_DIR --csv METADATA.csv --split_dir SPLIT_DIR --k 5 --seed 7
 ```

### Evaluate Classifier
We provide two scripts for evaluating model performance. eval_classifier.py is suited for evaluation on a single augmentation (or no-augment) while eval_classifier_tta.py should be used when ensembling across multiple augmentations.


```
python eval_classifier.py --exp_code NAME --embedder <uni_v1,gigapath,virchow> --model_type <mamba,clam_sb,linear> --checkpoint_dir CHECKPOINT_DIR --eval_data_csv METADATA.csv --data_dir EMBEDDING_DIR --augments AUG  --k 5 --case_level
```

```
python eval_classifier_tta.py --exp_code NAME --embedder <uni_v1,gigapath,virchow> --model_type <mamba,clam_sb,linear> --checkpoint_dir CHECKPOINT_DIR --eval_data_csv METADATA.csv --data_dir EMBEDDING_DIR --augments AUG1 [AUG2] [AUG3] --k 5
```

## Acknowledgements
We would like to thank the authors of the following repos for publicly releasing their code, which has significantly contributed to our own work.

CLAM: https://github.com/mahmoodlab/CLAM

UNI: https://github.com/mahmoodlab/UNI

MambaMIL: https://github.com/isyangshu/MambaMIL

MCC Loss: https://github.com/daniel-scholz/address-class-imbalance