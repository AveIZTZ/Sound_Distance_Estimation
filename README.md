# Sound_Distance_Estimation
This is the baseline for sound distance estimation task in IASP lab.

## Installation Guide
```
conda create --name dist python==3.9
conda activate dist
pip install -r requirements.txt
```
## Datasets
The array is a linear array with 64 microphones spaced 0.02 meters apart. The sound source is located in the endfire direction at a distance ranging from 0 to 1.5 meters.
```
/home/yujiezhu/data/data_for_SED/
```

## Extract Audio Features
```
# in dist_model_code/parameters_z.py:
dataset_dir   ='/home/yujiezhu/data/data_for_SED/test/input/',
feat_label_dir='/home/yujiezhu/data/data_for_SED/test/processed/',

# Extract audio features
python dist_model_code/batch_feature_extraction.py

## Model Training

### Change output path in config/A_RC_SED-SDE.yaml:
```
model_dir: 
dcase_output_dir: 
output_dir: 

```

### Distance estimation model training:
```
python dist_model_code/train_seldnet.py
```
