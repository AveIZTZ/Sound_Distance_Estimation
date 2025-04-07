# Sound_Distance_Estimation
This is the baseline for sound distance estimation task in IASP lab.

## Installation Guide
```
conda create --name dist python=3.9
conda activate dist
pip install -r requirements.txt
```
## Datasets
The array is a linear array with 64 microphones spaced 0.02 meters apart. The sound source is located in the endfire direction at a distance ranging from 0 to 1.5 meters.
```
/home/yujiezhu/code/sound_distance_estimation/data_new/input
```

## Extract Audio Features
```
# in dist_model_code/parameters.py, line 104 and 105:
params['dataset_dir'] = '/home/yujiezhu/code/sound_distance_estimation/data_new/input'
params['feat_label_dir'] = '……/data_new/processed/'

# Extract audio features
python dist_model_code/batch_feature_extraction.py

# Change lines in .npy files, in process_zyj/change.py:
fold_name = “”
npy_folder = "data_new/processed/mic_dev_label"
csv_folder = "data_new/input/metadata_dev/test"

python process_zyj/change.py
```

## Model Training

### Change data path in dist_model_code/parameters.py, line 152, 153, and 158:
```
params['dataset_dir'] = '……/input/'
params['feat_label_dir'] = '……/processed/'

params['ind_data_train_test_split'] = [[……], [……], [……]] # train/valid/test dataset, meet your fold_name
```

### Distance estimation model training:
Training Locata with MSE(mean square error) loss

```
python dist_model_code/train_seldnet.py 4
```
Training Starss with TAPE(Thresholded mean absolute error) threshold of 0.4 loss

```
python dist_model_code/train_seldnet.py 5
```
