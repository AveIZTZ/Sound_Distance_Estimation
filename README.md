# Sound_Distance_Estimation
This is the baseline for sound distance estimation in IASP lab.

## Installation Guide
```
conda create --name dist python=3.8
conda activate dist
pip install -r requirements.txt
```
## Datasets
```
/home/yujiezhu/code/sound_distance_estimation/data_new/input
```

## Extract Audio Features
```
# in dist_model_code/parameters.py, line 104 and 105:
params['dataset_dir'] = 'data_new/input/'
params['feat_label_dir'] = 'data_new/processed/'

# Extract audio features
python dist_model_code/batch_feature_extraction.py

# Change lines in .npy files:
fold_name = “”
npy_folder = "data_new/processed/mic_dev_label"
csv_folder = "data_new/input/metadata_dev/test"

python process_zyj/change.py
```

## Model Training

### Change data path in dist_model_code/parameters.py, line 152 and 153:
```
params['dataset_dir'] = '……/input/'
params['feat_label_dir'] = '……/processed/'
```

### Event detection based pre-training:

We already provide a pre-trained model at models/pretrained_dcase_event_detector.h5 . So this step is optional.

Pretraining only the event classifier
```
python dist_model_code/train_seldnet.py 2
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
