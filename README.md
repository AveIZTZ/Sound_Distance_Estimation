# Sound_Distance_Estimation
This is the baseline for sound distance estimation in IASP lab.

## Installation Guide
```python
conda create --name dist python=3.8
conda activate dist
pip install -r requirements.txt
```
## Extract Audio Features
```python
# in dist_model_code/batch_feature_extraction.py, line 104 and 105:
params['dataset_dir'] = 'data_new/input/'
params['feat_label_dir'] = 'data_new/processed/'

# Extract audio features
python dist_model_code/batch_feature_extraction.py
```

## Model Training
### Event detection based pre-training:

We already provide a pre-trained model at models/pretrained_dcase_event_detector.h5 . So this step is optional.

Pretraining only the event classifier
```python
python dist_model_code/train_seldnet.py 2
```

### Distance estimation model training:
Training Locata with MSE(mean square error) loss

```python
python dist_model_code/train_seldnet.py 4
```
Training Starss with TAPE(Thresholded mean absolute error) threshold of 0.4 loss

```python
python dist_model_code/train_seldnet.py 5
```
