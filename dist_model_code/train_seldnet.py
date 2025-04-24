#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os, shutil, argparse
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import seldnet_model
import parameters
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D, compute_dist_metrics_perm3, compute_dist_metrics_extended_perm3
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model
from models_dcase.resnet_conformer_audio import ResnetConformer_sed_doa_nopool
import yaml

def train_epoch(data_generator, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for data, target in data_generator.generate():
        # load one batch of data
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches

    return train_loss

def test_epoch_onlyDist(data_generator, model, criterion, dcase_output_folder, params, device):

    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0

    perm_2 = params['permutation_2']
    perm_3 = params['permutation_3']

    with torch.no_grad():
        for data, target in data_generator.generate():
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            output = model(data)
            loss = criterion(output, target)
            output = reshape_3Dto2D(output)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}

            for frame_cnt in range(output.shape[0]):
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                if perm_2 or perm_3:
                    output_dict[frame_cnt].append([output[frame_cnt][0], output[frame_cnt][1]])
                else:
                    output_dict[frame_cnt].append([output[frame_cnt][0]])
            data_generator.write_output_format_file_onlyDist(output_file, output_dict, dist_and_mask=(perm_2 or perm_3))

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss /= nb_test_batches

    return test_loss

def main(args):
    
    with open(os.path.join('config', '{}.yaml'.format(args.config_name)), 'r') as f:
        params = yaml.safe_load(f)
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling
    model_name = os.path.join(params['output_dir'], 'best_model.h5')

    params_train = params.copy()
    params_val   = params.copy()
    params_train['dataset_dir']    = args.train_dataset_dir
    params_train['feat_label_dir'] = args.train_lable_dir
    params_val['dataset_dir']      = args.test_dataset_dir
    params_val['feat_label_dir']   = args.test_lable_dir

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:7" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # Load train and validation data
    print('Loading training dataset:')
    data_gen_train = cls_data_generator.DataGenerator(params=params_train)

    print('Loading validation dataset:')
    data_gen_val = cls_data_generator.DataGenerator(params=params_val, shuffle=False, per_file=True)

    # Collect i/o data size and load model configuration
    data_in, data_out = data_gen_train.get_data_sizes()
    model = seldnet_model.CRNN(data_in, data_out, params).to(device)
    #model = ResnetConformer_sed_doa_nopool(in_channel=10, in_dim=64, out_dim=39).to(device)

    print('---------------- SELD-net -------------------')
    print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
    print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
        params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
        params['fnn_size']))
    print(model)

    # start training
    patience_cnt, epoch_cnt = 0, 0
    best_val_loss, best_train_loss = 9999, 9999
    best_val_mae = 9999
    best_val_gen = 9999
    best_epoch = -1

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = seldnet_model.perm_3(params['perm_3_loss_type'], only_mask=params['perm_3_onlyMask'], thr=params['perm_3_loss_mpe_type_thr'])
    criterion_val = seldnet_model.perm_3(loss_type='mae', only_mask=params['perm_3_onlyMask'])

    while patience_cnt < params['patience']:
        # ---------------------------------------------------------------------
        # TRAINING
        # ---------------------------------------------------------------------
        start_time = time.time()
        train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device)
        train_time = time.time() - start_time

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        start_time = time.time()
        dcase_output_val_folder = os.path.join(params['output_dir'], 'results/epoch_{}'.format(epoch_cnt))
        cls_feature_class.create_folder(dcase_output_val_folder)
        val_loss = test_epoch_onlyDist(data_gen_val, model, criterion_val, dcase_output_val_folder, params, device)

        agg_metrics = compute_dist_metrics_perm3(params_val, ref_files_folder=dcase_output_val_folder)
        val_gt_mae, val_gt_mpe, val_pred_mae = agg_metrics['global']['gt_mae'], agg_metrics['global']['gt_mpe'], agg_metrics['global']['overall_mae']

        if params['perm_3_loss_type'] in ['mae','mse']:
            val_gen = val_gt_mae
        else:
            val_gen = val_gt_mpe

        if val_gen < best_val_gen:
            best_val_gen = val_gen
            torch.save(model.state_dict(), model_name)
            patience_cnt = 0
            best_epoch = epoch_cnt
        else:
            patience_cnt += 1

        # save model
        checkpoint_output_dir = os.path.join(params['output_dir'], 'check_points')
        cls_feature_class.create_folder(checkpoint_output_dir)
        model_path = os.path.join(checkpoint_output_dir, 'checkpoint_epoch{}.h5'.format(epoch_cnt))
        torch.save(model.state_dict(), model_path)

        print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss, 3)}, '
                f'val_loss, val_pred_mae: {round(val_loss, 3)}, {round(val_pred_mae, 3)}')

        epoch_cnt += 1

    # ---------------------------------------------------------------------
    # Evaluate on unseen test data
    # ---------------------------------------------------------------------
    print('Load best model weights')
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    print('Loading unseen test dataset:')
    data_gen_test = cls_data_generator.DataGenerator(params=params_val, shuffle=False, per_file=True)

    # Dump results in DCASE output format for calculating final scores
    dcase_output_test_folder = os.path.join(params['output_dir'], 'test_results')
    cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
    print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

    test_loss = test_epoch_onlyDist(data_gen_test, model, criterion_val, dcase_output_test_folder, params, device)
    agg_metrics = compute_dist_metrics_extended_perm3(params_val, ref_files_folder=dcase_output_test_folder)
    val_gt_mae, val_gt_median, val_gt_std = agg_metrics['global']['gt_mae'], agg_metrics['global']['gt_median'], agg_metrics['global']['gt_std']
    print(f'test_loss, test_mae, test_median, test_std : {round(test_loss, 3)}, {round(val_gt_mae, 3)}, {round(val_gt_median, 3)}, {round(val_gt_std, 3)}')
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='A_RC_SED-SDE', help='name of config')
    parser.add_argument('--train_dataset_dir', default="/home/yujiezhu/data/data_for_SED/train/input/")
    parser.add_argument('--train_lable_dir',   default="/home/yujiezhu/data/data_for_SED/train/processed/")
    parser.add_argument('--test_dataset_dir',  default="/home/yujiezhu/data/data_for_SED/test/input/")
    parser.add_argument('--test_lable_dir',    default="/home/yujiezhu/data/data_for_SED/test/processed/")
    args = parser.parse_args()

    main(args)