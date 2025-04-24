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

    params_val   = params.copy()
    params_val['dataset_dir']      = args.test_dataset_dir
    params_val['feat_label_dir']   = args.test_lable_dir

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:7" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    print('Loading unseen test dataset:')
    data_gen_test = cls_data_generator.DataGenerator(params=params_val, shuffle=False, per_file=True)

    # Collect i/o data size and load model configuration
    data_in, data_out = data_gen_test.get_data_sizes()
    model = seldnet_model.CRNN(data_in, data_out, params).to(device)
    #model = ResnetConformer_sed_doa_nopool(in_channel=10, in_dim=64, out_dim=39).to(device)

    criterion_val = seldnet_model.perm_3(loss_type='mae', only_mask=params['perm_3_onlyMask'])

    print('Load best model weights')
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

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
    parser.add_argument('--test_dataset_dir',  default="/home/yujiezhu/data/data_for_SED/test/input/")
    parser.add_argument('--test_lable_dir',    default="/home/yujiezhu/data/data_for_SED/test/processed/")
    args = parser.parse_args()

    main(args)