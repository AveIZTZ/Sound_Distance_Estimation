# Contains routines for labels creation, features extraction and normalization
#
import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import shutil
import math
import wave
import contextlib
from scipy import signal
from multiprocessing import Pool
import time

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

# augmentation list
aug_list = {}
aug_list['aug1'] = [2,4,1,3]
aug_list['aug2'] = [4,2,3,1]
aug_list['aug3'] = [1,2,3,4]
aug_list['aug4'] = [2,1,4,3]
aug_list['aug5'] = [3,1,4,2]
aug_list['aug6'] = [1,3,2,4]
aug_list['aug7'] = [4,3,2,1]
aug_list['aug8'] = [3,4,1,2]

class FeatureClass:
    def __init__(self, params, is_eval=False):
        """
        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._dataset_dir    = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'dev')
        self._audio_dir = os.path.join(self._dataset_dir, self._dataset_combination)
        self._desc_dir  = os.path.join(self._dataset_dir, 'metadata_dev')
        self._feat_label_dir = params['feat_label_dir']

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4

        self._multi_accdoa = params['multi_accdoa']
        self._use_salsalite = params['use_salsalite']

        self._onlyDist = params['only_dist']
        self._chan_swap_aug = params['chan_swap_aug']
        self._chan_aug_folds = params['chan_aug_folds']

        if self._use_salsalite and self._dataset=='mic':
            # Initialize the spatial feature constants
            self._lower_bin = np.int(np.floor(params['fmin_doa_salsalite'] * self._nfft / np.float(self._fs)))
            self._lower_bin = np.max((1, self._lower_bin))
            self._upper_bin = np.int(np.floor(np.min((params['fmax_doa_salsalite'], self._fs//2)) * self._nfft / np.float(self._fs)))

            # Normalization factor for salsalite
            c = 343
            self._delta = 2 * np.pi * self._fs / (self._nfft * c)
            self._freq_vector = np.arange(self._nfft//2 + 1)
            self._freq_vector[0] = 1
            self._freq_vector = self._freq_vector[None, :, None]  # 1 x n_bins x 1 

            # Initialize spectral feature constants
            self._cutoff_bin = np.int(np.floor(params['fmax_spectra_salsalite'] * self._nfft / np.float(self._fs)))
            assert self._upper_bin <= self._cutoff_bin, 'Upper bin for doa featurei {} is higher than cutoff bin for spectrogram {}!'.format()
            self._nb_mel_bins = self._cutoff_bin-self._lower_bin 
        else:
            self._nb_mel_bins = params['nb_mel_bins']
            self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
        # Sound event classes dictionary
        self._nb_unique_classes = params['unique_classes']

        self._filewise_frames = {}

    def get_frame_stats(self):

        if len(self._filewise_frames)!=0:
            return

        print('Computing frame stats:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(self._audio_dir, self._desc_dir, self._feat_dir))
        """ for sub_folder in os.listdir(self._audio_dir):
            loc_aud_folder = os.path.join(self._audio_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                with contextlib.closing(wave.open(os.path.join(loc_aud_folder, wav_filename),'r')) as f: 
                    audio_len = f.getnframes()
                nb_feat_frames = int(audio_len / float(self._hop_len))
                nb_label_frames = int(audio_len / float(self._label_hop_len))
                self._filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
        """
        for file_cnt, file_name in enumerate(os.listdir(self._audio_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            with contextlib.closing(wave.open(os.path.join(self._audio_dir, wav_filename),'r')) as f: 
                audio_len = f.getnframes()
            nb_feat_frames = int(audio_len / float(self._hop_len))
            nb_label_frames = int(audio_len / float(self._label_hop_len))
            self._filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
        return

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        if audio.shape[1] > 4: # If more than 4 channels, keep only 6,10,26,22 channels
            audio = audio[:, [0,1,2,3]]
            new_fs = self._fs
            audio = signal.resample(audio, int(len(audio) * float(new_fs) / fs))
            fs = new_fs

        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input, _nb_frames):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len, win_length=self._win_len, window='hann')
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T
    
    def _get_spectrogram_for_file_chanSwapAug(self, audio_filename, chan_seq):
        audio_in, fs = self._load_audio(audio_filename)

        chan_seq = [i-1 for i in chan_seq]
        audio_in = audio_in[:, chan_seq]
        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        audio_spec = self._spectrogram(audio_in, nb_feat_frames)
        return audio_spec

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)
         
        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        audio_spec = self._spectrogram(audio_in, nb_feat_frames)
        return audio_spec

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_salsalite(self, linear_spectra):
        # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
        # spatial features
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase_vector = phase_vector / (self._delta * self._freq_vector)
        phase_vector = phase_vector[:, self._lower_bin:self._cutoff_bin, :]
        phase_vector[:, self._upper_bin:, :] = 0
        phase_vector = phase_vector.transpose((0, 2, 1)).reshape((phase_vector.shape[0], -1))

        # spectral features
        linear_spectra = np.abs(linear_spectra)**2
        for ch_cnt in range(linear_spectra.shape[-1]):
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
        linear_spectra = linear_spectra[:, self._lower_bin:self._cutoff_bin, :]
        linear_spectra = linear_spectra.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        
        return np.concatenate((linear_spectra, phase_vector), axis=-1)    

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------

    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)
        start_s = time.time()
        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(self._audio_dir, self._desc_dir, self._feat_dir))
        
        """ for sub_folder in os.listdir(self._audio_dir):
            loc_aud_folder = os.path.join(self._audio_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                wav_path = os.path.join(loc_aud_folder, wav_filename)
                feat_path = os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0]))
                self.extract_file_feature((file_cnt, wav_path, feat_path)) """

        for file_cnt, file_name in enumerate(os.listdir(self._audio_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            wav_path = os.path.join(self._audio_dir, wav_filename)
            feat_path = os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0]))
            self.extract_file_feature((file_cnt, wav_path, feat_path))

        print(time.time()-start_s)

    def extract_file_feature_chanSwapAug(self, _arg_in):
        _file_cnt, _wav_path, _feat_path = _arg_in
        for suf, chanSeq in aug_list.items():
            spect = self._get_spectrogram_for_file_chanSwapAug(_wav_path, chanSeq)

            #extract mel
            if not self._use_salsalite:
                mel_spect = self._get_mel_spectrogram(spect)

            feat = None
            if self._use_salsalite:
                feat = self._get_salsalite(spect)
            else:
                # extract gcc
                gcc = self._get_gcc(spect)
                feat = np.concatenate((mel_spect, gcc), axis=-1)

            if feat is not None:
                print('{}: {}_{}, {}'.format(_file_cnt, os.path.basename(_wav_path), suf, feat.shape ))
                np.save(_feat_path[:-4] + '_' + suf + '.npy', feat)

    def extract_file_feature(self, _arg_in):
        _file_cnt, _wav_path, _feat_path = _arg_in
        spect = self._get_spectrogram_for_file(_wav_path)

        #extract mel
        if not self._use_salsalite:
            mel_spect = self._get_mel_spectrogram(spect)

        feat = None
        if self._use_salsalite:
            feat = self._get_salsalite(spect)
        else:
            # extract gcc
            gcc = self._get_gcc(spect)
            feat = np.concatenate((mel_spect, gcc), axis=-1)

        if feat is not None:
            print('{}: {}, {}'.format(_file_cnt, os.path.basename(_wav_path), feat.shape ))
            np.save(_feat_path, feat)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))
        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(spec_scaler, normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(os.path.join(self._feat_dir_norm, file_name), feat_file)
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self.get_frame_stats()
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._audio_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)

        for sub_folder in os.listdir(self._desc_dir):
            loc_desc_folder = os.path.join(self._desc_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
                desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
                desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                label_mat = self.get_labels_for_file(desc_file, nb_label_frames)
                print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    def extract_all_labels_onlyDist(self):
        self.get_frame_stats()
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(self._audio_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)

        """ for sub_folder in os.listdir(self._desc_dir):
            loc_desc_folder = os.path.join(self._desc_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
                desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
                desc_file = self.convert_output_format_polar_to_cartesian_onlyDist(desc_file_polar)
                
                if self._onlyDist:
                    label_mat = self.get_labels_for_file_onlyDist(desc_file, nb_label_frames)
                else:
                    label_mat = self.get_labels_for_file(desc_file, nb_label_frames)

                if self._chan_swap_aug and (int(file_name.split('_')[0][4:]) in self._chan_aug_folds):
                    for suf, _ in aug_list.items():
                        print('{}: {}_{}, {}'.format(file_cnt, file_name, suf, label_mat.shape))
                        np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0]+'_'+suf)), label_mat)
                else:
                    print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                    np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat) """

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
            desc_file = self.load_output_format_file(os.path.join(self._desc_dir, file_name))
            #desc_file = self.convert_output_format_polar_to_cartesian_onlyDist(desc_file)
            
            if self._onlyDist:
                label_mat = self.get_labels_for_file_onlyDist(desc_file, nb_label_frames)
            else:
                label_mat = self.get_labels_for_file(desc_file, nb_label_frames)

            if self._chan_swap_aug and (int(file_name.split('_')[0][4:]) in self._chan_aug_folds):
                for suf, _ in aug_list.items():
                    print('{}: {}_{}, {}'.format(file_cnt, file_name, suf, label_mat.shape))
                    np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0]+'_'+suf)), label_mat)
            else:
                print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 2:  # only dist
                _output_dict[_frame_ind].append([float(_words[1])])
            if len(_words) == 3:  # dist + pred_mask
                _output_dict[_frame_ind].append([float(_words[1]), float(_words[2])])
            if len(_words) == 5: #polar coordinates 
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            elif len(_words) == 6: # cartesian coordinates
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
        _fid.close()
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:

                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict
    
    def convert_output_format_polar_to_cartesian_onlyDist(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    _dist = tmp_val[4]
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], _dist])
        return out_dict
    
    def get_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        se_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        x_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, self._nb_unique_classes))

        # Iterate through each frame in the description file
        for frame_ind, active_event_list in _desc_file.items():
            # If the frame index is less than the number of label frames
            if frame_ind < _nb_label_frames:
                # Iterate through each active event in the frame
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]

        label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
        return label_mat

    def get_labels_for_file_onlyDist(self, _desc_file, _nb_label_frames):
        dist_label = np.zeros((_nb_label_frames, 1))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                for active_event in active_event_list:
                    dist_label[frame_ind, 0] = active_event[1]

        return dist_label

    def write_output_format_file_onlyDist(self, _output_format_file, _output_format_dict, dist_and_mask=False):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
                if dist_and_mask:
                    _fid.write('{},{},{}\n'.format(int(_frame_ind), float(_value[0]), float(_value[1])))
                else:
                    _fid.write('{},{}\n'.format(int(_frame_ind), float(_value[0])))
        _fid.close()


    # ------------------------------- Misc public functions -------------------------------
    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination)
        )
    
    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination)
        )
    
    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir,
               '{}_label'.format('{}_adpit'.format(self._dataset_combination) if self._multi_accdoa else self._dataset_combination)
        )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def delete_and_create_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

