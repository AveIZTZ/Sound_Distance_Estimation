def get_params(argv='1'):
    print("SET: {}".format(argv))
    params = dict(
        # INPUT PATH
        dataset_dir   ='/home/yujiezhu/data/data_for_SED/test/input/',
        # OUTPUT PATHS
        feat_label_dir='/home/yujiezhu/data/data_for_SED/test/processed/',

        # DATASET LOADING PARAMETERS
        dataset='mic',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=16000,
        hop_len_s=0.016,
        label_hop_len_s=0.08,
        nb_mel_bins=64,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA

        # distance
        only_dist=True,

        # synth_and_real_dcase=False,# use synth and modified real dcase dataset
        chan_swap_aug=False, # use channel swap augmentation
        chan_aug_folds=[1,3],
        unique_classes = 13
    )

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params