import os

        # "1": "A_Carotid_L",
        # "2": "A_Carotid_R",
        # "3": "Arytenoid",
        # "4": "Bone_Mandible",
        # "5": "Brainstem",
        # "6": "BuccalMucosa",
        # "7": "Cavity_Oral",
        # "8": "Cochlea_L",
        # "9": "Cochlea_R",
        # "10": "Cricopharyngeus",
        # "11": "Esophagus_S",
        # "12": "Eye_AL",
        # "13": "Eye_AR",
        # "14": "Eye_PL",
        # "15": "Eye_PR",
        # "16": "Glnd_Lacrimal_L",
        # "17": "Glnd_Lacrimal_R",
        # "18": "Glnd_Submand_L",
        # "19": "Glnd_Submand_R",
        # "20": "Glnd_Thyroid",
        # "21": "Glottis",
        # "22": "Larynx_SG",
        # "23": "Lips",
        # "24": "OpticChiasm",
        # "25": "OpticNrv_L",
        # "26": "OpticNrv_R",
        # "27": "Parotid_L",
        # "28": "Parotid_R",
        # "29": "Pituitary",
        # "30": "SpinalCord"
# Preprocessing using preserved HU in dilated part of mask
data_config = {
    'raw_dir':  None,
    'data_dir': '/code/Compare/UaNet/data_1/raw',
    'preprocessed_data_dir': '/code/Compare/UaNet/data_1/preprocessed',
    # 28 OAR names, names from the original dicom RT
    'roi_names': ['A_Carotid_L', 'A_Carotid_R', 'Arytenoid', 'Bone_Mandible', 'Brainstem', 'BuccalMucosa', 'Cavity_Oral',
        'Cochlea_L', 'Cochlea_R', 'Cricopharyngeus', 'Esophagus_S', 'Eye_AL', 'Eye_AR', 'Eye_PL',
        'Eye_PR', 'Glnd_Lacrimal_L', 'Glnd_Lacrimal_R', 'Glnd_Submand_L', 'Glnd_Submand_R', 'Glnd_Thyroid', 'Glottis',
        'Larynx_SG', 'Lips', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R', 'Pituitary', 'SpinalCord'],
    
    # name used for legend for the figures in the paper, for better consistency
    'paper_roi_names': ['A_Carotid_L', 'A_Carotid_R', 'Arytenoid', 'Bone_Mandible', 'Brainstem', 'BuccalMucosa', 'Cavity_Oral',
        'Cochlea_L', 'Cochlea_R', 'Cricopharyngeus', 'Esophagus_S', 'Eye_AL', 'Eye_AR', 'Eye_PL',
        'Eye_PR', 'Glnd_Lacrimal_L', 'Glnd_Lacrimal_R', 'Glnd_Submand_L', 'Glnd_Submand_R', 'Glnd_Thyroid', 'Glottis',
        'Larynx_SG', 'Lips', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R', 'Pituitary', 'SpinalCord'],

    # data configuration
    # maximum z, x, y slices to load, in order to reduce data preparation time
    # These numbers are chosen according to the max_crop_size and jitter
    # since the max input would be centered at the image with size train_max_crop_size,
    # there is no need to load more than that.
    'num_slice': 180,
    'num_x': 272,
    'num_y': 272,

    # maximum input size to the network
    'train_max_crop_size': [112, 240, 240], 
    'bbox_border': 8,
    'pad_value': -1024,
    'jitter_range': [4, 16, 16],
    'stride': [16, 32, 32],
    'test_max_size': [256, 320, 320], 

    # whether to do affine and elastic transformation
    'do_elastic': True
    ,
    'do_postprocess': False,
}


def get_anchors(bases, aspect_ratios):
    """
    Generate anchor according to the scale and aspect ratios

    bases: the scale for each anchor box
    aspect ratios: d:h:w for each anchor box
    """
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors


bases = [7, 15, 30, 50]
aspect_ratios = [[1, 2.5, 2.5], [1, 2.5, 5.], [1, 5., 2.5]]

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),

    # # of input channel, since it is CT image, we only have one channel
    'chanel': 1,

    # The feature map used for detection is a downsampled by stride
    'stride': 8,

    # The smallest feature map in the network is downsampled by max_stride
    'max_stride': 16,

    # Random sample num_neg negative samples for rpn proposals
    'num_neg': 80000,

   
    'rpn_train_bg_thresh_high': 0.1,
    'rpn_train_fg_thresh_low': 0.5,
    
    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.5,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.5,

    # detection network configuration
    # extra 1 for background
    'num_class': len(data_config['roi_names']) + 1,

    # ROI pooling size
    'rcnn_crop_size': (6, 6, 6),
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.2,
    'rcnn_train_batch_size': 128,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.5,
    'rcnn_test_nms_pre_score_threshold': 0.5,
    'rcnn_test_nms_overlap_threshold': 0.5,

    # controlling the strength of bounding box regression losses
    'box_reg_weight': [10., 10., 10., 5., 5., 5.]
}


def lr_shedule(epoch, init_lr=0.001, total=100):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

train_config = {
    'net': 'UaNet',
    'batch_size': 1,

    'lr_schedule': lr_shedule,
    'optimizer': 'Adam',
    'momentum' : 0.9,
    'weight_decay': 1e-4,

    # total # of epochs
    'epochs': 100,

    # save check point (model weights) every epoch_save epochs
    'epoch_save': 1,

    # starting epoch_rcnn, add the rcnn branch for training
    'epoch_rcnn': 20,

    # starting epoch_mask, add the mask branch for training
    'epoch_mask': 25,

    # num_workers for data loader
    'num_workers': 4,

    # training data is the combination of dataset1, dataset2 training data (A total of 215)
    # Because of the patient privacy, the access to the training data in dataset 1 
    # will be granted on a case by case basis by submitting a request to the corresponding authors, 
    # subjecting to the review and approval by IRB.
    # 
    # validation data was extracted from the training data, around 10%
    # 
    # TODO:
    # You will have to generate your own training data somehow, in order to train the model
    # put all the training filenames into csv, and set the csv path here
    'train_set_name': 'split/release_mydataset_train.csv',
    'val_set_name': 'split/release_mydataset_val.csv',
    'test_set_name': 'split/release_mydataset_train.csv',
    'DATA_DIR': data_config['preprocessed_data_dir'],
    'ROOT_DIR': os.getcwd()
}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3


train_config['RESULTS_DIR'] = os.path.join(train_config['ROOT_DIR'], 'results')
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'experiment_5')
train_config['initial_checkpoint'] = None #train_config['out_dir'] + '/model/076.ckpt'

config = dict(data_config, **net_config)
config = dict(config, **train_config)
