import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import random
import shutil


def prepare_data(config, shuffle=True):
    print('==> Preparing data...')
    if config['data_split'] == 'cross_validation':
        n_fold_split(config)
    elif config['data_split'] == 'default':
        copy_data_split(config, '../default')
    elif config['data_split'] == 'same_as_teacher':
        copy_data_split(config, os.path.join(config['workspace_root'], 'result', config['pretrained_teacher']))

    config['data_dir'] = os.path.join(config['workspace_root'], 'data')
    if config['dataset'] == 'BraTS_2018':
        config['data_dir'] = os.path.join(config['data_dir'], 'MICCAI_BraTS_2018_Data_Training')
    elif config['dataset'] == 'BraTS_2019':
        config['data_dir'] = os.path.join(config['data_dir'], 'MICCAI_BraTS_2019_Data_Training')

    train_set = BratsDataset(config, mode='train')
    valid_set = BratsDataset(config, mode='valid')


    num_workers = 1 if config['debug_mode'] else 2
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=shuffle, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


class BratsDataset(Dataset):
    def __init__(self, config, mode):
        super(BratsDataset, self).__init__()
        self.config = config
        self.subject_list = []
        if mode == 'train':
            dataset_file = os.path.join(config['data_split_dir'], "train_set_fold_%d.txt" % config['fold'])
        elif mode == 'valid':
            dataset_file = os.path.join(config['data_split_dir'], "valid_set_fold_%d.txt" % config['fold'])
        with open(dataset_file, 'r') as f:
            self.subject_list = f.readlines()[0].split(",")

    def load_data(self, index, return_filename=False):
        """To load image data and segmentation annotaions"""
        subject = self.subject_list[index]
        nii_list = []
        for mode in self.config['modality_list']:
            nii_list.append(self.load_nii(subject, mode))
        img = np.array(nii_list, dtype=np.float32)
        seg = np.array(self.load_nii(subject, 'seg'))
        img, seg = self.crop(img, seg)
        if return_filename:
            return img, seg, subject
        else:
            return img, seg

    def crop(self, img, seg):
        if self.config['crop_mode'] == "center":
            c1_l = int((img.shape[1] - self.config['img_size'][0])/2)
            c2_l = int((img.shape[2] - self.config['img_size'][1])/2)
            c3_l = int((img.shape[3] - self.config['img_size'][2])/2)
        elif self.config['crop_mode'] == "random":
            c1_l = random.randint(0, img.shape[1] - self.config['img_size'][0])
            c2_l = random.randint(0, img.shape[2] - self.config['img_size'][1])
            c3_l = random.randint(0, img.shape[3] - self.config['img_size'][2])

        c1_r = int(c1_l + self.config['img_size'][0])
        c2_r = int(c2_l + self.config['img_size'][1])
        c3_r = int(c3_l + self.config['img_size'][2])

        img = img[:, c1_l: c1_r, c2_l: c2_r, c3_l: c3_r]
        seg = seg[c1_l: c1_r, c2_l: c2_r, c3_l: c3_r]

        if self.config['half_size']:
            img = img[:, ::2, ::2, ::2]
            seg = seg[::2, ::2, ::2]
        return img, seg

    def load_nii(self, subject, mode):
        """To load NIFTI file"""
        if self.config['gpu_cluster'] == 'TELECOM' and subject.split('/')[0] != "..":
            subject = "../" + "/".join(subject.split('/')[5:])
        if self.config['gpu_cluster'] == 'SJTU' and subject.split('/')[0] == "..":
            subject = "/DATA7_DB7/data/mhhu/projPRIM/" + "/".join(subject.split('/')[1:])
        subject_path = subject
        subject_name = os.listdir(subject_path)[0].split('.')[0].split('_')[:-1]
        subject_name.append(mode)
        img_full_name = '_'.join(subject_name) + '.nii.gz'
        img_path = os.path.join(subject_path, img_full_name)
        return nib.load(img_path).get_data()

    def __getitem__(self, index):
        if self.config['return_filename']:
            img, seg, filename = self.load_data(index, self.config['return_filename'])
        else:
            img, seg = self.load_data(index)

        label = seg_label(seg, self.config['softmax'])

        # pre-processing & augmentation
        if self.config['random_mirror_flip']:
            img, label = mirror_flip(img, label)
            img_T1_ = torch.from_numpy(img[0][np.newaxis, ...]).type('torch.FloatTensor')
            img_T1ce_ = torch.from_numpy(img[1][np.newaxis, ...]).type('torch.FloatTensor')
            img_T2_ = torch.from_numpy(img[2][np.newaxis, ...]).type('torch.FloatTensor')
            img_Flair_ = torch.from_numpy(img[3][np.newaxis, ...]).type('torch.FloatTensor')
            origin = [img_T1_, img_T1ce_, img_T2_, img_Flair_]

        if self.config['data_preprocessing']:
            img = scale_non_zero(img, config=self.config)
        if self.config['data_augmentation']:
            img = intensity_shift_non_zero(img)

        img_T1 = torch.from_numpy(img[0][np.newaxis, ...]).type('torch.FloatTensor')
        img_T1ce = torch.from_numpy(img[1][np.newaxis, ...]).type('torch.FloatTensor')
        img_T2 = torch.from_numpy(img[2][np.newaxis, ...]).type('torch.FloatTensor')
        img_Flair = torch.from_numpy(img[3][np.newaxis, ...]).type('torch.FloatTensor')
        label = torch.from_numpy(label)
        if self.config['return_filename']:
            return [img_T1, img_T1ce, img_T2, img_Flair], label, filename, origin
        return [img_T1, img_T1ce, img_T2, img_Flair], label

    def __len__(self):
        return len(self.subject_list)


def seg_label(seg, softmax=False):
    """To transform the segmentation to label.

    Args:
        seg: annotations which use different values to segment the data. (0-background, 1-non-enhancing tumor core(NET),
        2-the peritumoral edema(ED), 4-the GD-enhancing tumor(ET)).

        ET = ET
        TC = ET + NET/NCR
        WT = ET + NET/NCR + ED

    Returns:
        A numpy array contains 3 channels(ET, TC, WT). In each channel, pixels are labeled as 0 or 1.
    """

    if softmax:
        label_0 = 1 * (seg == 0)
        label_1 = 1 * (seg == 1)
        label_2 = 1 * (seg == 2)
        label_4 = 1 * (seg == 4)
        labels = [label_0, label_1, label_2, label_4]
    else:
        label_ET = np.zeros(seg.shape)
        label_ET[np.where(seg == 4)] = 1

        label_TC = np.zeros(seg.shape)
        label_TC[np.where((seg == 1) | (seg == 4))] = 1

        label_WT = np.zeros(seg.shape)
        label_WT[np.where(seg > 0)] = 1

        labels = [label_ET, label_TC, label_WT]
    return np.asarray(labels, dtype=np.float32)


def scale_non_zero(imgs, config):
    """Scale each channel to z-score on non-zero voxels only.

    Args:
        img: multi-channel brats image data.

    Returns:
        image after scaling, same size as input.
    """

    for i, img in enumerate(imgs):
        mask = img != 0
        img[(img > config['upper_clip']) & (mask != 0)] = config['upper_clip']
        mean = img[mask].mean()
        std = img[mask].std()
        img = img.astype(dtype=np.float32)
        img[mask] = (img[mask] - mean) / std
        imgs[i] = img

    return imgs


def mirror_flip(img, label):
    """A random axis mirror flip (for all 3 axes) with a probability 0.5.

    Args:
        img: multi-channel brats image data.

    Returns:
        image after random flips, same size as input.
    """
    if random.random() < 0.5:
        img = img[:, ::-1, :, :].copy()
        label = label[:, ::-1, :, :].copy()
    if random.random() < 0.5:
        img = img[:, :, ::-1, :].copy()
        label = label[:, :, ::-1, :].copy()
    if random.random() < 0.5:
        img = img[:, :, :, ::-1].copy()
        label = label[:, :, :, ::-1].copy()
    return img, label


def intensity_shift_non_zero(img):
    """A random (per channel) intensity shift (-0.1 .. 0.1 of image std) on input images.

    Args:
        img: multi-channel brats image data.

    Returns:
        image after random intensity shift, same size as input.
    """
    for i, channel in enumerate(img):
        shift = random.random() * 0.2 - 0.1  # random intensity shift (-0.1..0.1 of image std)
        count = np.zeros(channel.shape)
        count[np.where(channel > 0)] = 1
        img[i] = channel + shift * count
    return img


def n_fold_split(config):
    config['data_dir'] = os.path.join(config['workspace_root'], 'data')
    config['data_dir'] = os.path.join(config['data_dir'], 'MICCAI_%s_Data_Training' % config['dataset'])

    HGG_dir = os.path.join(config['data_dir'], 'HGG')
    LGG_dir = os.path.join(config['data_dir'], 'LGG')

    if config['dataset'] == 'BraTS_2018':
        HGG_patient_18 = os.listdir(HGG_dir)
        LGG_patient_18 = os.listdir(LGG_dir)
    else:
        name_mapping = os.path.join(config['data_dir'], 'name_mapping.csv')
        mapping = np.loadtxt(name_mapping, delimiter=",", dtype='U', usecols=(0, 2, 4))
        HGG_patient_new = [m[2] for m in mapping if m[0] == 'HGG' and m[1] == 'NA']
        LGG_patient_new = [m[2] for m in mapping if m[0] == 'LGG' and m[1] == 'NA']
        HGG_patient_18 = [m[2] for m in mapping if m[0] == 'HGG' and m[1] != 'NA']
        LGG_patient_18 = [m[2] for m in mapping if m[0] == 'LGG' and m[1] != 'NA']
        data_set_new = [os.path.join(HGG_dir, patient) for patient in HGG_patient_new] + [os.path.join(LGG_dir, patient) for patient in LGG_patient_new]

    data_set_HGG = [os.path.join(HGG_dir, patient) for patient in HGG_patient_18]
    data_set_LGG = [os.path.join(LGG_dir, patient) for patient in LGG_patient_18]

    r = random.random
    random.seed(41)
    random.shuffle(data_set_HGG, random=r)
    random.shuffle(data_set_LGG, random=r)

    n = config['num_fold']
    n_fold = [data_set_HGG[i::n] + data_set_LGG[i::n] for i in range(n)]

    if not os.path.exists(config['data_split_dir']):
        os.mkdir(config['data_split_dir'])

    for i in range(n):
        train_set_path = os.path.join(config['data_split_dir'], 'train_set_fold_%d.txt' % (i+1))
        valid_set_path = os.path.join(config['data_split_dir'], 'valid_set_fold_%d.txt' % (i+1))

        valid_set = n_fold[i]
        if config['dataset'] == 'BraTS_2019':
            valid_set += data_set_new
        train_set = list(set(data_set_HGG + data_set_LGG) - set(valid_set))

        with open(train_set_path, 'w') as f:
            f.writelines(",".join(train_set))
        with open(valid_set_path, 'w') as f:
            f.writelines(",".join(valid_set))


def copy_data_split(config, file_dir):
    source = os.path.join(file_dir, "data_split")
    target = os.path.join(config['result_path'], "data_split")
    if not os.path.exists(target):
        shutil.copytree(source, target)


