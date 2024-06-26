import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import random
import os

bat_class_list = ['Bbar', 'Myotis', 'Ppip', 'Ptief', 'Pnat', 'Ppyg', 'Plecotus', 'Nnoc', 'Nyctaloid', 'Phoch',
                  'Pipistrelloid', 'Nlei']


class BatNoiseDataset(Dataset):
    """ This dataset class is for bat-noise-distinction.
        Args:
            bat_class_list (list): contains a list of specific bat species.
                                   Only MFCCs of these species are collected here.
            sec (int): is the timestamp of which second of a recording the MFCC stems from.
            dir_bats_txt (str): contains the main path for the bat txts of all species, but will be completed based on
                                the given timestamp parameter "sec".
                                Only MFCCs of this timestamp will be collected here.
            attr_oh (str): contains the abbreviation of orientation and height of the recorded data for path correction.
        NOTE: Bat calls and noise are both stored separately.
              Thus, have to be joined explicitly into the same variable.
              With this logistics of timestamps ("sec") and species list ("bat_class_list"), MFCCs and labels have to
               be created only once and are then efficiently referred by txt-files to form distinct Dataset instances.
              Another advantage is that these txt-files can be created quickly with customised conditions for their
               entries.
              Pytorch is also providing a fast way to combine multiple Dataset instance into one ConcatDataset().
    """

    def __init__(self, dir_main, dir_bats_txt, bat_class_list, dir_noise_txt, sec="", attr_oh="", shuffle=False):
        super().__init__()
        self._dir_main = dir_main

        self._bat_dir = os.path.join(os.path.join(dir_main, 'bat_calls'), attr_oh)
        self._bat_dir_labels = os.path.join(self._bat_dir, f'Labels_{sec}')
        self._bat_dir_MFCCs = os.path.join(self._bat_dir, f'MFCCs_{sec}')

        self._noise_dir = os.path.join(os.path.join(dir_main, 'noise'), attr_oh)
        self._noise_dir_labels = os.path.join(self._noise_dir, 'Labels')
        self._noise_dir_MFCCs = os.path.join(self._noise_dir, 'MFCCs')

        # # self.img_ids = []  # receives the identity of each image
        self.imgs = []  # receives the path of each image
        self.labels = []   # receives the path of each label

        self.shuffle = shuffle  # shuffle final arrays with paths of images and labels in unison, randomize before split

        # start with adding all paths of bat images and labels to two arrays
        for i, bat_class in enumerate(bat_class_list):
            bat_call_txt = dir_bats_txt + f"{bat_class}.txt"
            with open(bat_call_txt, "r") as f:
                file_name_list = f.read().split('\n')
            for bat_sample in file_name_list:
                if len(bat_sample) > 0:
                    self.imgs.append(os.path.join(self._bat_dir_MFCCs, bat_sample + '.npy').replace(os.sep, '/'))
                    self.labels.append(os.path.join(self._bat_dir_labels, bat_sample + '.npy').replace(os.sep, '/'))

        # continue with adding all paths of noise images and labels to the same two arrays
        with open(dir_noise_txt, "r") as f:
            file_name_list = f.read().split('\n')
        for noise_sample in file_name_list:
            if len(noise_sample) > 0:
                self.imgs.append(os.path.join(self._noise_dir_MFCCs, noise_sample + '.npy').replace(os.sep, '/'))
                self.labels.append(os.path.join(self._noise_dir_labels, noise_sample + '.npy').replace(os.sep, '/'))

        if self.shuffle:
            permutation = np.random.permutation(len(self.imgs))
            self.imgs, self.labels = np.asarray(self.imgs), np.asarray(self.labels)
            self.imgs = self.imgs[permutation]
            self.labels = self.labels[permutation]
            self.imgs, self.labels = self.imgs.tolist(), self.labels.tolist()

        assert (len(self.imgs) == len(self.labels))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        _img = np.load(self.imgs[idx])
        _lbl = np.load(self.labels[idx])
        sample = {'image': _img, 'label': _lbl}
        # sample['image_path'] = self.imgs[idx]
        # sample['label_path'] = self.labels[idx]
        # # sample['id'] = self.img_ids[idx]
        return sample


class BatDataset(Dataset):
    """ This dataset class is for performance tests on only specific bat species.
        For more precise explanations of the initial arguments check BatNoiseDataset() above.
    """

    def __init__(self, dir_main, dir_bats_txt, bat_class_list, sec="", attr_oh="", spec_type="MFCC"):
        super().__init__()
        self._dir_main = dir_main

        self._bat_dir = os.path.join(os.path.join(dir_main, 'bat_calls'), attr_oh)
        self._bat_dir_labels = os.path.join(self._bat_dir, f'Labels_{sec}')
        if spec_type == "MFCC":
            self._bat_dir_specs = os.path.join(self._bat_dir, f'MFCCs_{sec}')
        elif spec_type == "LinSpec":
            self._bat_dir_specs = os.path.join(self._bat_dir, f'LinSpecs_{sec}')
        else:
            print("Undefined type of spectrogram \"spec_type\"!")
            return
        self.imgs = []  # receives the path of each image
        self.labels = []  # receives the path of each label

        self.species_labels = []  # receives the bat species of the current image (string)

        # start with adding all paths of bat images and labels to two arrays
        for i, bat_class in enumerate(bat_class_list):
            bat_call_txt = dir_bats_txt + f"{bat_class}.txt"
            with open(bat_call_txt, "r") as f:
                file_name_list = f.read().split('\n')
            for bat_sample in file_name_list:
                if len(bat_sample) > 0:
                    self.imgs.append(os.path.join(self._bat_dir_specs, bat_sample + '.npy').replace(os.sep, '/'))
                    self.labels.append(os.path.join(self._bat_dir_labels, bat_sample + '.npy').replace(os.sep, '/'))
                    self.species_labels.append(bat_class)

        assert (len(self.imgs) == len(self.labels))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        _img = np.load(self.imgs[idx])
        _lbl = np.load(self.labels[idx])
        sample = {'image': _img, 'label': _lbl}
        sample['image_path'] = self.imgs[idx]
        # sample['label_path'] = self.labels[idx]
        sample["text_label"] = self.species_labels[idx]
        return sample


class BatDatasetRecycle(Dataset):
    """ This dataset class is for training and testing on bat calls. (for specific orientation and height "attr_oh")
        Handles multiple species in one session based on "bat_class_list".
        It enables an efficient way of creating balanced datasets for species distinction.
        For "dir_bats_txt", insert a txt-file that contains all filenames of a species over all seconds.
    """
    def __init__(self, dir_main, dir_bats_txt, bat_class_list, n_per_species=20, attr_oh="", spec_type="MFCC"):
        super().__init__()
        self._dir_main = dir_main
        self._bat_dir = os.path.join(os.path.join(dir_main, 'bat_calls'), attr_oh)

        self.imgs = []  # receives the path of each image
        self.labels = []  # receives the path of each label

        self.species_labels = []  # receives the bat species of the current image (string)

        # start with adding all paths of bat images and labels to two arrays
        for i, bat_class in enumerate(bat_class_list):
            bat_call_txt = dir_bats_txt + f"{bat_class}.txt"
            try:
                with open(bat_call_txt, "r") as f:
                    pass
            except FileNotFoundError:
                print(f"The file {bat_call_txt} does not exist!")
                continue

            with open(bat_call_txt, "r") as f:
                file_name_list = f.read().split('\n')
                random.shuffle(file_name_list)
                if len(file_name_list) >= n_per_species:
                    file_name_list = file_name_list[0: n_per_species]
                else:
                    print(f"Subset is smaller than {n_per_species}! "
                          f"Length of species {bat_class} is {len(file_name_list)}.")

            for bat_sample in file_name_list:
                if len(bat_sample) > 0:  # bypass any kind of empty line inside the txt-file
                    sec = bat_sample.split("_Sec_", 1)[1]  # works for all seconds/series of digits
                    self._bat_dir_labels = os.path.join(self._bat_dir, f'Labels_{sec}')
                    if spec_type == "MFCC":
                        self._bat_dir_specs = os.path.join(self._bat_dir, f'MFCCs_{sec}')
                    elif spec_type == "LinSpec":
                        self._bat_dir_specs = os.path.join(self._bat_dir, f'LinSpecs_{sec}')
                    else:
                        print("Undefined type of spectrogram \"spec_type\"!")
                        return

                    self.imgs.append(os.path.join(self._bat_dir_specs, bat_sample + '.npy').replace(os.sep, '/'))
                    self.labels.append(os.path.join(self._bat_dir_labels, bat_sample + '.npy').replace(os.sep, '/'))
                    self.species_labels.append(bat_class)

        assert (len(self.imgs) == len(self.labels))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        _img = np.load(self.imgs[idx])
        _lbl = np.load(self.labels[idx])
        sample = {'image': _img, 'label': _lbl}
        sample['image_path'] = self.imgs[idx]
        # sample['label_path'] = self.labels[idx]
        sample['text_label'] = self.species_labels[idx]
        return sample


class NoiseDataset(Dataset):
    """ This dataset class is for performance tests on only the noise class.
        For more precise explanations of the initial arguments check BatNoiseDataset() above.
    """

    def __init__(self, dir_main, dir_noise_txt, attr_oh="", spec_type="MFCC"):
        super().__init__()
        self._dir_main = dir_main

        self._noise_dir = os.path.join(os.path.join(dir_main, 'noise'), attr_oh)
        self._noise_dir_labels = os.path.join(self._noise_dir, 'Labels')
        if spec_type == "MFCC":
            self._noise_dir_specs = os.path.join(self._noise_dir, 'MFCCs')
        elif spec_type == "LinSpec":
            self._noise_dir_specs = os.path.join(self._noise_dir, 'LinSpecs')
        else:
            print("Undefined type of spectrogram \"spec_type\"!")
            return
        self.imgs = []  # receives the path of each image
        self.labels = []  # receives the path of each label

        # continue with adding all paths of noise images and labels to the two arrays
        with open(dir_noise_txt, "r") as f:
            file_name_list = f.read().split('\n')
        for noise_sample in file_name_list:
            if len(noise_sample) > 0:
                self.imgs.append(os.path.join(self._noise_dir_specs, noise_sample + '.npy').replace(os.sep, '/'))
                self.labels.append(os.path.join(self._noise_dir_labels, noise_sample + '.npy').replace(os.sep, '/'))

        assert (len(self.imgs) == len(self.labels))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        _img = np.load(self.imgs[idx])
        _lbl = np.load(self.labels[idx])
        sample = {'image': _img, 'label': _lbl}
        sample['image_path'] = self.imgs[idx]
        # sample['label_path'] = self.labels[idx]
        sample['text_label'] = "Noise"  # (preventive:) keep compatibility with BatDataset for Concat
        return sample


def partition_dataset(dataset, train_size, val_size, train_sub_size, toy_factor=None, data_limit_num=None):
    """ Creates three main partitions of train, val and test. Randomized split.
        Additionally creates one sub partition of train, called train_subset_1.
        This sub-partition can be used to efficiently compute evaluation metrics that represent the full training data.
        Used to detect overfitting faster.
        "data_limit_num" should be an integer. Reduce the overall size before splitting.
    """
    # # # Define Toy Size Dataset from original data to check automation procedure
    if toy_factor is not None and type(toy_factor) == float:
        dataset, _ = random_split(dataset, [int(len(dataset) * toy_factor),
                                            int(len(dataset)) - int(len(dataset) * toy_factor)])
    # # # End
    if type(data_limit_num) is int:
        # print("Assertion-print: ", len(dataset), data_limit_num)
        assert(len(dataset) >= data_limit_num)
        dataset = Subset(dataset, random.sample(range(0, len(dataset)), data_limit_num))
    train_set, val_test_set = random_split(dataset, [int(len(dataset) * train_size),
                                                     int(len(dataset)) - int(len(dataset) * train_size)])
    val_set, test_set = random_split(val_test_set, [int(len(val_test_set) * val_size * 2),
                                                    int(len(val_test_set)) - int(len(val_test_set) * val_size * 2)])
    train_subset_1, _ = random_split(train_set, [int(len(train_set) * train_sub_size),
                                                 int(len(train_set)) - int(len(train_set) * train_sub_size)])
    partitions = []
    partitions.append(train_set), partitions.append(val_set), \
        partitions.append(train_subset_1), partitions.append(test_set)
    return partitions


if __name__ == "__main__":
    from torch.utils.data import DataLoader, ConcatDataset
    from data.path_provider import dir_results, dir_results_remote, provide_paths

    system_mode = input("Enter system mode. Choose either 'local_mode' or 'remote_mode': ")
    if system_mode == "local_mode":
        path_prefix = dir_results
        main_paths = "argparse_configs/mp_local_2019.txt"
        local_or_remote = "local"
    elif system_mode == "remote_mode":
        path_prefix = dir_results_remote
        main_paths = "argparse_configs/mp_remote_2019.txt"
        local_or_remote = "remote"
    else:
        path_prefix = "CHANGE_DIR"

    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote=local_or_remote, year=2019)

    # Define dataset instance
    full_set_Noise = NoiseDataset(dir_main, dir_noise_txt, "W_05", "MFCC")
    full_set_Bats = BatDataset(dir_main, dir_bats_txt, ["Myotis"], sec=0 + 1, attr_oh="W_05", spec_type="MFCC")
    full_set = ConcatDataset([full_set_Bats, full_set_Noise])
    partitions = partition_dataset(full_set, train_size=0.6, val_size=0.2, train_sub_size=0.05, toy_factor=0.3)
    train_loader = DataLoader(partitions[0], batch_size=1, shuffle=True, num_workers=0)

    for i in range(10):
        batch = next(iter(train_loader))
        MFCC = batch["image"]
        label = batch["label"]
        print(MFCC)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(label)

        # images_path = batch["image_path"]
        # labels_path = batch["label_path"]
        #
        # print("images_path:", images_path, i)
        # print("labels_path:", labels_path, i)

    for i, batch in enumerate(train_loader, 0):
        MFCC = batch["image"]
        label = batch["label"]
        # print(MFCC)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(label)

        # images_path = batch["image_path"]
        # labels_path = batch["label_path"]
        #
        # print("images_path:", images_path, i)
        # print("labels_path:", labels_path, i)
