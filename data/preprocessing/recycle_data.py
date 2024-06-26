""" This file creates a balanced dataset from recycled bat calls.
    First: create the txt-files with proper number of calls per class using combinations over several seconds.
    Second: create all necessary Dataset class objects based on these txt-files.
    Third: combine all Datasets into one and start training and testing.
"""
import pandas as pd
from data.preprocessing.data_separator import noise_file_name_txt
from data.preprocessing.data_separator import clean_bat_recordings_table, clean_noise_recordings_table
from data.dataset import BatDataset, BatDatasetRecycle, NoiseDataset, partition_dataset
from torch.utils.data import ConcatDataset
import random


def create_balanced_data_bats(bat_species, n_per_species, attr_oh, dir_data_txts, year, dir_main, spec_type):
    """ Create a balanced Dataset for a single bat species. (Uses BatDatasetRecycle())
        Is currently randomly sampling from a txt-file with all recycled recordings up to 10 seconds.
        Is limited to one combination of orientation and height of the measurements. ("attr_oh")
    """
    dir_bats_txt = r"{dir_data_txts}\data_{year}\{attr_oh}\Bat_type_{arg}_".format(
        dir_data_txts=dir_data_txts, year=year, attr_oh=attr_oh, arg="Combined")
    dataset = BatDatasetRecycle(dir_main, dir_bats_txt, [bat_species], n_per_species, attr_oh, spec_type)
    return dataset


def create_stretched_data_bats(bat_species, n_sec, attr_oh, dir_data_txts, year, dir_main, spec_type):
    """ Create a merged Dataset for a single bat species across the first n_sec seconds.
        Even if the number of recycled seconds "n_sec" impacts the species distribution, when multiple
         species are combined, it still almost preserves the real population.
    """
    dataset_list = []
    for sec in range(n_sec):
        dir_bats_txt = r"{dir_data_txts}\data_{year}\{attr_oh}\Bat_type_{sec}_".format(
            dir_data_txts=dir_data_txts, year=year, attr_oh=attr_oh, sec=sec+1)
        try:
            dataset_tmp = BatDataset(dir_main, dir_bats_txt, [bat_species], sec=sec+1, attr_oh=attr_oh,
                                     spec_type=spec_type)
            dataset_list.append(dataset_tmp)
            print("File/Directory found: ", dir_bats_txt)
        except FileNotFoundError:
            print("File/Directory not found: ", dir_bats_txt)
    try:
        merged_dataset = ConcatDataset(dataset_list)
        return merged_dataset
    except AssertionError:
        print("dataset_list is empty. Cannot create a ConcatDataset.")
        return []


def create_join_heights_orientation_bats(bat_list, n_per_species, n_sec, attributes, dir_data_txts, year, dir_main,
                                         spec_type, pre_balanced=False):
    """ Create a merged Dataset for a list of species from all heights and orientations within a year.
        Can process "create_stretched_data_bats()" for recycled recordings with preserved population
         and "create_balanced_data_bats()" for recycled recordings and controlled number of samples per species.
        Note: Since not all heights contain all species of "bat_list" the balancing effect by "n_per_species" can get
              lost with the joining process over heights.
              Therefore it must be corrected in the step of partitioning the data.
    """
    dataset_list = []
    for attr in attributes:
        for bat_class in bat_list:
            if pre_balanced:
                dataset_tmp = create_balanced_data_bats(bat_species=bat_class, n_per_species=n_per_species,
                                                        attr_oh=attr, dir_data_txts=dir_data_txts, year=year,
                                                        dir_main=dir_main, spec_type=spec_type)
            else:
                dataset_tmp = create_stretched_data_bats(bat_species=bat_class, n_sec=n_sec,
                                                         attr_oh=attr, dir_data_txts=dir_data_txts, year=year,
                                                         dir_main=dir_main, spec_type=spec_type)
            dataset_list.append(dataset_tmp)
    try:
        merged_dataset = ConcatDataset(dataset_list)
        return merged_dataset
    except AssertionError:
        print("dataset_list is empty for Bats. Cannot create a ConcatDataset.")
        return []


def create_join_heights_orientation_noise(csv_path_main, size, attributes, dir_data_txts, year, dir_main, spec_type):
    """ Create a merged Dataset for noise from all heights and orientations within a year. """
    dataset_list = []
    for attr in attributes:
        if year == 2019:
            csv_path = r"{csv_path_main}\evaluation_{attr}m_05_19-11_19.csv".format(csv_path_main=csv_path_main,
                                                                                    attr=attr)
        elif year == 2020:
            csv_path = r"{csv_path_main}\evaluation_{attr}m_04_20-11_20.csv".format(csv_path_main=csv_path_main,
                                                                                    attr=attr)
        # Create a table from a csv
        recordings_table = pd.read_csv(csv_path)
        # Create a table of noise only from the csv
        noise_table = clean_noise_recordings_table(recordings_table)
        # Check the alignment of labeled noise filenames and existent labels/wav-files
        dir_noise_wl = r"{dir_main}\noise\{attr}\Labels".format(dir_main=dir_main, attr=attr)
        noise_size = noise_file_name_txt(noise_table, dir_data_txts, year, dir_noise_wl, attr, size)
        dir_noise_txt = r"{dir_data_txts}\data_{year}\{attr}\Noise_file_names{noise_size}.txt".format(
            dir_data_txts=dir_data_txts, year=year, attr=attr, noise_size=noise_size)
        # Create a dataset of noise files from specific height and orientation via "attr"
        dataset_tmp = NoiseDataset(dir_main, dir_noise_txt, attr, spec_type)
        dataset_list.append(dataset_tmp)

    try:
        merged_dataset = ConcatDataset(dataset_list)
        return merged_dataset
    except AssertionError:
        print("dataset_list is empty for Noise. Cannot create a ConcatDataset.")
        return []


def count_len_joined_txt_file(dir_txt_file, n_samples):
    """ Count the number of non empty lines in a txt-file of joined heights in same orientation.
        Because empty lines are kept in the joined txt-files for readability.
    """
    with open(dir_txt_file, "r") as f:
        file_name_list = f.read().split('\n')
        random.shuffle(file_name_list)
        if len(file_name_list) >= n_samples:
            file_name_list = file_name_list[0: n_samples]
        else:
            pass
    file_name_list = [x for x in file_name_list if len(x) > 0]
    return len(file_name_list)


def return_min_size_of_all_joined_txt_files_bats(bat_list, n_per_species, dir_data_txts, year, txt_postfix, orientation):
    """ Required to use "data_limit" in automated manner.
        "orientation" = "West" / "East"
    """
    dir_bats_txt = r"{dir_data_txts}\data_{year}\Bat_type_{arg}_".format(
        dir_data_txts=dir_data_txts, year=year, arg=f"{txt_postfix}_{orientation}")
    len_of_joined_txt_files_collector = []
    for bat_class in bat_list:
        bat_call_txt = dir_bats_txt + f"{bat_class}.txt"
        try:
            with open(bat_call_txt, "r") as f:
                pass
        except FileNotFoundError:
            print(f"The file {bat_call_txt} does not exist!")
            continue
        len_of_joined_txt_files_collector.append(count_len_joined_txt_file(bat_call_txt, n_samples=n_per_species))
    return min(len_of_joined_txt_files_collector)


def create_joined_partitions_bats(bat_list, n_per_species, n_sec, attributes, dir_data_txts, year, dir_main,
                                  train_size, val_size, train_sub_size, toy_factor=None, spec_type=None,
                                  pre_balanced=False, data_limit_auto=False, data_limit_num=None, orientation=None):
    """ "data_limit*" is correcting the lost class-balancing impact from "n_per_species" after the Dataset instances
         of a species is joined along the heights at the same orientation.
         "data_limit_auto" performs automatic computation of highest appropriate data_limit.
         It maximises the smallest number of examples per species, to get the largest perfectly balanced set.
         "data_limit_num" can be set manually if "data_limit_auto" is set to False.
         NOTE: "data_limit_auto" is only working for the case of joined heights at fixed orientation! (Required case)
               But any concatenated Dataset build upon Datasets of joint heights will maintain perfect balance.
    """
    partition_container_train = []
    partition_container_val = []
    partition_container_train_sub = []
    partition_container_test = []
    if data_limit_auto and len(attributes) == 4:
        data_limit_num = return_min_size_of_all_joined_txt_files_bats(bat_list, n_per_species, dir_data_txts, year,
                                                                      txt_postfix="Combined_all_Heights",
                                                                      orientation=orientation)
    for bat_class in bat_list:
        # "dataset_sub_tmp" comprises all filenames of a certain bat species over all heights and recycled seconds
        # for a given orientation and only if "n_per_species" is large enough
        dataset_sub_tmp = create_join_heights_orientation_bats(bat_list=[bat_class], n_per_species=n_per_species,
                                                               n_sec=n_sec, attributes=attributes,
                                                               dir_data_txts=dir_data_txts, year=year,
                                                               dir_main=dir_main, pre_balanced=pre_balanced,
                                                               spec_type=spec_type)
        partitions = partition_dataset(dataset_sub_tmp, train_size, val_size, train_sub_size, toy_factor=toy_factor,
                                       data_limit_num=data_limit_num)
        partition_container_train.append(partitions[0])
        partition_container_val.append(partitions[1])
        partition_container_train_sub.append(partitions[2])
        partition_container_test.append(partitions[3])

    merged_train = ConcatDataset(partition_container_train)
    merged_val = ConcatDataset(partition_container_val)
    merged_train_sub = ConcatDataset(partition_container_train_sub)
    merged_test = ConcatDataset(partition_container_test)
    return merged_train, merged_val, merged_train_sub, merged_test


def create_joined_partitions_noise(csv_path_main, size, attributes, dir_data_txts, year, dir_main,
                                   train_size, val_size, train_sub_size, toy_factor=None, spec_type=None):
    """ """
    dataset = create_join_heights_orientation_noise(csv_path_main=csv_path_main, size=size, attributes=attributes,
                                                    dir_data_txts=dir_data_txts, year=year, dir_main=dir_main,
                                                    spec_type=spec_type)
    partitions = partition_dataset(dataset, train_size, val_size, train_sub_size, toy_factor=toy_factor)
    return partitions[0], partitions[1], partitions[2], partitions[3]


def expand_data_partitions(dataset_list):
    """ Combines same data partitions of multiple partitioned datasets.
        dataset_list contains a list of datasets which are already partitioned into train, val, train_sub and test.
        Expected use-cases: all year and orientation pairs or bat and noise. """
    partition_container_train = []
    partition_container_val = []
    partition_container_train_sub = []
    partition_container_test = []

    for dataset in dataset_list:
        partition_container_train.append(dataset[0])
        partition_container_val.append(dataset[1])
        partition_container_train_sub.append(dataset[2])
        partition_container_test.append(dataset[3])

    merged_train = ConcatDataset(partition_container_train)
    merged_val = ConcatDataset(partition_container_val)
    merged_train_sub = ConcatDataset(partition_container_train_sub)
    merged_test = ConcatDataset(partition_container_test)
    return merged_train, merged_val, merged_train_sub, merged_test


if __name__ == "__main__":
    from data.preprocessing.data_conditioning import create_bat_tables_by_time_stamps, create_bat_txts_by_time_stamps
    from data.path_provider import provide_paths

    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote="local", year=2019)

    dir_bats_wav = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Bats_only_data".format(arg=dir_raw_data)
    csv_path = r"{arg}\evaluation_{k}m_05_19-11_19.csv".format(k="W_05", arg=csv_path_main)
    recordings_table = pd.read_csv(csv_path)
    bats_only_table = clean_bat_recordings_table(recordings_table)
    noise_table = clean_noise_recordings_table(recordings_table)
    #
    bats_table_collection = create_bat_tables_by_time_stamps(bats_only_table, 10, lookahead=0.8)
    [print(len(k_sub)) for k_sub in bats_table_collection]
    create_bat_txts_by_time_stamps(bats_table_collection, dir_data_txts, year=2019, dir_bats_wav=dir_bats_wav,
                                   path_spec="W_05", size=400)

    #
    bat_class_list_1 = ['Bbar', 'Myotis', 'Ppip', 'Ptief', 'Pnat', 'Ppyg', 'Plecotus', 'Nnoc', 'Nyctaloid']


