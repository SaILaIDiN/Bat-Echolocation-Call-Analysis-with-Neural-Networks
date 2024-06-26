import numpy as np
import os
import re


def create_noise_labels(path_file_names, path_labels, dir_noise_wav):
    """ Take the txt file with the noise image filenames,
        create a same named .npy file but with the label as the single entry.
        Either try a number or the one-hot-encoded version.
    """
    with open(path_file_names, "r") as f:
        file_name_list = f.read().split('\n')

    counter = 0
    for noise_sample in file_name_list:
        if len(noise_sample) > 0:
            noise_wav = os.path.join(dir_noise_wav, noise_sample)
            if os.path.isfile(noise_wav):
                noise_label = np.asarray([1, 0])
                np.save(os.path.join(path_labels, noise_sample), noise_label)
                counter += 1
                print(counter)
                if counter > 10000:
                    break


bat_class_list = ['Bbar', 'Myotis', 'Ppip', 'Ptief', 'Pnat', 'Ppyg', 'Plecotus', 'Nnoc', 'Nyctaloid', 'Phoch',
                  'Pipistrelloid', 'Nlei']
# NOTE: This order is not in accordance to my colleague model, but this list still decides the labels for each bat type!


def create_bat_labels(path_file_names, path_labels, dir_bats_wav):
    """ Take the txt file(s) with the bat image filenames,
        create a same named .npy file but with the label as the single entry.
        Either try a number or the one-hot-encoded version.
        NOTE: This is for bat-noise-classification. (2 Classes)
    """

    counter = 0
    for bat_class in bat_class_list:
        bat_call_txt = path_file_names + f"{bat_class}.txt"
        try:
            with open(bat_call_txt, "r") as f:
                file_name_list = f.read().split('\n')
        except:
            continue

        for bat_sample in file_name_list:
            if len(bat_sample) > 0:
                bat_sample_base = re.sub("_Sec_[0-9][0-9]|_Sec_[0-9]", "", bat_sample)
                bat_wav = os.path.join(dir_bats_wav, bat_sample_base)
                if os.path.isfile(bat_wav):
                    bat_label = np.asarray([0, 1])
                    np.save(os.path.join(path_labels, bat_sample), bat_label)
                    counter += 1
                    print(counter)


if __name__ == "__main__":
    from data.path_provider import provide_paths

    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote="local", year=2019)
    # # # Case 1: Noise labels for Noise-Bat-Distinction (2-Class Problem)
    # define a path to store the labels
    dir_noise_labels = r"{arg}\noise\Labels".format(arg=dir_main)
    # Define class "Noise" as [1, 0] and class "Bat" as [0, 1]

    # create_noise_labels(dir_noise_txt, dir_noise_labels)

    # # # Case 2: Bat species labels (All labels stored in one folder for now, for both tasks)
    # define a path to store the labels
    dir_bats_labels = r"{arg}\bat_calls\Labels_".format(arg=dir_main)

    # create_bat_labels(dir_bats_txt, dir_bats_labels)
