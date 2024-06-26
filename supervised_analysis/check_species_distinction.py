""" This script performs the bat species classification/distinction.


    Included classes are:
     - the individual species Pnat, Ppip, Ppyg since their number of examples is enough
     - the representative joined classes (Phoch, Ptief) that contain recordings of Pipistrellus species within the
       frequency interval of 40-42 kHz (Ptief, representing either Pnat or Ppip) and
       50-52 kHz (Phoch, representing either Ppip or Ppyg)

    Excluded classes are:
     - all species and genera that are not in genus Pipistrellus
     - the Pipistrelloid superclass which is containing Phoch and Ptief cases (redundancy)

    Optional:
    Data Recycling can be performed to naturally increase the number of examples per class.
    Especially useful for rare classes.

    Evaluation Phase 1:
    - Cross-Validation between different year orientation pairs will be performed without Ppyg, Phoch,
      because they are underrepresented in general and Phoch is even non existent in East 2019
    - Here the discriminative ability of a network will be checked with respect to changing years and orientation
    - This test should be performed once with and without Ptief to investigate its impact on general model behavior
    - Logged measures are:
      - average f1-score, precision, recall
      - class-based f1-score, (precision, recall)
      - overall accuracy (agnostic to class-distribution)
      - confusion matrix

    Evaluation Phase 2:
    - Merge all years and orientation pairs into one dataset
    - include Ppyg and Phoch as well
    - train and validate and test on the partitions of this one large dataset
    - This test should be performed once with and without Ptief and Phoch to investigate the general model behavior
    - Logged measures are:
      - average f1-score, precision, recall
      - class-based f1-score, (precision, recall)
      - overall accuracy (agnostic to class-distribution)
      - confusion matrix

"""
from torch.utils.data import ConcatDataset
from data.preprocessing.recycle_data import create_joined_partitions_bats, expand_data_partitions
from data.path_provider import provide_paths

# # # Dataset (evaluation phase 1)
# Define the species to investigate
# bat_class_list = ['Ppip', 'Pnat']
# n_classes = 2
# bat_class_list = ['Ppip', 'Pnat', 'Ptief']
# n_classes = 3
# # # # Dataset (evaluation phase 2)
bat_class_list = ['Ppip', 'Pnat', 'Ppyg']
n_classes = 3
# bat_class_list = ['Ppip', 'Pnat', 'Ppyg', 'Ptief', 'Phoch']
# n_classes = 5

spec_type = "MFCC"  # or "LinSpec"
# Create combined dataset per species across all heights of a single orientation
# Year 2019
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="remote", year=2019)
attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]
joined_pip_W_2019 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_W_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_pip_E_2019 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_E_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")


dataset_W_2019 = joined_pip_W_2019
dataset_E_2019 = joined_pip_E_2019
print("Length joined_pip_W_2019: ", len(joined_pip_W_2019[0]))
print("Length joined_pip_E_2019: ", len(joined_pip_E_2019[0]))


# # # Year 2020
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="remote", year=2020)
attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]
joined_pip_W_2020 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_W_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_pip_E_2020 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_E_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")


dataset_W_2020 = joined_pip_W_2020
dataset_E_2020 = joined_pip_E_2020
print("Length joined_pip_W_2020: ", len(joined_pip_W_2020[0]))
print("Length joined_pip_E_2020: ", len(joined_pip_E_2020[0]))

dataset_collector = [joined_pip_W_2019, joined_pip_E_2019, joined_pip_W_2020, joined_pip_E_2020]
joined_dataset_full_pip = expand_data_partitions(dataset_collector)
print("Length full dataset: ", len(ConcatDataset(joined_dataset_full_pip)))


# # # Create the composition of partitions for each test (train_own, val_own, val_foreign, train_foreign+test_foreign)
# 00_Test_0_NW19_train_NW19_test
bat_dataset_00 = [dataset_W_2019[0], dataset_W_2019[1], dataset_W_2019[3], dataset_W_2019[2]]

# 01_Test_1_NE19_train_NW19_test
bat_dataset_01 = [dataset_E_2019[0], dataset_E_2019[1], dataset_W_2019[1],
                  ConcatDataset([dataset_W_2019[0], dataset_W_2019[3]])]
# 02_Test_1_NW19_train_NE19_test
bat_dataset_02 = [dataset_W_2019[0], dataset_W_2019[1], dataset_E_2019[1],
                  ConcatDataset([dataset_E_2019[0], dataset_E_2019[3]])]
# 03_Test_2_NE20_train_NW20_test
bat_dataset_03 = [dataset_E_2020[0], dataset_E_2020[1], dataset_W_2020[1],
                  ConcatDataset([dataset_W_2020[0], dataset_W_2020[3]])]
# 04_Test_2_NW20_train_NE20_test
bat_dataset_04 = [dataset_W_2020[0], dataset_W_2020[1], dataset_E_2020[1],
                  ConcatDataset([dataset_E_2020[0], dataset_E_2020[3]])]
# 05_Test_3_NE19_train_NE20_test
bat_dataset_05 = [dataset_E_2019[0], dataset_E_2019[1], dataset_E_2020[1],
                  ConcatDataset([dataset_E_2020[0], dataset_E_2020[3]])]
# 06_Test_3_NE20_train_NE19_test
bat_dataset_06 = [dataset_E_2020[0], dataset_E_2020[1], dataset_E_2019[1],
                  ConcatDataset([dataset_E_2019[0], dataset_E_2019[3]])]
# 07_Test_4_NW19_train_NW20_test
bat_dataset_07 = [dataset_W_2019[0], dataset_W_2019[1], dataset_W_2020[1],
                  ConcatDataset([dataset_W_2020[0], dataset_W_2020[3]])]
# 08_Test_4_NW20_train_NW19_test
bat_dataset_08 = [dataset_W_2020[0], dataset_W_2020[1], dataset_W_2019[1],
                  ConcatDataset([dataset_W_2019[0], dataset_W_2019[3]])]

# 09_Test_5_Full_train_Full_test
bat_dataset_FULL = [joined_dataset_full_pip[0], joined_dataset_full_pip[1], joined_dataset_full_pip[3],
                    joined_dataset_full_pip[2]]
