""" This script compares the data from 2019 with the data from 2020. (Bat-Genus-Distinction)
    Different cases are investigated:
     - the species classification between Northeast 2019 and Northwest 2019 (all heights combined)
     - the species classification between Northeast 2020 and Northwest 2020 (all heights combined)
     - the species classification between Northeast 2019 and Northeast 2020 (all heights combined)
     - the species classification between Northwest 2019 and Northwest 2020 (all heights combined)

    The first evaluation phase would be to simply check the performance metrics by their pure numbers:
     - average f1-score, precision, recall, accuracy
     - class-based f1-score, precision, recall, accuracy

    Included classes are:
     - the two superclasses/genus (Gattung) Myotis and Plecotus since no single species distinction was possible
       for the human expert,
     - the superclass (Artengruppe) Nyctaloid since it has much more examples than Nnoc
       which increases the distribution of useful features in the training data
     - the individual species Pnat, Ppip, Ppyg
     - representative joined classes (Phoch, Ptief, Pipistrelloid) of Pipistrellus

    Excluded classes are:
     - Nlei, Nnoc, Bbar.

    Optional (but highly recommended, since we compare different years and orientations):
    Data Recycling can be performed to naturally increase the number of examples per class.
    Especially useful for rare classes.

    Additional dataset composition:
    - combines all heights and orientations of both years into one pool of data to maximize
      the number of examples per species.

"""
from torch.utils.data import ConcatDataset
from data.preprocessing.recycle_data import create_joined_partitions_bats, expand_data_partitions
from data.path_provider import provide_paths

# # # Dataset
# Define the species to investigate
# bat_class_list = ['Bbar', 'Myotis', 'Ppip', 'Ptief', 'Pnat', 'Ppyg', 'Plecotus', 'Nnoc', 'Nyctaloid', 'Phoch',
#                   'Pipistrelloid', 'Nlei']
# bat_class_list = ['Plecotus', 'Myotis', 'Bbar', 'Pnat', 'Ppip', 'Ppyg', 'Nyctaloid']

# Create precedent/preliminary Dataset of the already complete superclasses 'Nyctaloid' and 'Myotis'
# All Pipistrellus classes will be merged into one class (species are randomly but equally partitioned in separation)
# In the training and evaluation phase, the pipistrellus species will be treated as a single genus/superclass
bat_class_list = ['Nyctaloid', 'Myotis', 'Plecotus', 'Ppip', 'Ptief', 'Phoch', 'Pnat', 'Ppyg', 'Pipistrelloid']
n_classes = 4

spec_type = "MFCC"  # or "LinSpec"
# Create combined dataset per species across all heights of a single orientation
# Year 2019
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="remote", year=2019)
attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]
joined_bats_W_2019 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_W_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_bats_E_2019 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_E_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")

dataset_W_2019 = joined_bats_W_2019
dataset_E_2019 = joined_bats_E_2019
print("Length joined_bats_W_2019: ", len(joined_bats_W_2019[0]))
print("Length joined_bats_E_2019: ", len(joined_bats_E_2019[0]))


# # # Year 2020
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="remote", year=2020)
attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]
joined_bats_W_2020 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_W_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_bats_E_2020 = create_joined_partitions_bats(
    bat_class_list, n_sec=10, attributes=attribute_E_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")

dataset_W_2020 = joined_bats_W_2020
dataset_E_2020 = joined_bats_E_2020
print("Length joined_bats_W_2020: ", len(joined_bats_W_2020[0]))
print("Length joined_bats_E_2020: ", len(joined_bats_E_2020[0]))

dataset_collector = [joined_bats_W_2019, joined_bats_E_2019, joined_bats_W_2020, joined_bats_E_2020]
joined_dataset_full_bats = expand_data_partitions(dataset_collector)
print("Length full dataset: ", len(ConcatDataset(joined_dataset_full_bats)))


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
bat_dataset_FULL = [joined_dataset_full_bats[0], joined_dataset_full_bats[1], joined_dataset_full_bats[3],
                    joined_dataset_full_bats[2]]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from collections import Counter


    def count_species_in_dataset(dataset):
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
        print("Length dataloader: ", len(data_loader))
        bat_species_collector = []
        for batch in data_loader:
            text_label = batch["text_label"]
            bat_species_collector.append(text_label[0])

        print("Species: ", Counter(bat_species_collector).keys())
        print("Species count: ", Counter(bat_species_collector).values())
        return


    count_species_in_dataset(joined_bats_W_2019[0])
    count_species_in_dataset(joined_bats_E_2019[0])
    count_species_in_dataset(joined_bats_W_2020[0])
    count_species_in_dataset(joined_bats_E_2020[0])

