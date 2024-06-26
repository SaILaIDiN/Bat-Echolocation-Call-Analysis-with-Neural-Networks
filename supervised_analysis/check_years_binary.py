""" This script is for comparison of the data from 2019 with the data from 2020. (Bat/Noise-Distinction)
    Different cases are investigated:
     - the binary classification between Northeast 2019 and Northwest 2019 (all heights combined)
     - the binary classification between Northeast 2020 and Northwest 2020 (all heights combined)
     - the binary classification between Northeast 2019 and Northeast 2020 (all heights combined)
     - the binary classification between Northwest 2019 and Northwest 2020 (all heights combined)

    Excluded classes are Nlei, Phoch, Ptief, Pipistrelloid, Nnoc.
    These are either excluded because of too small number of examples in general or because the number of
    examples in the superclass like Phoch, Ptief, Pipistrelloid is very small compared to the number of
    examples of individual species that are also common in that superclass.

    Included classes are:
     - the two superclasses (Gattung) Myotis and Plecotus since no single species distinction was possible
       for the human experts,
     - the individual species Bbar, Pnat, Ppip, Ppyg since their number of examples is enough and their
       representative joined classes (Phoch, Ptief, Pipistrelloid) cannot offer sufficient examples
       and we want to maximize the discriminative ability
     - the superclass (Artengruppe) Nyctaloid since it has much more examples than Nnoc
       which increases the distribution of useful features in the training data

    Optional:
    Data Recycling can be performed to naturally increase the number of examples per class.
    Especially useful for rare classes.

    PERFORM ALL SPLITS SPECIES BY SPECIES AND CONCAT SUBSETS
    Procedure - Test 1:
    - Split: NE19 (60:20:20), NW19 (60:20:20)
    - Train: NE19-train, Val1: NE19-val, Val2: NW19-val
    - Test: NE-19-train on NW-19-train+test  (not using val to have direct comparison)
    - repeat the same thing the other way around
    Procedure - Test 2:
    - Repeat Test 1 for NE20 and NW20
    Procedure - Test 3:
    - Split: NE19 (60:20:20), NE20 (60:20:20)
    - Train: NE19-train, Val1: NE19-val, Val2: NE20-val
    - Test: NE-19-train on NE-20-train+test  (not using val to have direct comparison)
    - repeat the same thing the other way around
    Procedure - Test 4:
    - Repeat Test 3 for NW19 and NW20

"""
from torch.utils.data import ConcatDataset
from data.preprocessing.recycle_data import create_joined_partitions_bats, create_joined_partitions_noise, \
    expand_data_partitions
from data.path_provider import provide_paths

# # # Dataset
# Define the species to investigate
bat_class_list = ['Plecotus', 'Myotis', 'Bbar', 'Pnat', 'Ppip', 'Ppyg', 'Nyctaloid']
# Create combined dataset per species across all heights of a single orientation
# Year 2019
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="remote", year=2019)
spec_type = "MFCC"
attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]
joined_bats_W_2019 = create_joined_partitions_bats(bat_class_list, n_sec=5, attributes=attribute_W_2019,
                                                   dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
                                                   train_size=0.6, val_size=0.2, train_sub_size=0.05,
                                                   pre_balanced=False, toy_factor=None, n_per_species=None,
                                                   spec_type=spec_type)
joined_bats_E_2019 = create_joined_partitions_bats(bat_class_list, n_sec=5, attributes=attribute_E_2019,
                                                   dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
                                                   train_size=0.6, val_size=0.2, train_sub_size=0.05,
                                                   pre_balanced=False, toy_factor=None, n_per_species=None,
                                                   spec_type=spec_type)
joined_noise_W_2019 = create_joined_partitions_noise(csv_path_main=csv_path_main, size=None,
                                                     attributes=attribute_W_2019, dir_data_txts=dir_data_txts,
                                                     year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                     train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_2019 = create_joined_partitions_noise(csv_path_main=csv_path_main, size=None,
                                                     attributes=attribute_E_2019, dir_data_txts=dir_data_txts,
                                                     year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                     train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
dataset_E_2019 = expand_data_partitions([joined_bats_E_2019, joined_noise_E_2019])
dataset_W_2019 = expand_data_partitions([joined_bats_W_2019, joined_noise_W_2019])


# # # Year 2020
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="remote", year=2020)
spec_type = "MFCC"
attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]
joined_bats_W_2020 = create_joined_partitions_bats(bat_class_list, n_sec=5, attributes=attribute_W_2020,
                                                   dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
                                                   train_size=0.6, val_size=0.2, train_sub_size=0.05,
                                                   pre_balanced=False, toy_factor=None, n_per_species=None,
                                                   spec_type=spec_type)
joined_bats_E_2020 = create_joined_partitions_bats(bat_class_list, n_sec=5, attributes=attribute_E_2020,
                                                   dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
                                                   train_size=0.6, val_size=0.2, train_sub_size=0.05,
                                                   pre_balanced=False, toy_factor=None, n_per_species=None,
                                                   spec_type=spec_type)
joined_noise_W_2020 = create_joined_partitions_noise(csv_path_main=csv_path_main, size=None,
                                                     attributes=attribute_W_2020, dir_data_txts=dir_data_txts,
                                                     year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                     train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_2020 = create_joined_partitions_noise(csv_path_main=csv_path_main, size=None,
                                                     attributes=attribute_E_2020, dir_data_txts=dir_data_txts,
                                                     year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                     train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
dataset_E_2020 = expand_data_partitions([joined_bats_E_2020, joined_noise_E_2020])
dataset_W_2020 = expand_data_partitions([joined_bats_W_2020, joined_noise_W_2020])

# # # Create the composition of partitions for each test (train_own, val_own, val_foreign, train_foreign+test_foreign)
# 01_Test_1_NE19_train_NW19_test
dataset_01 = [dataset_E_2019[0], dataset_E_2019[1], dataset_W_2019[1],
              ConcatDataset([dataset_W_2019[0], dataset_W_2019[3]])]
# 02_Test_1_NW19_train_NE19_test
dataset_02 = [dataset_W_2019[0], dataset_W_2019[1], dataset_E_2019[1],
              ConcatDataset([dataset_E_2019[0], dataset_E_2019[3]])]
# 03_Test_2_NE20_train_NW20_test
dataset_03 = [dataset_E_2020[0], dataset_E_2020[1], dataset_W_2020[1],
              ConcatDataset([dataset_W_2020[0], dataset_W_2020[3]])]
# 04_Test_2_NW20_train_NE20_test
dataset_04 = [dataset_W_2020[0], dataset_W_2020[1], dataset_E_2020[1],
              ConcatDataset([dataset_E_2020[0], dataset_E_2020[3]])]
# 05_Test_3_NE19_train_NE20_test
dataset_05 = [dataset_E_2019[0], dataset_E_2019[1], dataset_E_2020[1],
              ConcatDataset([dataset_E_2020[0], dataset_E_2020[3]])]
# 06_Test_3_NE20_train_NE19_test
dataset_06 = [dataset_E_2020[0], dataset_E_2020[1], dataset_E_2019[1],
              ConcatDataset([dataset_E_2019[0], dataset_E_2019[3]])]
# 07_Test_4_NW19_train_NW20_test
dataset_07 = [dataset_W_2019[0], dataset_W_2019[1], dataset_W_2020[1],
              ConcatDataset([dataset_W_2020[0], dataset_W_2020[3]])]
# 08_Test_4_NW20_train_NW19_test
dataset_08 = [dataset_W_2020[0], dataset_W_2020[1], dataset_W_2019[1],
              ConcatDataset([dataset_W_2019[0], dataset_W_2019[3]])]


if __name__ == "__main__":
    from torch.utils.data import DataLoader, ConcatDataset


    def check_length_of_batch_samples(dataset, batch_size):
        """ remove MFCCs in bat-noise-task (probably noise) that are not [20, 586];
            error triggers for batch_size > 1 """

        dataset = ConcatDataset([dataset[0], dataset[1], dataset[3]])
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)
        for i, batch in enumerate(data_loader, 0):
            images = batch["image"].unsqueeze(1)  # Shape [N, H, W] -> [N, C, H, W]
            bin_labels = batch["label"]
            species_labels = batch["text_label"]
            image_path = batch["image_path"]

            if images.shape[3] != 586:
                print("Wrong size: ", images.shape)
                print("species_label: ", species_labels)
                print("image_path: ", image_path)

    print("dataset_W_2019")
    check_length_of_batch_samples(dataset_W_2019, batch_size=1)
    print("dataset_E_2019")
    check_length_of_batch_samples(dataset_E_2019, batch_size=1)
    print("dataset_W_2020")
    check_length_of_batch_samples(dataset_W_2020, batch_size=1)
    print("dataset_E_2020")
    check_length_of_batch_samples(dataset_E_2020, batch_size=1)
