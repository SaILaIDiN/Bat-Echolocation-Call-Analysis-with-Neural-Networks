""" For noise-analysis: Train the convolutional AE with the joined noise but evaluate separately for each height due
    to required custom label definitions. """

from torch.utils.data import ConcatDataset
from data.preprocessing.recycle_data import create_joined_partitions_bats, create_joined_partitions_noise
from data.path_provider import provide_paths

# # # Year 2019 Noise
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="local", year=2019)
spec_type = "LinSpec"
attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]

joined_noise_W_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=190,
                                                         attributes=attribute_W_2019, dir_data_txts=dir_data_txts,
                                                         year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                         train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_05_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=190,
                                                            attributes=["W_05"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_33_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=190,
                                                            attributes=["W_33"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_65_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=190,
                                                            attributes=["W_65"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_95_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=190,
                                                            attributes=["W_95"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)

joined_noise_E_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                         attributes=attribute_E_2019, dir_data_txts=dir_data_txts,
                                                         year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                         train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_05_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_05"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_33_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_33"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_65_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_65"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_95_2019_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_95"], dir_data_txts=dir_data_txts,
                                                            year=2019, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)


# # # Year 2020 Noise
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="local", year=2020)
spec_type = "LinSpec"
attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]

joined_noise_W_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                         attributes=attribute_W_2020, dir_data_txts=dir_data_txts,
                                                         year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                         train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_10_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["W_10"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_35_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["W_35"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_65_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["W_65"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_W_95_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["W_95"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)

joined_noise_E_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                         attributes=attribute_E_2020, dir_data_txts=dir_data_txts,
                                                         year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                         train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_10_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_10"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_35_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_35"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_65_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_65"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)
joined_noise_E_95_2020_lin = create_joined_partitions_noise(csv_path_main=csv_path_main, size=250,
                                                            attributes=["E_95"], dir_data_txts=dir_data_txts,
                                                            year=2020, dir_main=dir_main, train_size=0.6, val_size=0.2,
                                                            train_sub_size=0.05, toy_factor=None, spec_type=spec_type)


# Create precedent/preliminary Dataset of the already complete superclasses 'Nyctaloid' and 'Myotis'
# All Pipistrellus classes will be merged into one class (species are randomly but equally partitioned in separation)
# In the training and evaluation phase, the pipistrellus species will be treated as a single genus/superclass

# # # Year 2019 Bats
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="local", year=2019)
spec_type = "LinSpec"
attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]

# # # Set Nyctaloid
bat_class_list = ['Nyctaloid']
joined_nyctaloid_W_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.95, val_size=0.025, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="West")
joined_nyctaloid_E_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.95, val_size=0.025, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="East")

# # # Set Myotis
bat_class_list = ['Myotis']
joined_myotis_W_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.95, val_size=0.025, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="West")
joined_myotis_E_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.95, val_size=0.025, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="East")

# # # Set Plecotus
bat_class_list = ['Plecotus']
joined_plecotus_W_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.8, val_size=0.1, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_plecotus_E_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.8, val_size=0.1, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")

# # # Set Psuper
bat_class_list = ['Ppip', 'Ptief', 'Phoch', 'Pnat', 'Ppyg', 'Pipistrelloid']
joined_pip_W_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.0315, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_pip_E_2019_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2019, dir_data_txts=dir_data_txts, year=2019, dir_main=dir_main,
    train_size=0.0375, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")

# # # Year 2020 Bats
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, _ = provide_paths(
    local_or_remote="local", year=2020)
spec_type = "LinSpec"
attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]

# # # Set Nyctaloid
bat_class_list = ['Nyctaloid']
joined_nyctaloid_W_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="West")
joined_nyctaloid_E_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="East")

bat_class_list = ['Myotis']
joined_myotis_W_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="West")
joined_myotis_E_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.6, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=250, data_limit_auto=True, orientation="East")

bat_class_list = ['Plecotus']
joined_plecotus_W_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.8, val_size=0.1, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_plecotus_E_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.8, val_size=0.1, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")

bat_class_list = ['Ppip', 'Ptief', 'Phoch', 'Pnat', 'Ppyg', 'Pipistrelloid']
joined_pip_W_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_W_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.025, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="West")
joined_pip_E_2020_lin = create_joined_partitions_bats(
    bat_class_list, n_sec=1, attributes=attribute_E_2020, dir_data_txts=dir_data_txts, year=2020, dir_main=dir_main,
    train_size=0.035, val_size=0.2, train_sub_size=0.05, pre_balanced=False, toy_factor=None, spec_type=spec_type,
    n_per_species=None, data_limit_auto=False, orientation="East")


# # # Sets combined across years and orientations (2019 & 2020)
dataset_joined_nyctaloid = ConcatDataset([joined_nyctaloid_W_2019_lin[1], joined_nyctaloid_E_2019_lin[1],
                                          joined_nyctaloid_W_2020_lin[1], joined_nyctaloid_E_2020_lin[1]])
dataset_joined_myotis = ConcatDataset([joined_myotis_W_2019_lin[1], joined_myotis_E_2019_lin[1],
                                       joined_myotis_W_2020_lin[1], joined_myotis_E_2020_lin[1]])
dataset_joined_plecotus = ConcatDataset([joined_plecotus_W_2019_lin[1], joined_plecotus_E_2019_lin[1],
                                         joined_plecotus_W_2020_lin[1], joined_plecotus_E_2020_lin[1]])
dataset_joined_psuper = ConcatDataset([joined_pip_W_2019_lin[1], joined_pip_E_2019_lin[1],
                                       joined_pip_W_2020_lin[1], joined_pip_E_2020_lin[1]])
dataset_joined_noise = ConcatDataset([joined_noise_W_2019_lin[1], joined_noise_E_2019_lin[1],
                                      joined_noise_W_2020_lin[1], joined_noise_E_2020_lin[1]])

# # # Check distributions of some classes
print(len(joined_nyctaloid_W_2019_lin[0]), len(joined_nyctaloid_E_2019_lin[0]), len(joined_nyctaloid_W_2020_lin[0]),
      len(joined_nyctaloid_E_2020_lin[0]))
print(len(joined_myotis_W_2019_lin[0]), len(joined_myotis_E_2019_lin[0]), len(joined_myotis_W_2020_lin[0]),
      len(joined_myotis_E_2020_lin[0]))
print(len(joined_plecotus_W_2019_lin[0]), len(joined_plecotus_E_2019_lin[0]), len(joined_plecotus_W_2020_lin[0]),
      len(joined_plecotus_E_2020_lin[0]))
print(len(joined_pip_W_2019_lin[0]), len(joined_pip_E_2019_lin[0]), len(joined_pip_W_2020_lin[0]),
      len(joined_pip_E_2020_lin[0]))

# 3490 1768 9261 992
# 1049 807 1171 253
# 7 6 81 21
# 4778 3999 6107 4515
