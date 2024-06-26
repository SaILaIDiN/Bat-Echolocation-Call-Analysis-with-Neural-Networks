import pandas as pd
import os
import collections
from data.path_provider import provide_paths


system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
    local_or_remote="local", year=2020)


def path_lookup(index):
    """ Simple lookup table to assign correct path by index. """
    path_to_check_arr = []
    path_to_check_arr.append(r"{arg}\evaluation_E_10m_05_20-11_20.csv".format(arg=csv_path_main))
    path_to_check_arr.append(r"{arg}\evaluation_E_35m_05_20-11_20.csv".format(arg=csv_path_main))
    path_to_check_arr.append(r"{arg}\evaluation_E_65m_05_20-11_20.csv".format(arg=csv_path_main))
    path_to_check_arr.append(r"{arg}\evaluation_E_95m_05_20-11_20.csv".format(arg=csv_path_main))
    path_to_check_arr.append(r"{arg}\evaluation_W_10m_04_20-11_20.csv".format(arg=csv_path_main))
    path_to_check_arr.append(r"{arg}\evaluation_W_35m_04_20-11_20.csv".format(arg=csv_path_main))
    path_to_check_arr.append(r"{arg}\evaluation_W_65m_04_20-11_20.csv".format(arg=csv_path_main))
    path_to_check_arr.append(r"{arg}\evaluation_W_95m_04_20-11_20.csv".format(arg=csv_path_main))

    dir_to_compare_arr = []
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_10".format(arg=dir_raw_data))
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_35".format(arg=dir_raw_data))
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_65".format(arg=dir_raw_data))
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_95".format(arg=dir_raw_data))
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_10".format(arg=dir_raw_data))
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_35".format(arg=dir_raw_data))
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_65".format(arg=dir_raw_data))
    dir_to_compare_arr.append(r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_95".format(arg=dir_raw_data))

    path_to_check = path_to_check_arr[index]
    dir_to_compare = dir_to_compare_arr[index]
    return path_to_check, dir_to_compare


def dist_loop():
    """
        Parses through all subsets of the data by height and orientation and checks if the listed filenames in the csv
        are provided as wav-files in the dataset. It also computes a distribution of class counts that are not provided.
    """

    counter_container = collections.Counter()
    for i in range(0, 8):
        df = pd.read_csv(path_lookup(i)[0])
        unique_wav_set = df.FileName.unique()
        # print(unique_wav_set)
        # print(len(unique_wav_set))

        list_dir = os.listdir(path_lookup(i)[1])
        # print(list_dir)
        # print(len(list_dir))

        # Check histogram of classes in csv-file
        df_class1 = df.Class1
        # print(df_class1)

        dist_csv = collections.Counter(df_class1)
        # print("Dist_csv: ", dist_csv)
        counter_container += dist_csv

        # # Check the missing wav-files in the dir
        set_csv = set(df.FileName)
        # print(len(set_csv))
        set_dir = set(list_dir)
        # print(len(set_dir))

        set_diff = set_csv.difference(set_dir)
        # print(len(set_diff))
        list_diff = list(set_diff)
        df_missed = [df[df["FileName"] == list_diff[i]].Class1 for i in range(len(list_diff))]
        # df_missed = [df[df["FileName"] == list_diff[i]].Class1 for i in range(0, 2)]

        # print("DF_missed: ", df_missed)
        df_missed_class1 = [df_missed[i].values[0] for i in range(len(df_missed))]
        # print(df_missed_class1)
        dist_missed = collections.Counter(df_missed_class1)
        # print("Dist_missed: ", dist_missed)
    print("COUNTER-FULL: ", counter_container)


if __name__ == "__main__":
    dist_loop()
