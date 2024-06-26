""" This file takes the cleaned data and conditions the dataset in favor of the task and the model.
    - includes the creation of txt files with bat names separated by species and time position in recording
    - includes creation of images (spectrogram, MFCC,...) for different time positions
    - includes creation of the corresponding labels and specific naming convention for uniqueness
    (NOTE: currently it is only for the bat species! And only for the binary task!)
"""

import os
from data.preprocessing.data_separator import clean_bat_recordings_table, bat_file_name_txt, \
    clean_noise_recordings_table, noise_file_name_txt
from data.preprocessing.image_creator import create_bat_MFCC, create_noise_MFCC, create_bat_spectrogram, \
    create_noise_spectrogram
from data.preprocessing.label_creator import create_bat_labels, create_noise_labels


def create_bat_tables_by_time_stamps(raw_bats_table, highest_sec=1, lookahead=1):
    """ Creates a set of individual lists with all the entries of calls with sufficient length (over given interval) """

    bats_table_collection = []
    for starting_point in range(1, highest_sec + 1):
        raw_bats_table = raw_bats_table.astype({"Length [s]": "float64"})
        bats_table_at_sec = raw_bats_table[raw_bats_table["Length [s]"] >= float(1/lookahead * starting_point)]
        bats_table_collection.append(bats_table_at_sec)
    return bats_table_collection


def create_bat_txts_by_time_stamps(bats_table_collection, dir_data_txts, year, dir_bats_wav, path_spec="", size=100000,
                                   combined_txt=False):
    """ Creates and names individual txt files that list the file names of all bat recordings of sufficient length """

    highest_sec = len(bats_table_collection)
    for starting_point in range(1, highest_sec + 1):
        if combined_txt:
            bat_file_name_txt(bats_table_collection[starting_point - 1], dir_data_txts, year, dir_bats_wav,
                              path_spec=path_spec, txt_postfix="Combined",
                              entry_postfix=f"_Sec_{starting_point}", size=size, combined_txt=True,
                              starting_point=starting_point)
        else:
            bat_file_name_txt(bats_table_collection[starting_point - 1], dir_data_txts, year, dir_bats_wav,
                              path_spec=path_spec, txt_postfix=f"{starting_point}",
                              entry_postfix=f"_Sec_{starting_point}", size=size)


def create_bat_txt_combined_heights(bat_class_list, attr_oh_list, dir_data_txts, year, txt_postfix, txt_name_combined):
    """ Join the combined txt files of the recycled seconds across all heights of the same orientation.
        For each species. Limit "attr_oh_list" to all heights of only one orientation.
    """
    counter = 0
    for bat_type in bat_class_list:
        path_combined_txt = f"{dir_data_txts}/data_{year}/Bat_type_{txt_name_combined}_{bat_type}.txt"
        open(path_combined_txt, 'w').close()
        for path_spec in attr_oh_list:
            path_tmp = f"{dir_data_txts}/data_{year}/{path_spec}/Bat_type_{txt_postfix}_{bat_type}.txt"
            try:
                with open(path_tmp, mode='r') as f:
                    file_name_list = f.read().split('\n')
                with open(path_combined_txt, mode='a') as f:
                    for file_name in file_name_list:
                        f.write(f"{file_name}\n")
            except FileNotFoundError:
                print(f"File {path_tmp} does not exist!")
                counter += 1
                continue
    print("Counter: ", counter)


def create_bat_spectrograms_by_time_stamps(bats_table_collection, dir_bats_spec, dir_bats_wav, dir_bats_txt, n_fft=2048):
    """ Creates and names individual folders with linear spectrograms of all bat recordings of sufficient length """

    highest_sec = len(bats_table_collection)
    counter = 0
    for starting_point in range(1, highest_sec + 1):
        dir_bats_txt_tmp = dir_bats_txt + f"{starting_point}_"
        dir_bats_spec_tmp = dir_bats_spec + f"{starting_point}"
        if not os.path.exists(dir_bats_spec_tmp):
            os.makedirs(dir_bats_spec_tmp)
        try:
            create_bat_spectrogram(starting_point - 1, starting_point, dir_bats_spec_tmp, dir_bats_wav,
                                   dir_bats_txt_tmp, n_fft=n_fft)
        except ValueError:
            counter += 1
            print("Linear spectrogram creation error, no. ", counter)
            break


def create_bat_MFCCs_by_time_stamps(bats_table_collection, dir_bats_MFCC, dir_bats_wav, dir_bats_txt, n_fft=2048):
    """ Creates and names individual folders with MFCCs of all bat recordings of sufficient length """

    highest_sec = len(bats_table_collection)
    counter = 0
    for starting_point in range(1, highest_sec + 1):
        dir_bats_txt_tmp = dir_bats_txt + f"{starting_point}_"
        dir_bats_MFCC_tmp = dir_bats_MFCC + f"{starting_point}"
        if not os.path.exists(dir_bats_MFCC_tmp):
            os.makedirs(dir_bats_MFCC_tmp)
        try:
            create_bat_MFCC(starting_point - 1, starting_point, 20, dir_bats_MFCC_tmp, dir_bats_wav, dir_bats_txt_tmp,
                            n_fft=n_fft)
        except ValueError:
            counter += 1
            print("MFCC creation error, no. ", counter)
            break


def create_bat_labels_by_time_stamps(bats_table_collection, dir_bats_labels, dir_bats_wav, dir_bats_txt):
    """ Creates and names individual folders with labels of all bat recordings of sufficient length """

    highest_sec = len(bats_table_collection)
    for starting_point in range(1, highest_sec + 1):
        dir_bats_txt_tmp = dir_bats_txt + f"{starting_point}_"
        dir_bats_labels_tmp = dir_bats_labels + f"{starting_point}"
        if not os.path.exists(dir_bats_labels_tmp):
            os.makedirs(dir_bats_labels_tmp)
        create_bat_labels(dir_bats_txt_tmp, dir_bats_labels_tmp, dir_bats_wav)


def create_data_directory_by_time_stamps(bats_table_collection, dir_main, dir_bats_txt, dir_noise_txt):
    """ Returns a triple that creates or stores necessary directories to navigate through by time stamps """
    triple_dirs = []
    highest_sec = len(bats_table_collection)
    for starting_point in range(1, highest_sec + 1):
        dir_main_tmp = dir_main
        dir_bats_txt_tmp = dir_bats_txt + f"{starting_point}_"
        dir_noise_txt_tmp = dir_noise_txt
        triple_dirs.append((dir_main_tmp, dir_bats_txt_tmp, dir_noise_txt_tmp))
    return triple_dirs


def create_noise_txts(noise_table, dir_data_txts, year, dir_noise_wav, path_spec="", size=100000):
    """ Does the same as noise_file_name_txt(). This function currently renames it for better reading. """
    noise_file_name_txt(noise_table, dir_data_txts, year, dir_noise_wav, path_spec=path_spec, size=size)


def create_noise_MFCCs_and_Labels(dir_noise_MFCC, dir_noise_wav, dir_noise_labels, dir_noise_txt, n_fft=2048):
    """ It is practical to combine both these steps in one function, because noise recordings occur only with 1.556s
        time length, so that there is no way to recycle them as with bat recordings.
    """
    if not os.path.exists(dir_noise_MFCC):
        os.makedirs(dir_noise_MFCC)
    create_noise_MFCC(20, dir_noise_MFCC, dir_noise_wav, dir_noise_txt, n_fft=n_fft)
    if not os.path.exists(dir_noise_labels):
        os.makedirs(dir_noise_labels)
    create_noise_labels(dir_noise_txt, dir_noise_labels, dir_noise_wav)


def create_noise_spectrograms(dir_noise_spec, dir_noise_wav, dir_noise_txt, n_fft=2048):
    """ Creates the linear spectrograms. Noise recordings occur only with 1.556s
        time length, so that there is no way to recycle them as with bat recordings.
    """
    if not os.path.exists(dir_noise_spec):
        os.makedirs(dir_noise_spec)
    create_noise_spectrogram(dir_noise_spec, dir_noise_wav, dir_noise_txt, n_fft=n_fft)


if __name__ == "__main__":
    import pandas as pd
    from data.path_provider import provide_paths

    # # Year 2019
    year = 2019
    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote="local", year=year)
    attribute_W = ["W_05", "W_33", "W_65", "W_95"]
    attribute_E = ["E_05", "E_33", "E_65", "E_95"]
    attribute_W_and_E = attribute_W + attribute_E

    # # FOR BATS 2019
    dir_bats_wav_list = [
        r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Bats_only_data".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_West\Joint_copy_W_33".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_West\Joint_copy_W_65".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_West\Joint_copy_W_95".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East_05\Joint_copy_E_05".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Joint_copy_E_33".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Joint_copy_E_65".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Joint_copy_E_95".format(arg=dir_raw_data)
    ]

    # # # Year 2020
    # year = 2020
    # system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
    #     local_or_remote="local", year=year)
    # attribute_W = ["W_10", "W_35", "W_65", "W_95"]
    # attribute_E = ["E_10", "E_35", "E_65", "E_95"]
    # attribute_W_and_E = attribute_W + attribute_E
    #
    # # # FOR BATS 2020
    # dir_bats_wav_list = [
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_10".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_35".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_65".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Joint_copy_W_95".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_10".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_35".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_65".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Joint_copy_E_95".format(arg=dir_raw_data)
    # ]

    # # # Other Paths 2019 & 2020
    def get_csv_path(year, attribute_W_and_E, index):
        system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
            local_or_remote="local", year=year)
        if year == 2019:
            csv_path_tmp = r"{arg}\evaluation_{attr}m_05_19-11_19.csv".format(
                attr=attribute_W_and_E[index], arg=csv_path_main)
        elif year == 2020:
            csv_path_tmp = r"{arg}\evaluation_{attr}m_04_20-11_20.csv".format(
                       attr=attribute_W_and_E[index], arg=csv_path_main)
        return csv_path_tmp


    def get_dir_bats_MFCC_and_Labels(year, attribute_W_and_E, index):
        system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
            local_or_remote="local", year=year)
        if year == 2019:
            dir_bats_MFCC_tmp = r"{arg}\bat_calls\{attr}\MFCCs_".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
            dir_bats_labels_tmp = r"{arg}\bat_calls\{attr}\Labels_".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
        elif year == 2020:
            dir_bats_MFCC_tmp = r"{arg}\bat_calls\{attr}\MFCCs_".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
            dir_bats_labels_tmp = r"{arg}\bat_calls\{attr}\Labels_".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
        return dir_bats_MFCC_tmp, dir_bats_labels_tmp


    def get_dir_bats_specs(year, attribute_W_and_E, index, dir_store):
        if year == 2019:
            dir_bats_specs_tmp = r"{arg}\bat_calls\{attr}\LinSpecs_".format(
                       attr=attribute_W_and_E[index], arg=dir_store)
        elif year == 2020:
            dir_bats_specs_tmp = r"{arg}\bat_calls\{attr}\LinSpecs_".format(
                       attr=attribute_W_and_E[index], arg=dir_store)
        return dir_bats_specs_tmp

    for i in range(0, 8):

        csv_path = get_csv_path(year, attribute_W_and_E, i)
        recordings_table = pd.read_csv(csv_path)
        bats_only_table = clean_bat_recordings_table(recordings_table)
        dir_bats_wav = dir_bats_wav_list[i]

        # Create tables
        k = create_bat_tables_by_time_stamps(bats_only_table, 1)
        [print(len(k_sub)) for k_sub in k]
        # Create txt files
        create_bat_txts_by_time_stamps(k, dir_data_txts, year, dir_bats_wav,
                                       path_spec= attribute_W_and_E[i], combined_txt=True)

        # Create MFCCs
        dir_bats_MFCC = get_dir_bats_MFCC_and_Labels(year, attribute_W_and_E, i)[0]
        dir_bats_txt = r"{arg}\data_{attr_2}\{attr}\Bat_type_".format(
            attr=attribute_W_and_E[i], attr_2=year, arg=dir_data_txts)
        create_bat_MFCCs_by_time_stamps(k, dir_bats_MFCC, dir_bats_wav, dir_bats_txt)

        # Create labels
        dir_bats_labels = get_dir_bats_MFCC_and_Labels(year, attribute_W_and_E, i)[1]
        create_bat_labels_by_time_stamps(k, dir_bats_labels, dir_bats_wav, dir_bats_txt)

        # Create LinSpec
        dir_bats_LinSpec = get_dir_bats_specs(year, attribute_W_and_E, i)
        dir_bats_txt = r"{arg}\data_{attr_2}\{attr}\Bat_type_".format(
            attr=attribute_W_and_E[i], attr_2=year, arg=dir_data_txts)
        create_bat_spectrograms_by_time_stamps(k, dir_bats_LinSpec, dir_bats_wav, dir_bats_txt)


    # # FOR NOISE 2019
    dir_noise_wav_list = [
        r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Noise_data".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_West\Noise_data_W_33".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_West\Noise_data_W_65".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_West\Noise_data_W_95".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East_05\Noise_data_E_05".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Noise_data_E_33".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Noise_data_E_65".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Noise_data_E_95".format(arg=dir_raw_data)
                         ]

    # # # FOR NOISE 2020
    # dir_noise_wav_list = [
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Noise_data_W_10".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Noise_data_W_35".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Noise_data_W_65".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NW\Data_For_West\Noise_data_W_95".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Noise_data_E_10".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Noise_data_E_35".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Noise_data_E_65".format(arg=dir_raw_data),
    #     r"{arg}\BATmode1_WMM_NE\Data_For_East\Noise_data_E_95".format(arg=dir_raw_data)
    #                      ]


    def get_dir_noise_MFCC_and_Labels(year, attribute_W_and_E, index):
        system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
            local_or_remote="local", year=year)
        if year == 2019:
            dir_noise_MFCC_tmp = r"{arg}\noise\{attr}\MFCCs".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
            dir_noise_labels_tmp = r"{arg}\noise\{attr}\Labels".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
        elif year == 2020:
            dir_noise_MFCC_tmp = r"{arg}\noise\{attr}\MFCCs".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
            dir_noise_labels_tmp = r"{arg}\noise\{attr}\Labels".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
        return dir_noise_MFCC_tmp, dir_noise_labels_tmp


    def get_dir_noise_spectrogram(year, attribute_W_and_E, index):
        system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
            local_or_remote="local", year=year)
        if year == 2019:
            dir_noise_spec_tmp = r"{arg}\noise\{attr}\LinSpecs".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
        elif year == 2020:
            dir_noise_spec_tmp = r"{arg}\noise\{attr}\LinSpecs".format(
                       attr=attribute_W_and_E[index], arg=dir_main)
        return dir_noise_spec_tmp


    for i in range(0, 8):
        size = len(os.listdir(dir_noise_wav_list[i])) if len(os.listdir(dir_noise_wav_list[i])) <= 10000 else 10000
        csv_path = get_csv_path(year, attribute_W_and_E, i)
        recordings_table = pd.read_csv(csv_path)
        dir_noise_wav = dir_noise_wav_list[i]

        # Create table
        noise_table = clean_noise_recordings_table(recordings_table)

        # Create txt file
        create_noise_txts(noise_table, dir_data_txts, year, dir_noise_wav,
                          path_spec=attribute_W_and_E[i], size=size)

        # Create MFCCs and labels
        dir_noise_MFCC = get_dir_noise_MFCC_and_Labels(year, attribute_W_and_E, i)[0]
        dir_noise_labels = get_dir_noise_MFCC_and_Labels(year, attribute_W_and_E, i)[1]
        dir_noise_txt = r"{arg}\data_{attr_2}\{attr}\Noise_file_names{size}.txt".format(
            attr=attribute_W_and_E[i], attr_2=year, size=size, arg=dir_data_txts)
        create_noise_MFCCs_and_Labels(dir_noise_MFCC, dir_noise_wav, dir_noise_labels, dir_noise_txt)

        # Create LinSpec
        dir_noise_spec = get_dir_noise_spectrogram(year, attribute_W_and_E, i)
        dir_noise_txt = r"{arg}\data_{attr_2}\{attr}\Noise_file_names{size}.txt".format(
            attr=attribute_W_and_E[i], attr_2=year, size=size, arg=dir_data_txts)
        create_noise_spectrogram(dir_noise_spec, dir_noise_wav, dir_noise_txt)
