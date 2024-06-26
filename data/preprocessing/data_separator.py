""" Create separate lists of file names for each bat species in the dataset/csv file.
    1. Get all unique classes from the csv file (remove duplicates)
    2. Remove all classes that are not a bat species (if present)
    3. Parse through the csv for each bat class and store the file names in a separate list
    Use these separate lists to parse through the joint bats-only recordings when creating images
    NOTE: using the csv file for bats only is recommended. Skip step 2. then.
"""
import os


def clean_bat_recordings_table(table):
    """ Removes noise entries and improper bat entries from the table """

    # Remove noise
    no_noise_table = table[table["Class1"] != "Noise"]
    # Remove test signals
    no_test_signal_table = no_noise_table[no_noise_table["Class1"] != "Testsignal"]
    # Remove special cases
    no_comments_table = no_test_signal_table[no_test_signal_table["Comment"].isnull()]
    no_class2_table = no_comments_table[no_comments_table["Class2"].isnull()]
    # print(len(no_class2_table))
    clean_bats_table = no_class2_table
    return clean_bats_table


def create_bats_only_csv(table, csv_name):
    table.to_csv(csv_name, float_format='%.3f')


def bat_file_names_dict(bats_table):
    """ returns a dictionary with all bat species as keys and a list of all corresponding file names """

    all_class1_entries = list(bats_table["Class1"])
    unique_classes = set(all_class1_entries)
    # print(unique_classes)

    bat_type_dict = dict()
    for bat_type in unique_classes:
        bat_type_table = bats_table[bats_table["Class1"] == bat_type]
        bat_type_file_names = list(bat_type_table["FileName"])
        bat_type_dict[bat_type] = bat_type_file_names
    return bat_type_dict


def bat_file_name_txt(bats_table, dir_data_txts, year, dir_bats_wav, path_spec="", txt_postfix="", entry_postfix="",
                      size=100000, combined_txt=False, starting_point=1):
    """ Create txt file for each bat type """

    bat_dict = bat_file_names_dict(bats_table)
    for bat_type in list(bat_dict.keys()):
        bat_type_tmp = bat_dict[bat_type]

        if not os.path.exists(dir_data_txts):
            os.makedirs(dir_data_txts)

        # path_spec is the folder of specific measurement direction and height for example "E_05" or "W_33"
        path_tmp = f"{dir_data_txts}/data_{year}/{path_spec}/Bat_type_{txt_postfix}_{bat_type}.txt"

        if combined_txt and starting_point == 1:
            mode = "w"
        elif combined_txt and starting_point > 1:
            mode = "a"
        else:
            mode = "w"  # cases without combined txt file

        with open(path_tmp, mode=f"{mode}") as wf:
            counter = 0
            for bat_call in bat_type_tmp:
                if counter <= size:
                    bat_wav = os.path.join(dir_bats_wav, bat_call)
                    if os.path.isfile(bat_wav):
                        wf.write(f"{bat_call}{entry_postfix}\n")
                        counter += 1
                else:
                    continue


def clean_noise_recordings_table(table):
    """ Removes all entries besides noise entries from the table """

    noise_table = table[table["Class1"] == 'Noise']
    return noise_table


def noise_file_name_txt(noise_table, dir_data_txts, year, dir_noise_wl, path_spec="", size=None):
    """ Create txt file for the file names of class noise """

    noise_filenames = list(noise_table["FileName"])

    # Include a logic to parse through the table and wav files to check the existent files to get the number
    counter_size = 0
    if size is None:
        for noise in noise_filenames:
            noise_wav = os.path.join(dir_noise_wl, noise)
            print(noise_wav)
            noise_label = os.path.join(dir_noise_wl, noise) + ".npy"
            print(noise_label)

            if os.path.isfile(noise_wav) or os.path.isfile(noise_label):  # noise_label[:-4]?
                counter_size += 1
                print("counter updated: ", counter_size)
            if counter_size >= 10000:
                break
        size = counter_size

    counter_check = 0
    with open(r"{dir_data_txts}\data_{year}\{path_spec}\Noise_file_names{size}.txt".format(
            dir_data_txts=dir_data_txts, year=year, path_spec=path_spec, size=size), mode="w") as wf:
        for noise in noise_filenames:
            noise_wav = os.path.join(dir_noise_wl, noise)
            noise_label = os.path.join(dir_noise_wl, noise) + ".npy"
            if os.path.isfile(noise_wav) or os.path.isfile(noise_label):
                wf.write(f"{noise}\n")
                counter_check += 1
            if size:
                if counter_check >= size:
                    break

    return size
