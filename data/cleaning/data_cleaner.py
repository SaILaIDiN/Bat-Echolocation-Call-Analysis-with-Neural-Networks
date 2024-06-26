import pandas as pd
import glob
import os as os
import shutil
from pathlib import Path


def shift_joint(csv_path, dir_raw_main, dir_joint_data):
    """ move recordings from any location in the original dataset structure to a single collecting folder """
    df = pd.read_csv(csv_path)
    list_filenames = df["FileName"].tolist()

    list_file_paths = []
    for filename in list_filenames:
        file_src_path = glob.glob(r"{attr_1}\*\*\*\{attr_2}".format(attr_1=dir_raw_main, attr_2=filename))
        print("glob return: ", file_src_path)
        if len(file_src_path) > 0:
            list_file_paths.append(file_src_path[0])  # works because each wav-file is unique
            print("HIT!")
            dest_path = Path(os.path.join(dir_joint_data, os.path.split(file_src_path[0])[1]))
            if not dest_path.is_file():
                shutil.move(file_src_path[0], dir_joint_data)
    return len(list_file_paths)


def shift_unlisted(dir_joint_data, dir_unlisted_data, all_recordings_filenames):
    """ move recordings to "Unlisted_data" folder if not listed in csv file """
    counter = 0
    for filename in os.listdir(dir_joint_data):
        if filename not in all_recordings_filenames:
            file_src_path = os.path.join(dir_joint_data, filename)
            shutil.move(file_src_path, dir_unlisted_data)
            counter += 1
    print(counter)


def shift_testsignals(dir_joint_data, dir_unused_data, testsignal_filenames):
    """ shift files with test signals to the "Unused_data" folder """
    for filename in os.listdir(dir_joint_data):
        if filename in testsignal_filenames:
            file_src_path = os.path.join(dir_joint_data, filename)
            shutil.copy(file_src_path, dir_unused_data)


def shift_special(dir_joint_data, dir_unused_data, special_case_filenames):
    """ shift special case bat recordings to "Unused_data" folder """
    for filename in os.listdir(dir_joint_data):
        if filename in special_case_filenames:
            file_src_path = os.path.join(dir_joint_data, filename)
            shutil.move(file_src_path, dir_unused_data)


def shift_noise(dir_joint_data, dir_noise_data, noise_filenames):
    """ shift noise signals to "Noise_data" folder """
    for filename in os.listdir(dir_joint_data):
        if filename in noise_filenames:
            file_src_path = os.path.join(dir_joint_data, filename)
            shutil.move(file_src_path, dir_noise_data)


# # # Generator function to later move recordings in chunks, for more stable data transfer
def chunks(recordings_list, chunk_size):
    """ Yield successive chunks along recordings_list.
        recordings_list should be a list of the file names from the csv.
    """
    for i in range(0, len(recordings_list), chunk_size):
        yield recordings_list[i:i + chunk_size]


# n = chunks(testsignal_fileNames, 10)
# [print(len(next(n))) for x in range(10)]


def shift_chunks(file_names_list, raw_src_path, file_dest_path, chunk_size):
    """ This function is for file chunks of sizes where shutil runs unstable. """
    chunks_list = list(chunks(file_names_list, chunk_size))
    counter = 0
    for chunk in chunks_list:
        for filename in os.listdir(raw_src_path):
            if filename in chunk:
                file_src_path = os.path.join(raw_src_path, filename)
                shutil.move(file_src_path, file_dest_path)
                counter += 1
    print(counter)


# shift_chunks(testsignal_fileNames, dir_joint_data, dir_unused_data, 1000)
# shift_chunks(noise_fileNames, dir_joint_data, dir_noise_data, 1000)
# shift_chunks(special_case_fileNames, dir_joint_data, dir_unused_data, 1000)
# shift_unlabeled()


if __name__ == "__main__":
    from data.path_provider import provide_paths

    # # # Path-Management of wav-files for data from 2019
    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote="local", year=2019)
    attribute_W = ["W_05", "W_33", "W_65", "W_95"]
    attribute_E = ["E_05", "E_33", "E_65", "E_95"]

    # # # Paths for West at 5 meters (2019)
    # dir_joint_data = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Joint_copy".format(arg=dir_raw_data)
    # dir_unlisted_data = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Unlisted_data".format(arg=dir_raw_data)
    # dir_unused_data = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Unused_data".format(arg=dir_raw_data)
    # dir_noise_data = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Noise_data".format(arg=dir_raw_data)
    # csv_path = r"{arg}\evaluation_{attr}m_05_19-11_19.csv".format(attr = attribute_W[0], arg=csv_path_main)

    # # # Paths for East at 5 meters (2019)
    # dir_main_part = r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East_05".format(arg=dir_raw_data)
    # dir_joint_data = os.path.join(dir_main_part, "Joint_copy_E_05")
    # dir_unlisted_data = os.path.join(dir_main_part, "Unlisted_data_E_05")
    # dir_unused_data = os.path.join(dir_main_part, "Unused_data_E_05")
    # dir_noise_data = os.path.join(dir_main_part, "Noise_data_E_05")
    # csv_path = r"{arg}\evaluation_{attr}m_05_19-11_19.csv".format(attr = attribute_E[0], arg=csv_path_main)

    # # # Paths for West at 33, 65 or 95 meters, change attribute (2019)
    # dir_main_part = r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_West".format(arg=dir_raw_data)
    # dir_joint_data = os.path.join(dir_main_part, f"Joint_copy_{attribute_W[3]}")
    # dir_unlisted_data = os.path.join(dir_main_part, f"Unlisted_data_{attribute_W[3]}")
    # dir_unused_data = os.path.join(dir_main_part, f"Unused_data_{attribute_W[3]}")
    # dir_noise_data = os.path.join(dir_main_part, f"Noise_data_{attribute_W[3]}")
    # csv_path = r"{arg}\evaluation_{attr}m_05_19-11_19.csv".format(attr = attribute_W[3], arg=csv_path_main)

    # # # Paths for East at 33, 65 or 95 meters, change attribute (2019)
    # dir_main_part = r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East".format(arg=dir_raw_data)
    # dir_joint_data = os.path.join(dir_main_part, f"Joint_copy_{attribute_E[3]}")
    # dir_unlisted_data = os.path.join(dir_main_part, f"Unlisted_data_{attribute_E[3]}")
    # dir_unused_data = os.path.join(dir_main_part, f"Unused_data_{attribute_E[3]}")
    # dir_noise_data = os.path.join(dir_main_part, f"Noise_data_{attribute_E[3]}")
    # csv_path = r"{arg}\evaluation_{attr}m_05_19-11_19.csv".format(attr = attribute_E[3], arg=csv_path_main)

    # recordings_table = pd.read_csv(csv_path)
    # print(len(recordings_table))

    # # # Remove all recordings that are not listed in the csv file
    # all_recordings_fileNames = list(recordings_table["FileName"])
    # print(len(all_recordings_fileNames))

    # # # Path-Management of wav-files for data from 2020 (different approach)
    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote="local", year=2020)
    attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
    attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]

    # Paths for West
    # dir_raw_main = r"{arg}\BATmode1_WMM_NW".format(arg=dir_raw_data)
    # dir_main_part = os.path.join(dir_raw_main, "Data_For_West")
    # dir_joint_data = os.path.join(dir_main_part, r"Joint_copy_{attr}".format(attr=attribute_W_2020[0]))
    # # dir_unlisted_data = os.path.join(dir_main_part, f"Unlisted_data_{attribute_W_2020[3]}")
    # # dir_unused_data = os.path.join(dir_main_part, f"Unused_data_{attribute_W_2020[3]}")
    # dir_noise_data = os.path.join(dir_main_part, f"Noise_data_{attribute_W_2020[0]}")
    # csv_path = r"{arg}\evaluation_{attr}m_04_20-11_20.csv".format(attr=attribute_W_2020[0], arg=csv_path_main)

    # Paths for East
    # dir_raw_main = r"{arg}\BATmode1_WMM_NE".format(arg=dir_raw_data)
    # dir_main_part = os.path.join(dir_raw_main, "Data_For_East")
    # dir_joint_data = os.path.join(dir_main_part, r"Joint_copy_{attr}".format(attr=attribute_E_2020[0]))
    # # dir_unlisted_data = os.path.join(dir_main_part, f"Unlisted_data_{attribute_E_2020[3]}")
    # # dir_unused_data = os.path.join(dir_main_part, f"Unused_data_{attribute_E_2020[3]}")
    # dir_noise_data = os.path.join(dir_main_part, f"Noise_data_{attribute_E_2020[0]}")
    # csv_path = r"{arg}\evaluation_{attr}m_04_20-11_20.csv".format(attr=attribute_E_2020[0], arg=csv_path_main)

    # Shift all wav-files of a specific measurement height from its original location to a joined folder (based on csv)
    # shift_joint(csv_path, dir_raw_main, dir_joint_data)

    # # Shift all wav-files that are not listed in the specific csv-file to another folder
    # # all_recordings_filenames is only useful if dir_joint_data is filled without a csv-file
    # recordings_table = pd.read_csv(csv_path)
    # all_recordings_filenames = recordings_table["FileName"].tolist()
    # shift_unlisted(dir_joint_data, dir_unlisted_data, all_recordings_filenames)
    #
    # # Shift all recordings of class "Testsignal" to another folder
    # recordings_table = pd.read_csv(csv_path)
    # testsignal_table = recordings_table[recordings_table["Class1"] == 'Testsignal']
    # testsignal_filenames = list(testsignal_table["FileName"])
    # shift_testsignals(dir_joint_data, dir_unused_data, testsignal_filenames)
    #
    # # Shift all special cases of bat recordings apart from the regular ones into another folder
    # # check the types in raw csv first, look for nan values (float) or previously removed separators such as ";"
    # recordings_table = pd.read_csv(csv_path)
    # comment_type_list = list(map(type, recordings_table["Comment"]))
    # comments_case_table = recordings_table[recordings_table["Comment"].notnull()]
    # class2_case_table = recordings_table[recordings_table["Class2"].notnull()]
    # # combine both tables
    # special_case_table = pd.concat([comments_case_table, class2_case_table])
    # special_case_filenames = list(special_case_table["FileName"])
    # shift_special(dir_joint_data, dir_unused_data, special_case_filenames)
    #
    # # Shift noise cases into a different folder
    # noise_table = recordings_table[recordings_table["Class1"] == 'Noise']
    # noise_filenames = list(noise_table["FileName"])
    # shift_noise(dir_joint_data, dir_noise_data, noise_filenames)

