""" This file is to correct the filename format of the East data.
    It makes the format match the one of the West data and to make East data unique along heights.
    The correction has to happen directly on the wav files and on the their csv file.
    NOTE: we do not correct the filenames of "Unlisted_data" and "Unused_data" folders because they are not used.
"""
import os
import pandas as pd
from data.path_provider import provide_paths

# # # FIX for wav-files
# Change the substring "05_95" in the filename by the correct single height value i.e. 05, 33, 65 or 95
# Technically need to do it for at least "Joint_copy_*" folder containing the bats and the "Noise_data_*" folder!
# But make it consistent with csv-file filename updates!
system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
    local_or_remote="local", year=2019)
dir_wav = [
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East_05\Joint_copy_E_05".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Joint_copy_E_33".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Joint_copy_E_65".format(arg=dir_raw_data),
        r"{arg}\NEU_20201126\BatMode-System\Rufaufnahmen\Data_For_East\Joint_copy_E_95".format(arg=dir_raw_data)
          ]
attribute_E = ["_E_05", "_E_33", "_E_65", "_E_95"]

for i in range(len(attribute_E)):
    for filename in os.listdir(dir_wav[i]):
        os.renames(os.path.join(dir_wav[i], filename), os.path.join(dir_wav[i], filename.replace("_E_05-95",
                                                                                                 attribute_E[i])))

# # # FIX for csv-files
attribute_E = ["_E_05", "_E_33", "_E_65", "_E_95"]

# for i in range(1, 4):
for i in range(len(attribute_E)):
    csv_path = r"{arg}\evaluation{attr}m_05_19-11_19.csv".format(attr=attribute_E[i], arg=csv_path_main)
    # Read in the csv-file to a dataframe, update all filenames and save it again under same name
    recordings_table = pd.read_csv(csv_path)
    all_filenames_list = list(recordings_table["FileName"])
    counter = 0
    for filename in all_filenames_list:
        counter += 1
        print("ACTION!", counter)
        recordings_table.replace({"FileName": filename}, filename.replace("_E_05-95", attribute_E[i]), inplace=True)

    recordings_table.to_csv(csv_path, index=False, float_format='%.3f')
