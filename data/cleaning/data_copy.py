import os
import shutil
from data.path_provider import dir_labeled_data_file_server


# # # Generator function to later move recordings in chunks, for more stable data transfer
def chunks(recordings_list, chunk_size):
    """ Yield successive chunks along recordings_list.
        recordings_list should be a list of the file names from the csv.
    """
    for i in range(0, len(recordings_list), chunk_size):
        yield recordings_list[i:i + chunk_size]


# n = chunks(testsignal_fileNames, 10)
# [print(len(next(n))) for x in range(10)]


def copy_chunks(file_names_list, raw_src_path, file_dest_path, chunk_size):
    """ This function is for file chunks of sizes where shutil runs unstable. """
    chunks_list = list(chunks(file_names_list, chunk_size))
    counter = 0
    for chunk in chunks_list:
        for filename in os.listdir(raw_src_path):
            if filename in chunk:
                file_src_path = os.path.join(raw_src_path, filename)
                shutil.copy(file_src_path, file_dest_path)
                counter += 1
    print(counter)


# # # Copy all recordings from W-05m into a single folder
main_path = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19".format(arg=dir_labeled_data_file_server)
subfolders_list = ["7_Juli_2019", "8_August_2019", "9_September_2019", "10_Oktober_2019"]

dest_path = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19/Joint_copy".format(arg=dir_labeled_data_file_server)

for subfolder in subfolders_list:
    source_path = os.path.join(main_path, subfolder)
    print(source_path)
    copy_chunks(os.listdir(source_path), source_path, dest_path, 1000)


#     for file in os.listdir(source_path):
#         file_src_path = os.path.join(source_path, file)
#         shutil.copy(file_src_path, dest_path)
