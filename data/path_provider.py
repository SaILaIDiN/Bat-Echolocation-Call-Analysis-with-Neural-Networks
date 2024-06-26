

def provide_paths(local_or_remote=None, year=None):
    if local_or_remote == "local":
        file_mp = open(r"local\path\to\project\argparse_configs\mp"
                       r"_local_{arg}.txt".format(arg=year), "r")
    elif local_or_remote == "remote":
        file_mp = open(r"remote\path\to\project\argparse_configs\mp"
                       r"_remote_{arg}.txt".format(arg=year), "r")
    else:
        print("path location must be defined correctly. Either \"local\" or \"remote\".")
    system_mode = r"{arg}".format(arg=file_mp.readline()[:-1])  # [:-1] removes added "\n"
    dir_main = r"{arg}".format(arg=file_mp.readline()[:-1])
    dir_bats_txt = r"{arg}".format(arg=file_mp.readline()[:-1])
    dir_noise_txt = r"{arg}".format(arg=file_mp.readline()[:-1])
    dir_data_txts = r"{arg}".format(arg=file_mp.readline()[:-1])
    csv_path_main = r"{arg}".format(arg=file_mp.readline()[:-1])
    dir_raw_data = r"{arg}".format(arg=file_mp.readline())
    file_mp.close()
    return system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data


# # # Definition of individual paths collected in here to improve readability of other modules
dir_labeled_data_file_server = r"folder\to\labeled\data"
dir_results = r"your\local\path\to\git\repository\results"
dir_results_remote = r"your\remote\path\to\git\repository\results"
dir_desktop = r"your\path\to\desktop"
