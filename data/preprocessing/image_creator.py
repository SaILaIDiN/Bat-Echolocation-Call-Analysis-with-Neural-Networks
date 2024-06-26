"""
    Author: Sercan Alipek
    This module takes the raw audio files in .wav format and computes the spectrogram of choice.
    The spectrograms in here are created via librosa library.
    To reduce redundancy and due to the use of few examples per class, the data augmentation is made offline,
     so the synthetic examples are stored for reuse.
"""

import numpy as np
import librosa as li
import librosa.display
import matplotlib.pyplot as plt
import os
import re


def create_noise_spectrogram(dir_noise_spec, dir_noise_wav, dir_noise_txt, n_fft=2048):
    with open(dir_noise_txt, "r") as f:
        file_name_list = f.read().split('\n')

    counter = 0
    sample_rate = 300000  # standard for .wav files (in Hz), automatically handled by sr=None
    for noise_sample in file_name_list:
        if len(noise_sample) > 0:  # last entry in file_name_list may be None -> would create nonexistent path!
            noise_wav = os.path.join(dir_noise_wav, noise_sample)
            if os.path.isfile(noise_wav):
                ts, sr = li.load(noise_wav, sr=None)
                ts = ts[:sr]  # limits the sample points to the first second, for same size spectrogram
                S = np.abs(li.stft(ts, n_fft=n_fft))
                S = S - S.mean(axis=1, keepdims=True)
                S_db = li.amplitude_to_db(S, ref=S.max())
                print(S_db.shape)
                np.save(os.path.join(dir_noise_spec, noise_sample), S_db)
                # plt.figure(figsize=(16, 12))
                # li.display.specshow(S_db, sr=sr)
                # plt.show()
                counter += 1
                if counter > 10000:
                    break


def create_noise_MFCC(n_mfcc, dir_noise_MFCC, dir_noise_wav, dir_noise_txt, n_fft=2048):
    with open(dir_noise_txt, "r") as f:
        file_name_list = f.read().split('\n')

    counter = 0
    sample_rate = 300000  # standard for .wav files (in Hz), automatically handled by sr=None
    for noise_sample in file_name_list:
        if len(noise_sample) > 0:  # last entry in file_name_list may be None -> would create nonexistent path!
            noise_wav = os.path.join(dir_noise_wav, noise_sample)
            if os.path.isfile(noise_wav):
                ts, sr = li.load(noise_wav, sr=None)
                ts = ts[:sr]  # limits the sample points to the first second, for same size MFCC
                mfcc = li.feature.mfcc(y=ts, sr=sr/10, n_mfcc=n_mfcc, n_fft=n_fft)
                print(mfcc.shape)
                np.save(os.path.join(dir_noise_MFCC, noise_sample), mfcc)
                # plt.figure(figsize=(16, 12))
                # li.display.specshow(mfcc, x_axis="time")
                # plt.show()
                counter += 1
                if counter > 10000:
                    break


bat_class_list = ['Bbar', 'Myotis', 'Ppip', 'Ptief', 'Pnat', 'Ppyg', 'Plecotus', 'Nnoc', 'Nyctaloid', 'Phoch',
                  'Pipistrelloid', 'Nlei']  # no order


def create_bat_spectrogram(t1, t2, dir_bats_spec, dir_bats_wav, dir_bats_txt, n_fft=2048):
    for bat_class in bat_class_list:
        bat_call_txt = dir_bats_txt + f"{bat_class}.txt"

        try:
            with open(bat_call_txt, "r") as f:
                file_name_list = f.read().split('\n')
        except:
            continue

        for bat_sample in file_name_list:
            bat_sample = re.sub("_Sec_[0-9][0-9]|_Sec_[0-9]", "", bat_sample)
            try:
                if len(bat_sample) > 0:  # last entry in file_name_list may be None -> would create nonexistent path!
                    bat_wav = os.path.join(dir_bats_wav, bat_sample)
                    if os.path.isfile(bat_wav):
                        ts, sr = li.load(bat_wav, sr=None)  # sr=None uses native sampling rate i.e. 300000Hz here
                        ts = ts[t1*sr:t2*sr]  # limits the sample points to the first second, for same size spectrogram
                        S = np.abs(li.stft(ts, n_fft=n_fft))
                        S = S - S.mean(axis=1, keepdims=True)
                        S_db = li.amplitude_to_db(S, ref=S.max())
                        print(S_db.shape)
                        np.save(os.path.join(dir_bats_spec, bat_sample + "_Sec_" + str(t2)), S_db)
            except ValueError:
                print("Non existent bat-sample in current timestamp: ", bat_sample)


def create_bat_MFCC(t1, t2, n_mfcc, dir_bats_MFCC, dir_bats_wav, dir_bats_txt, n_fft=2048):
    for bat_class in bat_class_list:
        bat_call_txt = dir_bats_txt + f"{bat_class}.txt"

        try:
            with open(bat_call_txt, "r") as f:
                file_name_list = f.read().split('\n')
        except:
            continue

        for bat_sample in file_name_list:
            bat_sample = re.sub("_Sec_[0-9][0-9]|_Sec_[0-9]", "", bat_sample)
            try:
                if len(bat_sample) > 0:  # last entry in file_name_list may be None -> would create nonexistent path!
                    bat_wav = os.path.join(dir_bats_wav, bat_sample)
                    if os.path.isfile(bat_wav):
                        ts, sr = li.load(bat_wav, sr=None)
                        ts = ts[t1*sr:t2*sr]  # limits the sample points to the first second, for same size spectrogram
                        mfcc = li.feature.mfcc(y=ts, sr=sr / 10, n_mfcc=n_mfcc, n_fft=n_fft)
                        print(mfcc.shape)
                        np.save(os.path.join(dir_bats_MFCC, bat_sample + "_Sec_" + str(t2)), mfcc)
            except ValueError:
                print("Non existent bat-sample in current timestamp: ", bat_sample)


if __name__ == "__main__":
    from data.path_provider import provide_paths

    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote="local", year=2019)
    # # # Create spectrogram and MFCCs of noise first
    dir_noise_spec = r"{arg}\noise\spectrograms".format(arg=dir_main)
    dir_noise_MFCC = r"{arg}\noise\MFCCs".format(arg=dir_main)
    dir_noise_wav = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Noise_data".format(arg=dir_raw_data)

    # # # Create spectrograms of bat calls for each type
    # all spectrograms into the same folder for noise-bat-distinction
    dir_bats_spec = r"{arg}\bat_calls\spectrograms".format(arg=dir_main)
    dir_bats_MFCC = r"{arg}\bat_calls\MFCCs_".format(arg=dir_main)
    dir_bats_wav = r"{arg}\ALT\Rufaufnahmen\WMM_NW_W_05_09.07.-12.11.19\Bats_only_data".format(arg=dir_raw_data)
