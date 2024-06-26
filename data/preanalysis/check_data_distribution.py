import pandas as pd
from data.preprocessing.data_separator import clean_bat_recordings_table
from data.preprocessing.data_conditioning import create_bat_tables_by_time_stamps, create_bat_txts_by_time_stamps
from data.path_provider import provide_paths


attribute_W_2019 = ["W_05", "W_33", "W_65", "W_95"]
attribute_E_2019 = ["E_05", "E_33", "E_65", "E_95"]
attribute_W_and_E_2019 = attribute_W_2019 + attribute_E_2019

attribute_W_2020 = ["W_10", "W_35", "W_65", "W_95"]
attribute_E_2020 = ["E_10", "E_35", "E_65", "E_95"]
attribute_W_and_E_2020 = attribute_W_2020 + attribute_E_2020


def create_overview_of_species_per_timestamp(attributes, year, output_global_postfix="Undefined"):
    lookahead = 0.8
    time_stamp_collection_sum_all_attributes = []
    for index, attr in enumerate(attributes):

        def get_csv_path(year, attribute_W_and_E, index):
            """ adapt csv path based on audio data path """
            _, _, _, _, _, csv_path_main, _ = provide_paths(local_or_remote="local", year=year)
            if year == 2019:
                csv_path_tmp = r"{arg}\evaluation_{attr}m_05_19-11_19.csv"\
                    .format(attr=attribute_W_and_E[index], arg=csv_path_main)
            elif year == 2020:
                csv_path_tmp = r"{arg}\evaluation_{attr}m_04_20-11_20.csv"\
                    .format(attr=attribute_W_and_E[index], arg=csv_path_main)
            return csv_path_tmp

        csv_path = get_csv_path(year, attributes, index)
        recordings_table = pd.read_csv(csv_path)
        bats_only_table = clean_bat_recordings_table(recordings_table)

        bats_table_collection = create_bat_tables_by_time_stamps(bats_only_table, 10, lookahead=lookahead)
        # create_bat_txts_by_time_stamps(bats_table_collection, year=year, path_spec=attr)

        # compute a class distribution for each timestamp
        # [print(f"MIN-SECOND: {i+1}\n", time_stamp["Class1"].value_counts()) for i, time_stamp in
        #  enumerate(bats_table_collection)]
        time_stamp_collection = []
        [time_stamp_collection.append([f"MIN-SECOND: {i+1}", time_stamp["Class1"].value_counts()]) for i, time_stamp in
         enumerate(bats_table_collection)]
        print(time_stamp_collection)

        # compute a summarized class distribution over each timestamp
        time_stamp_collection_tmp = []
        [time_stamp_collection_tmp.append(time_stamp["Class1"].value_counts()) for i, time_stamp in
         enumerate(bats_table_collection)]
        time_stamp_collection_sum = pd.concat(time_stamp_collection_tmp, axis=1).sum(axis=1)
        print("Summarized number of 1s MFCCs possible when using recycling: ", time_stamp_collection_sum)
        print("1st second distribution: ", bats_table_collection[0]["Class1"].value_counts())

        time_stamp_collection_sum_all_attributes.append(time_stamp_collection_sum)
        with open(f"Overview_{attr}.txt", "w") as output:
            for row in time_stamp_collection:
                output.write(str(row[0]) + "\n" + str(row[1]) + "\n")
            output.write("\n" + f"Summary of possible 1s MFCCs with lookahead={lookahead}" + "\n")
            output.write(str(time_stamp_collection_sum))

    time_stamp_global_sum = pd.concat(time_stamp_collection_sum_all_attributes, axis=1).sum(axis=1)
    with open(f"Overview_{output_global_postfix}.txt", "w") as output:
        output.write("\n" + f"Summary of possible 1s MFCCs across all attributes with lookahead={lookahead}" + "\n")
        output.write(str(time_stamp_global_sum))


if __name__ == "__main__":
    create_overview_of_species_per_timestamp(attribute_W_and_E_2019, 2019, output_global_postfix="2019")
    create_overview_of_species_per_timestamp(attribute_W_and_E_2020, 2020, output_global_postfix="2020")

    create_overview_of_species_per_timestamp(attribute_W_2019, 2019, output_global_postfix="2019_West")
    create_overview_of_species_per_timestamp(attribute_W_2020, 2020, output_global_postfix="2020_West")

    create_overview_of_species_per_timestamp(attribute_E_2019, 2019, output_global_postfix="2019_East")
    create_overview_of_species_per_timestamp(attribute_E_2020, 2020, output_global_postfix="2020_East")
