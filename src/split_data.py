import os
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params
from patsy import dmatrices

def split_and_saved_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"] 
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_data_path, sep=",")
    df['affair'] = (df.affairs > 0).astype(int)
    y, X = dmatrices('affair ~ rateMarriage + age + yearsMarried + children + \
                  religious + educ + C(occupation) + C(husbandOccupation)',
                  df, return_type="dataframe")


    # fix column names of X
    X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                            'C(occupation)[T.3.0]':'occ_3',
                            'C(occupation)[T.4.0]':'occ_4',
                            'C(occupation)[T.5.0]':'occ_5',
                            'C(occupation)[T.6.0]':'occ_6',
                            'C(husbandOccupation)[T.2.0]':'occ_husb_2',
                            'C(husbandOccupation)[T.3.0]':'occ_husb_3',
                            'C(husbandOccupation)[T.4.0]':'occ_husb_4',
                            'C(husbandOccupation)[T.5.0]':'occ_husb_5',
                            'C(husbandOccupation)[T.6.0]':'occ_husb_6'})


    data = X
    data['affair'] = y





    train, test = train_test_split(
        data, 
        test_size=split_ratio, 
        random_state=random_state
        )
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)
