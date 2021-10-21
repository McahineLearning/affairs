import os
import yaml
import pandas as pd
from get_data import read_params, get_data
import argparse
from patsy import dmatrices

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    df.dropna(inplace=True)
    df = df.rename(columns={"rate_marriage": "rateMarriage", "yrs_married": "yearsMarried","occupation_husb":"husbandOccupation"})
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, sep=",", index=False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)   