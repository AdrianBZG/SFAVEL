"""
Preprocessing functions for Fever dataset
"""

import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from utils import set_seed, GLOBAL_SEED, get_sentence_transformer, get_file_number_of_lines, save_to_pickle

logging.basicConfig(level=logging.INFO,
                    format='[preprocessing_fever:%(levelname)s] %(message)s')


def _read_fever_split(file_path):
    num_entries = get_file_number_of_lines(file_path)
    split_data = {}
    with open(file_path) as data_file:
        for line in tqdm(data_file, total=num_entries):
            claim_data = json.loads(line)
            assert all(key in claim_data for key in ["id", "verifiable", "label", "claim"])
            if claim_data["verifiable"] != "VERIFIABLE":
                # Only care about claims that can be verified
                continue

            claim_id = claim_data["id"]
            claim_label = claim_data["label"]
            claim_text = claim_data["claim"]
            split_data[claim_id] = {"claim": claim_text,
                                    "label": claim_label}

    return split_data


def _compute_fever_embeddings(split_data, language_model):
    split_emb = {}
    split_claims = [claim["claim"] for claim in split_data.values()]

    all_claims_emb = language_model.encode(list(split_claims),
                                           show_progress_bar=True,
                                           convert_to_tensor=False)

    for claim_id, claim_emb in zip(list(split_data.keys()), all_claims_emb):
        split_emb[claim_id] = claim_emb

    return split_emb


def _process_fever_split(split_name, file_path, language_model):
    assert split_name in {"train", "dev", "test"}

    split_data = _read_fever_split(file_path)
    split_embeddings = _compute_fever_embeddings(split_data, language_model)

    return {"split_data": split_data,
            "split_embeddings": split_embeddings}


def run_preprocessing(config):
    # Load required configs from the config dict
    fever_base_path = os.path.join(config.get('root_path'), 'fever')
    fever_output_path = config.get('output_path')
    language_model = get_sentence_transformer(config.get('language_model'))

    fever_split_paths = {"train": f"{fever_base_path}/train.jsonl",
                         "dev": f"{fever_base_path}/paper_dev.jsonl",
                         "test": f"{fever_base_path}/paper_test.jsonl"}

    for split_name, split_path in fever_split_paths.items():
        # Obtain the split data and embeddings
        split_data = _process_fever_split(split_name, split_path, language_model)

        # Save to disk as pickle
        split_data_output_path = os.path.join(f"{fever_output_path}", f"fever_{split_name}_data.pkl")
        split_emb_output_path = os.path.join(f"{fever_output_path}", f"fever_{split_name}_embeddings.pkl")
        save_to_pickle(split_data["split_data"], split_data_output_path)
        save_to_pickle(split_data["split_embeddings"], split_emb_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/preprocessing.json",
                        required=True,
                        help='Path to the json config for preprocessing')

    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.config_path):
        raise ValueError(f'Path to config {args.config_path} does not exist.')

    set_seed(GLOBAL_SEED)

    logging.info(f"Loading preprocessing parameters from: {args.config_path}")
    with open(args.config_path) as file:
        try:
            pp_params = json.load(file)
        except Exception as e:
            logging.error(e)

    logging.info(f"Parameters for preprocessing: {pp_params}")
    run_preprocessing(pp_params)
