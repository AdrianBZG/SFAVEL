"""
Preprocessing functions for FB15K-237 dataset
"""

import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from utils import set_seed, GLOBAL_SEED, get_sentence_transformer, save_to_pickle

logging.basicConfig(level=logging.INFO,
                    format='[preprocessing_fb15k237:%(levelname)s] %(message)s')


def _get_entities_relations_text(base_path):
    # Entity to text map
    ent2text = {}
    with open(os.path.join(base_path, "entity2text.txt"), 'r') as f:
        ent_lines = f.readlines()
        for line in ent_lines:
            temp = line.strip().split('\t')
            if len(temp) == 2:
                ent2text[temp[0]] = temp[1]

    # Relations to text map
    rel2text = {}
    with open(os.path.join(base_path, "relation2text.txt"), 'r') as f:
        rel_lines = f.readlines()
        for line in rel_lines:
            temp = line.strip().split('\t')
            rel2text[temp[0]] = temp[1]

    return ent2text, rel2text


def _read_fb15k237_split(file_path, ent2text, rel2text):
    split_data = {}
    with open(file_path) as data_file:
        triplet_lines = data_file.readlines()
        for line in tqdm(triplet_lines):
            temp = line.strip().split('\t')
            assert len(temp) == 3

            head, rel, tail = temp[0], temp[1], temp[2]

            if head not in ent2text or tail not in ent2text or rel not in rel2text:
                continue

            head_text, rel_text, tail_text = ent2text[head], rel2text[rel], ent2text[tail]

            assert all(len(text) for text in [head_text, rel_text, tail_text])

            triplet_head = head_text
            triplet_rel = rel_text
            triplet_tail = tail_text
            triplet_verbalized = f"{triplet_head} {triplet_rel} {triplet_tail}"

            split_data[len(split_data)] = {"head": triplet_head,
                                           "relation": triplet_rel,
                                           "tail": triplet_tail,
                                           "verbalized": triplet_verbalized}

    return split_data


def _compute_fb15k237_embeddings(split_data, language_model):
    split_emb = {}
    split_text = [triplet["verbalized"] for triplet in split_data.values()]

    all_triplets_emb = language_model.encode(list(split_text),
                                             show_progress_bar=True,
                                             convert_to_tensor=False)

    for triplet_key, triplet_emb in zip(list(split_data.keys()), all_triplets_emb):
        split_emb[triplet_key] = triplet_emb

    return split_emb


def _process_fb15k237_split(split_name, file_path, ent2text, rel2text, language_model):
    assert split_name in {"train", "dev", "test"}

    split_data = _read_fb15k237_split(file_path, ent2text, rel2text)
    split_embeddings = _compute_fb15k237_embeddings(split_data, language_model)

    return {"split_data": split_data,
            "split_embeddings": split_embeddings}


def run_preprocessing(config):
    # Load required configs from the config dict
    fb15k237_base_path = os.path.join(config.get('root_path'), 'FB15K-237')
    fb15k237_output_path = config.get('output_path')
    language_model = get_sentence_transformer(config.get('language_model'))

    ent2text, rel2text = _get_entities_relations_text(fb15k237_base_path)

    fb15k237_split_paths = {"train": f"{fb15k237_base_path}/train.tsv",
                            "dev": f"{fb15k237_base_path}/dev.tsv",
                            "test": f"{fb15k237_base_path}/test.tsv"}

    for split_name, split_path in fb15k237_split_paths.items():
        # Obtain the split data and embeddings
        split_data = _process_fb15k237_split(split_name, split_path, ent2text, rel2text, language_model)

        # Save to disk as pickle
        split_data_output_path = os.path.join(f"{fb15k237_output_path}", f"fb15k237_{split_name}_data.pkl")
        split_emb_output_path = os.path.join(f"{fb15k237_output_path}", f"fb15k237_{split_name}_embeddings.pkl")
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
