"""
Preprocessing functions for Wikidata dataset
"""

import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

from utils import set_seed, GLOBAL_SEED, get_sentence_transformer, save_to_pickle

logging.basicConfig(level=logging.INFO,
                    format='[preprocessing_wikidata:%(levelname)s] %(message)s')

GET_WIKIBASE_IDS = False
FILTER_WIKIDATA5M = False


def _obtain_entities_aliases(file_path):
    # Opening file
    with open(file_path) as fp:
        num_lines = sum(1 for _ in fp)

    entities_file = open(file_path, 'r')
    entities_aliases = {}

    for line in tqdm(entities_file, total=num_lines):
        entity = line.split("\t")[0].strip().rstrip()
        alias = line.split("\t")[1].strip().rstrip()
        entities_aliases[entity] = alias

    return entities_aliases


def _obtain_relations_aliases(file_path):
    # Opening file
    with open(file_path) as fp:
        num_lines = sum(1 for _ in fp)

    relations_file = open(file_path, 'r')
    relations_aliases = {}

    for line in tqdm(relations_file, total=num_lines):
        relation = line.split("\t")[0].strip().rstrip()
        alias = line.split("\t")[1].strip().rstrip()
        relations_aliases[relation] = alias

    return relations_aliases


def _process_wikidata_triplets(file_path, entities_aliases, relations_aliases):
    with open(file_path) as fp:
        num_lines = sum(1 for _ in fp)

    triplets = {}
    triplets_file = open(file_path, 'r')

    for line in tqdm(triplets_file, total=num_lines):
        head = entities_aliases.get(line.split("\t")[0].strip().rstrip())
        relation = relations_aliases.get(line.split("\t")[1].strip().rstrip())
        tail = entities_aliases.get(line.split("\t")[2].strip().rstrip())

        if not head or not relation or not tail:
            continue

        verbalized = f"{head} {relation} {tail}"
        triplets[len(triplets)] = {"head": head, "relation": relation, "tail": tail, "verbalized": verbalized}

    return triplets


def _obtain_triplets_embeddings(triplets, language_model):
    triplets_emb = {}
    all_triplets_verbalized = [triplet["verbalized"] for triplet in triplets.values()]
    print(f"Creating embeddings for {len(all_triplets_verbalized)} triplets")
    all_triplets_emb = language_model.encode(list(all_triplets_verbalized),
                                             show_progress_bar=True,
                                             convert_to_tensor=False)

    for emb in all_triplets_emb:
        triplets_emb[len(triplets_emb)] = emb

    return triplets_emb


def run_preprocessing(config):
    # Load required configs from the config dict
    wikidata5m_path = os.path.join(config.get('root_path'), 'wikidata5m')
    language_model = get_sentence_transformer(config.get('language_model'))

    entities_aliases = _obtain_entities_aliases(f"{wikidata5m_path}/wikidata5m_entity.txt")
    relations_aliases = _obtain_relations_aliases(f"{wikidata5m_path}/wikidata5m_relation.txt")

    triplets_output_path = os.path.join(config.get('output_path'), 'wikidata_triplets.pkl')
    triplets = _process_wikidata_triplets(f"{wikidata5m_path}/wikidata5m_all_triplet.txt", entities_aliases, relations_aliases)
    save_to_pickle(triplets, triplets_output_path)

    triplets_emb_output_path = os.path.join(config.get('output_path'), 'wikidata_triplets_emb.pkl')
    triplets_emb = _obtain_triplets_embeddings(triplets, language_model)
    save_to_pickle(triplets_emb, triplets_emb_output_path)


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