import json
import os
import pickle

import torch
import random
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

GLOBAL_SEED = 56739723
SENTENCE_TRANSFORMER = None
logging.basicConfig(level=logging.INFO,
                    format='[utils:%(levelname)s] %(message)s')


def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_model(model_to_freeze):
    for param in model_to_freeze.parameters():
        param.requires_grad = False


def get_model_size(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters


def get_available_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"  # Many issues with MPS so will force CPU here for now
    else:
        device = "cpu"

    return device


def get_huggingface_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.is_fast:
        raise ValueError('Only fast tokenizers are supported.')

    model = AutoModel.from_pretrained(model_name).to(get_available_device())

    return {"tokenizer": tokenizer,
            "model": model}


def get_sentence_transformer(model_name):
    global SENTENCE_TRANSFORMER
    if not SENTENCE_TRANSFORMER:
        SENTENCE_TRANSFORMER = SentenceTransformer(model_name, device=get_available_device())

    return SENTENCE_TRANSFORMER


def get_lm_emb_dimensionality(model_name=None):
    if not model_name:
        # Random assignment (no LLM)
        return 128

    with torch.no_grad():
        temp_model = get_sentence_transformer(model_name)
        dummy_input = ["This is a dummy input."]
        dummy_emb = temp_model.encode(dummy_input, show_progress_bar=False, convert_to_tensor=False)
        lm_emb_dim = dummy_emb[0].shape[0]

    return lm_emb_dim


def get_file_number_of_lines(file_path):
    with open(file_path) as fp:
        num_lines = sum(1 for _ in fp)

    return num_lines


def save_to_pickle(data, save_path):
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_from_checkpoint(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f'Model checkpoint does not exist at {model_path}')

    model = torch.load(model_path)
    return model


def load_params_from_file(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f'Path to config {file_path} does not exist.')

    logging.info(f"Loading parameters from: {file_path}")
    with open(file_path) as file:
        try:
            params = json.load(file)
        except Exception as e:
            logging.error(e)

    return params


def create_contrastive_labels(num_pos_pairs, previous_max_label, device='cpu'):
    # create labels that indicate what the positive pairs are
    labels = torch.arange(0, num_pos_pairs)
    labels = torch.cat((labels, labels)).to(device)
    # add an offset so that the labels do not overlap with any labels in the memory queue
    labels += previous_max_label + 1
    # we want to enqueue the output of encK, which is the 2nd half of the batch
    enqueue_mask = torch.zeros(len(labels)).bool()
    enqueue_mask[num_pos_pairs:] = True
    return labels, enqueue_mask
