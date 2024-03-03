import pytest
import torch
import os
import pickle

from sentence_transformers import SentenceTransformer

from src.utils import set_seed, freeze_model, get_model_size, get_available_device, get_huggingface_model, get_sentence_transformer, get_lm_emb_dimensionality, get_file_number_of_lines, save_to_pickle, load_model_from_checkpoint, load_params_from_file, create_contrastive_labels

def test_set_seed():
    set_seed(123)
    assert torch.initial_seed() == 123

def test_freeze_model():
    model = torch.nn.Linear(2, 2)
    freeze_model(model)
    for param in model.parameters():
        assert param.requires_grad == False

def test_get_model_size():
    model = torch.nn.Linear(2, 2)
    assert get_model_size(model) == 6

def test_get_available_device():
    device = get_available_device()
    assert device in ['cpu', 'cuda']

def test_get_huggingface_model():
    model_dict = get_huggingface_model('bert-base-uncased')
    assert 'tokenizer' in model_dict
    assert 'model' in model_dict

def test_get_sentence_transformer():
    model = get_sentence_transformer('bert-base-nli-mean-tokens')
    assert isinstance(model, SentenceTransformer)

def test_get_lm_emb_dimensionality():
    dim = get_lm_emb_dimensionality('bert-base-nli-mean-tokens')
    assert dim == 768

def test_get_file_number_of_lines():
    with open('test.txt', 'w') as f:
        f.write('Hello\nWorld')
    assert get_file_number_of_lines('test.txt') == 2
    os.remove('test.txt')

def test_save_to_pickle():
    data = {'hello': 'world'}
    save_to_pickle(data, 'test.pkl')
    with open('test.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    assert loaded_data == data
    os.remove('test.pkl')

def test_load_model_from_checkpoint():
    model = torch.nn.Linear(2, 2)
    torch.save(model, 'model_checkpoints/test.pt')
    loaded_model = load_model_from_checkpoint('model_checkpoints/test.pt')
    assert isinstance(loaded_model, torch.nn.Linear)
    os.remove('model_checkpoints/test.pt')
