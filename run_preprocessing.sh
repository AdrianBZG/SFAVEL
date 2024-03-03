CONFIG_PATH="configs/preprocessing.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/preprocessing/preprocessing_fb15k237.py --config_path=$CONFIG_PATH
python src/preprocessing/preprocessing_fever.py --config_path=$CONFIG_PATH
python src/preprocessing/preprocessing_wikidata.py --config_path=$CONFIG_PATH