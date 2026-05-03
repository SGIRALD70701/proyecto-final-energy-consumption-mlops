import os


def test_config_exists():
    assert os.path.exists("config.yaml")


def test_data_folder_exists():
    assert os.path.exists("data")


def test_train_script_exists():
    assert os.path.exists("src/train.py")