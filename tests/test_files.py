from pathlib import Path 

def test_dataset_exists():
    assert Path("data/raw/titanic.csv").exists()

def test_training_script_exists():
    assert Path("src/train.py").exists()

def test_deploy_script_exists():
    assert Path("src/deploy.py").exists()