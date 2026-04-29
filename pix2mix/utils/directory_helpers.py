import os

# Train directory helpers

TRAIN_DIRECTORY: str = "out/"
def train_setup(path: str = TRAIN_DIRECTORY) -> tuple[str, str]:
    model_dir = os.path.join(path, "model_weights/")
    os.makedirs(model_dir, exist_ok=True)
    logfile = os.path.join(path, "train.log")
    return model_dir, logfile

def get_modelfile(name: str, location: str) -> str:
    modelfile = os.path.join(location, f"{name}.pt")
    return modelfile

# Test directory helper

def create_test(model_name: str) -> tuple[str, str]:
    if model_name != "unknown":
        model_name = os.path.split(model_name)[1]
        model_name = os.path.splitext(model_name)[0]
    os.makedirs(model_name)
    log_path = os.path.join(model_name, "logs.csv")
    return model_name, log_path

def get_test_directory(name: str, test_path: str) -> tuple[str, str]:
    test_directory = os.path.join(test_path, name)
    os.makedirs(test_directory)
    figure_path = os.path.join(test_directory, f"{name}.png")
    np_path = os.path.join(test_directory, f"{name}.npy")
    return np_path, figure_path