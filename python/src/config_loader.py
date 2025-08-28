import importlib.util
import sys
import os

_config_modules = {}

def load_parameters(config_name):
    if config_name in _config_modules:
        return _config_modules[config_name]

    config_path = os.path.join(os.getcwd(), config_name)
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[config_name] = module
    spec.loader.exec_module(module)

    _config_modules[config_name] = module
    return module
