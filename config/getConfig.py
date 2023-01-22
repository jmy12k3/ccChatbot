import os
from configparser import SafeConfigParser

config_file = os.getcwd() + "/config/settings.ini"
if not os.path.exists(config_file):
    config_file = os.path.dirname(os.getcwd()) + "/config/settings.ini"


def get_config():
    parser = SafeConfigParser()
    parser.read(config_file)
    _conf_ints = [(key, int(value)) for key, value in parser.items("ints")]
    _conf_floats = [(key, float(value)) for key, value in parser.items("floats")]
    _conf_strings = [(key, str(value)) for key, value in parser.items("strings")]
    return dict(_conf_ints + _conf_floats + _conf_strings)
