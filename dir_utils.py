import os
from datetime import datetime

def get_stem(file_path):
    basename = os.path.basename(file_path)
    stem = os.path.splitext(basename)[0]
    return stem


def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')[:-7]


def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
