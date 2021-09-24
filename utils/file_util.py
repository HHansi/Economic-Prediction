# Created by Hansi at 12/1/2020
import os


def create_folder_if_not_exist(path, is_file_path=False):
    """
    Method to create folder if it does not exist
    parameters
    -----------
    :param path: str
        Path to folder or file
    :param is_file_path: boolean, optional
        Boolean to indicate whether given path is a file path or a folder path
    :return:
    """
    if is_file_path:
        folder_path = os.path.dirname(os.path.abspath(path))
    else:
        folder_path = path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
