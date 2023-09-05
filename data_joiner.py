import os
import shutil
import re


def join_data(input_path: os.path, output_path: os.path = "output"):
    directories = os.listdir(input_path)
    for directory in directories:
        out_counter = len(os.listdir(output_path))
        subdirectories = [int(x) for x in os.listdir(os.path.join(input_path, directory))]
        subdirectories.sort()
        for subdirectory in subdirectories:
            os.rename(os.path.join(input_path, directory, str(subdirectory)),
                      os.path.join(output_path, str(subdirectory+out_counter)))


if __name__ == '__main__':
    join_data(r"C:\Users\Pere Alzamora\Desktop\Data_TFG", r"C:\Users\Pere Alzamora\Desktop\output")
