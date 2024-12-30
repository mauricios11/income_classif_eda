#
# current path: ./notebooks/import_modules.py

#libraries
import os
import sys

def import_to_nb(directory: str= '.', show_content: bool= False):
    CURRENT_DIR = os.getcwd()
    TARGET_DIR = os.path.join(CURRENT_DIR, os.pardir, directory)
    
    if show_content:
        content = [os.path.join(TARGET_DIR, item) for item in os.listdir(TARGET_DIR)]
        print(f'directory: {directory.upper()}')
        counter = 1
        for item in content:
            print(f'file {counter} :{item}')
            counter +=1
        
    sys.path.append(TARGET_DIR)
    

