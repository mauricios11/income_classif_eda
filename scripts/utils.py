#
# current path: ./utils/utils.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Tuple

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 

class Utils:
    def __init__(self):
        pass

    #-funcs 01-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """ Appereance preset for the plots"""
    def load_appereance(self, style: str= 'darkgrid', size: Tuple[int, int] = (10, 8)):
        
        sns.set_style(style, {'gridcolor': '.6', 'grid.linestyle': ':'})
        sns.set_context('notebook')
        plt.rcParams['figure.figsize'] = size # tamaño de las figuras
    
    #-funcs 02-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """split the df columns for have them ready for training
        Args:
            - df: dataframe
            - x (list): columns selected for the analysis
            - y (str) : column selected as target """
    def select_columns(self, df: pd.DataFrame,
                       drop_cols: str | List[str],
                       target_y: str):
        
        df_copy = df.copy()
        x = df_copy.drop(columns= drop_cols, axis= 1)
        y = df_copy[target_y]
        
        return x, y
    
    #-funcs 03-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """Print the shape of provided dataframes.    
        Args:
        - data_list (list): List of pandas DataFrames to print shapes for.
        - train_test (bool): Option to print shapes specifically for train/test split.

        Raises:
        - ValueError: If train_test == True and data_list does not contain exactly 4 DataFrames. """

    def shape(self, data_list: List[pd.DataFrame] = None,
              col_names: List[str]= ['value_1', 'value_2'],
              train_test: bool = False):
        
        print(f'Shape:')
        if data_list and train_test == False:      
            if len(col_names) == len(data_list):
                for i in range(len(data_list)):
                    print(f'- {col_names[i]}: {data_list[i].shape}')
            else: 
                raise ValueError('ERROR: The column_names list must have the same len than data_list')

        if train_test:     
            column_names = ['x_train', 'x_test ', 'y_train', 'y_test ']
            if len(data_list) == len(column_names):
                for i in range(len(column_names)):
                    print(f'- {column_names[i]}: {data_list[i].shape}')
            else:
                raise ValueError('ERROR: La lista proporcionada debe tener 4 valores (train / test split)')
        
        if not data_list:
            raise ValueError('ERROR: No se han proporcionado datos')
    #-func 03-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#         
    # def export_model(self, model, score, params):
    #    print(f'- Best score: {score}')
    #    print(f'- Best params: {params}')
        
    #    # Crear directorio si no existe
    #    os.makedirs('./best_models', exist_ok=True)
    #    score_file = str(round(score, 3)).replace('.', '-') # reemplazamos '.' -> '-'
    #    joblib.dump(model, f'./best_models/best_model_score{score_file}.pkl')

    #-func 04-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#         
   
    
    #-func 05-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#  

        