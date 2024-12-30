#
# path: ./scripts/utils_numerical_plots.py

# libraries
import math
import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd
import seaborn as sns

from typing import List



class NumericalPlots:
    
    #- func 1 -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#
    """Noise analisis. Compare two DF's with the same columns (comparison by one target column)
       - Args:
            - df_1, df_2 (pd.DataFrame): DF's to compare (must have the same columns)
            - cols (str) ---------------: Selection of columns (works better with numerical)
            - target (str) -------------: Name of target column
            - target_vale (str) --------: name of the category of the target column (works better with boolean values)
            - df_names (List[str]) -----: List of  strings within the DF's names (for title/label purposes) 
            
       - Example: "num_plots.compare_two_dfs_hist(df1, df2, 'column', 'column val', ['name1', 'name2'])"""
    def compare_two_dfs_hist(self, df_1: pd.DataFrame,
                             df_2: pd.DataFrame,
                             cols: List[str],
                             target: str,
                             target_value:str,
                             rotation: int = 0,
                             colors: List[str] = None,
                             df_names:List[str] = None):
        
        if df_names is None or len(df_names) != 2:
            raise ValueError("You must provide exactly two names for the DF's in -> 'df_names'")
        
        if not set(cols).issubset(df_1.columns) or not set(cols).issubset(df_2.columns):
            raise ValueError("The provided columns do not exist in both DF's")
        
        number_cols = len(cols)
        cols_num = min(3, number_cols) #-------------------------------------- establish the size for the plot grid (subplot)
        rows_num  = math.ceil(number_cols / cols_num) 
            
        fig, axes = plt.subplots(rows_num, cols_num,
                                figsize= (7 * cols_num, 6 * rows_num))
        
        axes = np.array(axes).flatten() #---------------------------------- make sure "axes" is a 1d array
        
        for i, col in enumerate (cols):
            
            if colors is not None and len(colors) == 2:
                color_1 = colors[0]
                color_2 = colors[1]
            if colors is None:
                color_1 = 'blue'
                color_2 = 'orange'
                
            ax = axes[i]
            
            df_a = df_1[df_1[target] == target_value][col]
            df_b = df_2[df_2[target] == target_value][col]
            
            label_1 = f'{df_names[0]} - {target_value}'
            label_2 = f'{df_names[1]} - {target_value}'
            
            sns.histplot(df_a, kde= True, label= label_1, fill= True, color= color_1, alpha= 0.5, ax= ax)
            sns.histplot(df_b, kde= True, label= label_2, fill= True, color= color_2, alpha= 0.5, ax= ax)
            
            if df_names is not None and len(df_names) == 2:
                title_text= f'Comparison of {col.upper()} (hist + KDE):\nTarget: {target} -> {df_names[0]} vs {df_names[1]}'
                ax.set_title(title_text, fontsize= 15, color = 'gray',
                            fontweight= 'bold', loc='left')
                ax.set_xlabel(col, fontsize= 15)
                if rotation != 0:
                    #ax.set_xticklabels(ax.get_xticklabels(), rotation= rotation)
                    ax.tick_params(axis= 'x', rotation= rotation)
                ax.legend()
                
        for j in range(i +1, len(axes)): # --------------------------------- eliminar plots vacíos
            fig.delaxes(axes[j])
               
        plt.tight_layout()
        plt.show()
        
        #- func 2 -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#
        
        
        
        
        
        
        