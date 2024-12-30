# funcs for initial exploration
# current path: ./utils/utils_initial_exploration.py

# libraries
#import joblib
import matplotlib.pyplot as plt
import numpy as np
#import os
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from typing import List, Tuple, Union



class InitialExploration:
    def __init__(self):
        #self.df_obj = df
        pass
        
        
    #-func 1-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """Perform initial exploration of a DataFrame.   
        Args:
            - df (pd.DataFrame)---------------: The DataFrame to explore.
            - data_types (bool, optional)-----: include data types of columns. Defaults to True.
            - nulls_in_cols (bool, optional)--: include null counts in columns. Defaults to True.
            - not_nulls_count (bool, optional): include count of non-null values in columns. Defaults to True.
            - total_nulls (bool, optional)----: include percentage of null values in columns. Defaults to True.
        
        raises:
            ValueError: If most_nulls is True, null_percent must be True.           
        Returns:
            pd.DataFrame: DataFrame containing exploration results.
              
        Example: initial_exploration(df = df_input, show_null_ammount = False, most_nulls = False)"""

    def initial_exploration(self, df  : pd.DataFrame,
                            data_types: bool= True, 
                            nulls_in_cols  : bool= True,
                            not_nulls_count: bool= True,
                            null_percent   : bool= True,
                            show_null_ammount: bool = False,
                            most_nulls     : bool= False) -> pd.DataFrame:
        df_copy = df.copy()#self.df_obj.copy()
        cols = ['type', 'not_null_count', 'null_count', 'null_percent']
        df_info = pd.DataFrame(columns=cols)

        if not_nulls_count:
            df_info['not_null_count'] = df_copy.notnull().sum()
        
        if data_types:
            df_info['type'] = df_copy.dtypes
            
        if nulls_in_cols:
            df_info['null_count'] = df_copy.isnull().sum()
            
        if null_percent:
            df_info['null_percent'] = round(df_copy.isnull().mean() * 100, 3)
            
        if show_null_ammount:
            print(f"Total null values: {df_copy.isnull().sum().sum()}")
            
        if most_nulls:
            if not null_percent:
                raise ValueError("ERROR: null_percent must be True to show column with most null values")
            else:
                print(f"Col with most null values: {df_info['null_count'].idxmax()}-> {df_info['null_percent'].max()}% of null values")
                if df_info['null_percent'].max() >= 10:
                    print("WARNING: This column has more than 10% of null values.")
                    print("(If you consider to use it in your analysis, you should search for more data\n",
                        "or investigate the reason of the null values)")
        
        return df_info

    #-func 2-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """print (range, min, max) values for each numeric column in a DataFrame."""

    def col_range_vals(self, df: pd.DataFrame, col: str = None):
        #df = self.df_obj
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if col: 
            if col in numeric_cols: 
                range_vals = (df[col].max()- df[col].min())
                min_val = df[col].min()
                max_val = df[col].max()
                text = f'Col range: {range_vals}, Min: {min_val}, Max: {max_val}' 
                print(text)
            else:
                raise ValueError(f'ERROR: "{col}" is not a numeric column')
            
        else:
            for col in numeric_cols:
                range_vals = round((df[col].max()- df[col].min()), 2)
                min_val = round(df[col].min(), 2)
                max_val = round(df[col].max(), 2)
                text = f'range: {range_vals}, Min: {min_val}, Max: {max_val} -> name: "{col}"' 
                print(text)


    #-func 3-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """ Calculate the correlation matrix for selected columns of a DataFrame and plot it.
        Notes:
        - Avoid to use categorical columns, it could lead to misleading results.
        - This function calculates the correlation matrix. If its given an already processed DF, it won't work.
        - Made for single plots, not for subplots.
        
        Args:
            - df (pd.DataFrame): DataFrame.
            - columns (list): List with column names to include in correlation matrix.
            - method (str, optional): Select a Method to use: ['pearson', 'kendall', 'spearman']. Defaults to 'pearson'.
            - type_plot (str, optional): Select a plot type: ['clustermap', 'heatmap']. Defaults to 'clustermap'.
            - annot_size (int, optional): Font size for annotations. Defaults to 25.
        
        raises:
            ValueError: If a column in the list does not exist in the DataFrame.
            ValueError: If a column is categorical.
        
        Example: correlation_plot(df = df_penguins, 
                                columns = ['num_col_1', 'num_col_2, 'num_col_3', 'num_col_4'],
                                method = 'pearson', type_plot = 'clustermap', annot_size = 25) """

    def correlation_plot(self, df  : pd.DataFrame,
                         columns   : List,
                         method    : str= 'pearson',
                         type_plot : str = 'clustermap',
                         annot_size: int= 25,
                         title     : str= None):
        #df = self.df_obj
        selected_columns = []
        for col in columns:
            if col not in df.columns: # iterar sobre las columnas
                raise ValueError(f"La columna {col} no existe en el DataFrame.")
            
            unique_vals = df[col].nunique()# cuenta la cantindad de valores  únicos en la col actual
            
            if df[col].dtype == 'object' or df[col].dtype.name == 'category': # verificar si es object, o categoricalDtype (dtype.name == 'category')
                raise ValueError('ERROR: Usage of categorical columns in a clustermap could lead to misleading results.\n - even if the column has been one-hot encoded')
        
        method_list = ['pearson', 'kendall', 'spearman']
        ticks_list =  [1, 0.5, 0, -0.5, -1]
        
        # Correlation matrix
        if method in method_list:
            corr_matrix = df[columns].corr(method= method)
        
        if type_plot in ['heatmap', 'clustermap']:
            if type_plot == 'clustermap':
                sns.clustermap(corr_matrix,
                               annot=True,
                               annot_kws={'fontsize': annot_size},
                               cbar_kws= {'ticks': ticks_list},
                               xticklabels= columns,
                               yticklabels= columns, 
                               center = 0,
                               vmin= -1, vmax= 1,
                               cmap= sns.diverging_palette(20,250, as_cmap= True),
                               linewidths=0.8, linecolor='white')
                
            if type_plot == 'heatmap':
                sns.heatmap(corr_matrix,
                            annot= True,
                            annot_kws= {'fontsize': annot_size},
                            cbar_kws= {'ticks': ticks_list}, 
                            xticklabels= columns,
                            yticklabels= columns, 
                            cmap= sns. diverging_palette(20, 250, as_cmap= True),
                            center = 0,
                            vmin= -1, vmax= 1,
                            linewidth= 0.8,
                            linecolor= 'white')
                if title:
                    plt.title(f'Correlation matrix ({method} algorithm):\nHeatmap\n{title}',
                              fontsize= 20, weight= 'bold')
                else:
                    plt.title(f'Correlation matrix ({method} algorithm):\nHeatmap\n',
                              fontsize= 20, weight= 'bold')
            
        plt.show()
        
    #-func 4-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """One-hot: encode a list of categorical columns in a DataFrame.
        Args:
            - df (pd.DataFrame)-------------: DataFrame to encode.
            - categorical_cols (list)-------: List of column names to encode.
            - drop_original (bool, optional): Drop original columns after encoding. Defaults to False.
        
        raises:
            ValueError: If a column in the list does not exist in the DataFrame.
            
        returns:
            pd.DataFrame: DataFrame with one-hot encoded columns.
        
        example: one_hot(df = df_input, categorical_cols = ['col_1', 'col_2'], drop_original = True)"""

    def one_hot(self, categorical_cols: List[str],drop_original: bool = False)-> pd.DataFrame:
        df = self.df_obj
        df_copy = df.copy()
        
        missing_cols = [col for col in categorical_cols if col not in df_copy.columns]
        if missing_cols:
            raise ValueError(f'{missing_cols} not in DataFrame')
        
        for col in categorical_cols:
            dummies = pd.get_dummies(df_copy[col],prefix=col, dtype= int)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            if drop_original:
                df_copy = df_copy.drop(col, axis=1)
                
        return df_copy

    #-func 5-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #(necesitará el coeficiente de correlación o matriz de confusión calculada previamente)

    def simple_heatmap(self, df: pd.DataFrame,
                       min_val : float | int,
                       max_val : float | int, 
                       center  : float | int = None,
                       title   : str = None,
                       confusion_matrix: bool = False):
        #df = self.df_obj
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(df, annot=True,
                    cmap=sns.diverging_palette(20,250, as_cmap= True),
                    annot_kws={'fontsize': 29},
                    vmax= max_val,
                    vmin= min_val,
                    fmt="",
                    center= center,
                    linewidths=0.5,
                    linecolor='white',
                    cbar= True)
        
        plt.title(title, fontsize= 20, weight= 'bold')
        
        if confusion_matrix:
            print(f'Confusion Matrix\n- correct values for 1st variable..: {df[0,0]}\n- incorrect values for 1st variable: {df[0,1]}\n')
            print(f'- correct values for 2nd variable..: {df[1,1]}\n- incorrect values for 2nd variable: {df[1,0]}')

    #-func 6-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """ Plot all the rows in a heatmap to find the number of null values (by row)    
        Args:
            - df (pd.DataFrame): The DataFrame to plot.
            - cmap (str, optional): The color map to use in the heatmap. (Defaults to "viridis")
        
        raises: 
            - (df)   TypeError: If the input is not a DataFrame.
            - (cmap) TypeError: If the cmap is not a string."""

    def plot_nulls_heatmap(self, df: pd.DataFrame, cmap: str= 'viridis', cbar= False) -> None:
        #df = self.df_obj
        if not isinstance(df, pd.DataFrame):
            raise TypeError('ERROR: The input must be a DataFrame')
        
        if not isinstance(cmap, str):
            raise TypeError("ERROR: The cmap must be a string")
        (   
            df.isnull()
            .transpose()
            .pipe(lambda df: sns.heatmap(data= df,
                                         cmap= cmap,
                                         yticklabels=True,
                                         #ocultar cmap
                                         cbar= cbar
                                         ))
        )
        plt.title('Null values by row\nYellow: missing, Purple: complete',
                  fontsize= 15, weight= 'bold')
        plt.show()
        
        
    #-func 7-#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        """Detecta outliers utilizando Z-score
        Args:
            df (pd.DataFrame): Dataset de entrada.
            threshold (float|int): Umbral para considerar un valor como outlier.
        
        Returns: Dos DF's: Df_1 (con los outliers detectados), DF_2 (SIN outliers)"""
    def outliers_zcore(self, df, threshold= 3):
        # zscore solo a las numéricas
        numerical_cols = df.select_dtypes(include= ['int','float']).columns
        z_scores = zscore(df[numerical_cols])
        
        # detectamos valores con Z-score mayor a threshold
        outliers = (abs(z_scores) > threshold).any(axis= 1) # any(axis= 1) -> cualquier columna
        num_outliers = outliers.sum()
        
        #propoción
        total_rows = df.shape[0]
        percent_outliers = round((num_outliers / total_rows) *100, 2)
        
        print(f'Z-score outliers count: {num_outliers}\nthreshold: {threshold}')
        print(f'Proporción de outliers (vs total de filas): {percent_outliers}')
        return df[outliers], df[~outliers] # outliers, no outliers
    
    #-func 8-#-#-#-#-#-#-#-#-#-#-#-#-#–#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    """Detecta outliers utilizando el rango intercuartílico (IQR). 
    Returns: Dos DF's: Df_1 (con los outliers detectados), DF_2 (SIN outliers)"""

    def outliers_iqr(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include= ['int', 'float']).columns
        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        
        #límites
        lower_outliers = q1 - 1.5 * iqr
        upper_outliers = q3 + 1.5 * iqr
        
        # identificamos los outliers
        outliers = (((df[numeric_cols] < lower_outliers) | (df[numeric_cols] > upper_outliers))
                    .any(axis= 1))
        num_outliers = outliers.sum()
        
        # sacar porcentaje de filas con outliers con respecto al total
        total_rows = df.shape[0]
        percent_outliers = round((num_outliers / total_rows) * 100, 2)
        
        print(f'Detección de outliers con IQR: {num_outliers} filas')
        print(f'Proporción de outliers (vs total de filas): {percent_outliers}')
        
        return df[outliers], df[~outliers]

    
    
    
    



