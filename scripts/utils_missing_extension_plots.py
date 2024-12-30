# API pandas extension for missing data functions (PLOTS ONLY)
# current path: ./utils/utils_missing_extension_plots.py

# libraries
import itertools
import janitor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import upsetplot
from statsmodels.graphics.mosaicplot import mosaic
from typing import List, Tuple
import warnings

# own functions
from utils_missing_extension import MissingMethods

@pd.api.extensions.register_dataframe_accessor('missing_plot')
#esta clase hereda de Missing Methods
class MissingPlotMethods(MissingMethods):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        
#-#-#-#-Plotting funcs (missing values)-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    def plot_nulls_heatmap(self, cmap: str= 'viridis') -> None:
        df = self._obj
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError('ERROR: The input must be a DataFrame')
        
        if not isinstance(cmap, str):
            raise TypeError("ERROR: The cmap must be a string")
        (   
            df.isnull()
            .transpose()
            .pipe(lambda df: sns.heatmap(data= df,
                                         cmap= cmap,
                                         yticklabels=True))
        )
        plt.show()
    
    #-func 01-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """Plots a bar chart with the number of missing vals per column"""
    def missing_variable_plot(self, df_name: str = 'dataset name', color= 'black'):
        df = self._obj.missing.missing_variable_summary().sort_values('n_missing')

        plot_range = range(1, len(df.index) + 1)

        plt.hlines(y= plot_range, xmin= 0, xmax= df.n_missing, color= color)

        plt.plot(df.n_missing, plot_range, "D", color= color)
        
        plt.title(f'count of null values per column\ndataset: {df_name}',
                  fontsize= 15,  weight= 'bold', loc='left')
        
        plt.yticks(plot_range, df.variable)
        plt.grid(axis = 'y')
        plt.xlabel('Number missing')
        plt.ylabel('Variable')

#-func 02-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """Plots a  histogram of frequencies for the missing values per row"""
    def missing_case_plot(self, df_name: str = 'dataset name'):

        df = self._obj.missing.missing_case_summary()

        sns.displot(data= df, x= 'n_missing', binwidth= 1, color= 'black')

        plt.grid(axis= 'x')
        plt.title(f'rows count with "n" missing values\ndataset: {df_name}',
                  fontsize= 15, weight= 'bold', loc='left')  
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")

#-func 03-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """Plots a grouped bar chart with the proportion of missing vs complete values"""
    # reminder: add tick rotation
    def missing_variable_span_plot(self, column: str,
                                   span_every: int, 
                                   rotation: int= 0,
                                   figsize: Tuple[int,int]= None,
                                   font_size= 15):

        (
            self._obj.missing
                .missing_variable_span(variable  = column,
                                       span_every= span_every)
                .plot.bar(
                    x= 'span_counter',
                    y= ['percent_missing', 'percent_complete'],
                    stacked= True,
                    width= 1,
                    color= ['black', 'lightgray'],
                    figsize= figsize)
        )

        plt.xticks(rotation=rotation)
        plt.xlabel('Span number')
        plt.ylabel('Percentage missing')
        plt.legend(['Missing', 'Present'])
        plt.title(
            f'Percentage of missing values\ncolumn: {column.upper()}. Over a repeating span of {span_every}',
            loc='left', weight= 'bold', fontsize= font_size+3)
        
        plt.grid(False)
        plt.margins(0)
        plt.tight_layout(pad= 0)
        
#-func 04-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """(1) Plots the nulls whose values appear together (in successive rows)
           - using an 'upsetplot'"""
    def missing_upsetplot(self, cols: List[str] = None, **kwargs):
        #pandas version -> 2.2.1
        #upsetplot version -> 0.9.0
        if cols is None:
            cols = self._obj.columns.tolist()  # se convierten 'Series' a una lista

        # null values values count
        missing_data = self._obj.isna().value_counts(subset=cols)
        
        # remove the FutureWarnings
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            # upsetplot -> library
            upsetplot.plot(missing_data, **kwargs)
            
#-func 05-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """description"""
    def missing_mosaic_plot(self, target_col: str,
                            x_col: str,
                            y_col: str,
                            fig_size= (15, 8),
                            turn_off_labels: bool= False,
                            return_ifo: bool= False,
                            stats: bool= False):
        
        df = self._obj.copy()
        label = None
        target_name = target_col + '_bool'
        df_select = df[[target_col, x_col, y_col]]
        
        df_select = (df_select.assign(target_col = lambda df: df[target_col].isna()
                                      .replace([True, False],['NA', '!NA'])))
        
        df_select.rename(columns= {'target_col': target_name}, inplace= True)
        
        group = df_select.groupby([x_col, y_col, target_name],
                                  dropna= False,
                                  as_index= True).size()

        if return_ifo:
            return group
        
        else:
            if turn_off_labels:
                label = lambda key: ''
                
            fig, axes = plt.subplots(1, 1, figsize= fig_size)
            group.pipe(lambda df: mosaic(
                                    data= df,
                                    title= f'Nulls in {target_col} vs ({x_col} & {y_col})',
                                    statistic= stats,
                                    properties= lambda key: {'color': 'orange' if 'NA' in key else 'lightgreen'},
                                    gap= 0.013,
                                    ax = axes,
                                    #horizontal= True,
                                    #axes_label= True,
                                    labelizer= label))
            plt.show()
        
    #-func 05-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    
    
    
    
    
    
    