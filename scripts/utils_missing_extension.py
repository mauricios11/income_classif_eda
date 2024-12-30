# API pandas extension for missing data analysis
# current path: ./utils/utils_missing_extension.py

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


@pd.api.extensions.register_dataframe_accessor('missing')

class MissingMethods:
    def __init__(self, df: pd.DataFrame):
        self._obj = df
        self.color_list = ['#00b968', '#00b9b6', '#0087b9', '#005fb9', '#0030b9',
                           '#3e00b9', '#7000b9', '#ae00b9', '#b9007b', '#b90043']
        
#-func 0-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    """Common missing values for strings and numbers
        - WARNING: There are cases where these values are not considered as nulls,
                   depending on the context of the values in each dataset"""
    
    def common_missing_values(self)-> Tuple[List[str], List[int | float]]:
        
        common_na_strings = ['missing', 'NA', 'N A', 'n/a', 'n / a', 'N/A', '#N/A',
                             '# N/A', 'N / A', 'na', 'n a', 'nan', 'n a n', 'NaN',
                             'N a N', 'Nan', 'empty', 'Empty', 'EMPTY', ' ', 'blank',
                             'null', 'Null', 'NULL','none', 'None', 'NONE', 'nil',
                             'Nil','NIL', '*', '?', '!', '.', '']

        common_na_nums = [-9999, -999, 99, -1, 0, 99, 999, 9999, 77, 66, 88, 10000, 1000,
                          -9999.0, -999.0, -99.0, -1.0, 0.0, 99.0, 999.0, 9999.0, 77.0,
                          88.0, 66.0, 1000.0, 10000.0]
        
        return common_na_strings, common_na_nums

#-func 1-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    #cantidad de valores nulos
    def number_missing(self) -> int:
        return self._obj.isna().sum().sum()

#-func 2-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    #cantidad de valores no nulos
    def number_complete(self) -> int:
        return self._obj.size - self._obj.missing.number_missing()
    
#-func 3-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    """Conteo de valores nulos en cada columna de un df.
       recibe un df y devuelve: 
       - (1) diccionario con  {'columna': número nulos}
       - (2) print de los valores nulos por columna (summary)

        Args: 
        - print_counts: imprimir conteo de nulos por col (OPCIONAL)
        - save_list: guardar el diccionario con los nulos (OPCIONAL)                    
        - acceptable_nuls: flotante que indica el porcentaje de nulos
                            aceptable en una columna. Default -> (5% => 0.05)
        - summary: resumen de las columnas con nulos (OPCIONAL - Se recomienda activarlo)  """ 
    def count_nulls_per_col(self, print_counts: bool= False,
                            save_list: bool= False,
                            acceptable_nulls: float = 0.05,
                            summary: bool= True) -> dict | str:
        df = self._obj
        dict_nulls = {}
        
        for col in df.columns:
            dict_nulls[col] = df[col].isnull().sum()
            if print_counts:
                print(f'\nNulls in col: "{col.upper()}": {"-*"*3} {df[col].isnull().sum()}')

        if summary:
            calc = df.shape[0] * acceptable_nulls
            #resumen
            cols_no_nulls            = ([col for col, val in dict_nulls.items() 
                                        if val == 0 ])
            
            cols_acceptable_nulls    = [col for col, val in dict_nulls.items()
                                        if val <= calc and val > 0]
            
            cols_non_acceptable_nulls= [col for col, val in dict_nulls.items()
                                        if val > calc]
            
            print(f'ROWS number: {df.shape[0]}, COLUMNS number: {df.shape[1]}\n')
            print(f'- Columns without nulls: \n {cols_no_nulls}\n')
            print(f'- Acceptable columns (nulls less than {int(acceptable_nulls *100)}%):\n {cols_acceptable_nulls}\n')
            print(f'- Too much nulls:\n {cols_non_acceptable_nulls}\n')
            print(f'- Null value tolerance: {acceptable_nulls * 100}%')
        
        if save_list:
            dict_nulls = {key_col: nulls_num for key_col, nulls_num in dict_nulls.items() if nulls_num != 0}
            return dict_nulls
        
#-func 04-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si  
    """Null data by columns. Makes a DF with null data separated by column:
           - Having the proportion of complete vals vs missing values per column"""
    def missing_variable_summary(self) -> pd.DataFrame:
        return self._obj.isnull().pipe(
            
            lambda df_1: ( df_1.sum().reset_index(name= 'n_missing')
                           .rename(columns= {'index': 'variable'})
                           .assign(n_cases= len(df_1),
                                   percent_missing= lambda df_2: df_2.n_missing / df_2.n_cases *100) )
                    )
        
#-func 05-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    """ Makes a DF with the null data (by rows) """
    def missing_case_summary(self) -> pd.DataFrame:
        return self._obj.assign(case= lambda df: df.index,
                                n_missing= lambda df: df.apply(
                                                    axis= 'columns',
                                                    func= lambda row: row.isna().sum() ),
                                percent_missing= lambda df: df['n_missing'] / df.shape[1] * 100,
                               ) [ ['case', 'n_missing', 'percent_missing'] ]

#-func 06-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#revisar función        
    """Makes a DF with the number of missing vals by column
       returns: Proportion for nulls in each column"""
    # def missing_variable_table(self) -> pd.DataFrame:
    #     return (
    #         self._obj.missing.missing_variable_summary()
    #         .value_counts("n_missing")
    #         .reset_index()
    #         .rename(columns= {'n_missing': 'n_missing_in_variable', 0: 'n_variables'})
    #         .assign(pct_variables= lambda df: df.n_variables / df.n_variables.sum() * 100)
    #         .sort_values('pct_variables', ascending= False)
    #     )
        
    def missing_variable_table(self)-> pd.DataFrame:
            df_summary = self._obj.missing.missing_variable_summary()
            counts = (df_summary['n_missing']
                      .value_counts().reset_index())
            counts.columns =  ['n_missing', 'n_variables']
            counts['percent_vars'] =  counts['n_variables'] / counts['n_variables'].sum() *100
            counts = counts.sort_values('percent_vars', ascending = False)
            
            return counts

#-func 07-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# revisar función
    """Makes a DF with the number of missing vals per row
       n -> (nulls | full values), proportion """ 
    # def missing_case_table(self) -> pd.DataFrame:
    #     return (
    #         self._obj.missing.missing_case_summary()
    #         .value_counts('n_missing')
    #         .reset_index()
    #         .rename(columns= {'n_missing': 'n_missing_in_case', 0: 'n_cases'})
    #         .assign(pct_case= lambda df: df.n_cases / df.n_cases.sum() * 100)
    #         .sort_values("pct_case", ascending=False)
    #     )
    def missing_case_table(self) -> pd.DataFrame:
        df_summary = self._obj.missing.missing_case_summary()
        counts = (df_summary['n_missing']
                  .value_counts().reset_index())
        counts.columns = ['n_missing_in_case', 'n_cases']
        counts['percent_case'] = counts['n_cases'] / counts['n_cases'].sum() * 100
        counts = counts.sort_values('percent_case', ascending=False)
        
        return counts
   
#-func 08-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    """Makes a row block separation: indicating -> col and separation interval
           - having: n_missing, n_complete, percent_missing, percent_complete 
           - obtaining: n interval, n nulls, n complete, proportion (all this by row blocks)
           - inherits from n_missing """
    def missing_variable_span(self, variable: str, span_every: int) -> pd.DataFrame:
        return (
            self._obj.assign( span_counter= lambda df: ( 
                                             np.repeat(a= range(df.shape[0]),
                                             repeats= span_every)[: df.shape[0]] )
                            )
            .groupby('span_counter')
            .aggregate( n_in_span=(variable, 'size'),
                        n_missing=(variable, lambda s: s.isnull().sum()) )
            .assign(
                n_complete=  lambda df: df.n_in_span - df.n_missing,
                percent_missing= lambda df: df.n_missing / df.n_in_span * 100,
                percent_complete=lambda df: 100 - df.percent_missing,
            )
            .drop(columns=['n_in_span'])
            .reset_index()
        )
        
#-func 09-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    """(1) Makes a DF with null | full value streak intervals"""
    def missing_variable_run(self, variable) -> pd.DataFrame:
        rle_list = self._obj[variable].pipe(
                lambda s: [[len(list(g)), k] for k, g in itertools.groupby(s.isnull())] )

        return pd.DataFrame(data=rle_list, columns=['run_length', 'is_na']).replace(
            {False: 'complete', True: 'missing'}
        )
        
#-func 10-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    """(1) Makes a df witch sorts the cols (most to least) 
           - by the amount of missing values"""
    def sort_variables_by_missingness(self, ascending = False):

        return (
               self._obj.pipe(lambda df: 
                              (df[df.isna().sum().sort_values(ascending = ascending).index]))
               )

#-func 11-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    def create_shadow_matrix(self, true_string: str = 'missing',
                             false_string: str  = 'NOT MISSING',
                             only_missing: bool = False,
                             suffix: str = '_NA' ) -> pd.DataFrame:
        return (
            self._obj
            .isna()
            .pipe(lambda df: df[df.columns[ df.any() ] ] if only_missing else df)
            .replace( {False: false_string, True: true_string} )
            .add_suffix(suffix)
        )
        
#-func 12-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    """(1) Concatenates a shadow matrix with the current df"""
    def concat_shadow_matrix(self, true_str: str= 'Missing',
                           false_str: str= 'NOT MISSING',
                           only_missing: bool= False,
                           suffix: str = "_NA" ) -> pd.DataFrame:
        
        return pd.concat(objs= [ self._obj,
                                 self._obj.missing.create_shadow_matrix(
                                 true_string= true_str,
                                 false_string= false_str,
                                 only_missing= only_missing,
                                 suffix= suffix)],
                         axis="columns" )
        
#-func 13-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# revisar función
    def missing_scan_count(self, search) -> pd.DataFrame:
        return (
            self._obj.apply(axis= "rows", func= lambda column: column.isin(search))
            .sum()
            .reset_index()
            .rename(columns= {'index': 'variable', 0: 'n'})
            .assign(original_type= self._obj.dtypes.reset_index()[0])
        )


#-#-#-#-Plotting funcs (missing values)-#-#-#-#-#-#-#-#-#-#-#-#-#-#

#-func 14-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    # """(1) Plots a bar chart with the number of missing vals per column"""
    # def missing_variable_plot(self, df_name: str = 'dataset name'):
    #     df = self._obj.missing.missing_variable_summary().sort_values('n_missing')

    #     plot_range = range(1, len(df.index) + 1)

    #     plt.hlines(y= plot_range, xmin= 0, xmax= df.n_missing, color= 'black')

    #     plt.plot(df.n_missing, plot_range, "D", color= "black")
        
    #     plt.title(f'count of null values per column\ndataset: {df_name}',
    #               fontsize= 15,  weight= 'bold', loc='left')
        
    #     plt.yticks(plot_range, df.variable)
    #     plt.grid(axis = 'y')
    #     plt.xlabel('Number missing')
    #     plt.ylabel('Variable')

#-func 15-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    """(2) Plots a  histogram of frequencies for the missing values per row"""
    def missing_case_plot(self, df_name: str = 'dataset name'):

        df = self._obj.missing.missing_case_summary()

        sns.displot(data= df, x= 'n_missing', binwidth= 1, color= 'black')

        plt.grid(axis= 'x')
        plt.title(f'rows count with "n" missing values\ndataset: {df_name}',
                  fontsize= 15, weight= 'bold', loc='left')  
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")

#-func 16-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    # """(1) Plots a grouped bar chart with the proportion of missing vs complete values"""
    # # add tick rotation
    # def missing_variable_span_plot(self, column: str,
    #                                span_every: int, 
    #                                rotation: int= 0,
    #                                figsize: Tuple[int,int]= None,
    #                                font_size= 15):

    #     (
    #         self._obj.missing
    #             .missing_variable_span(variable  = column,
    #                                    span_every= span_every)
    #             .plot.bar(
    #                 x= 'span_counter',
    #                 y= ['percent_missing', 'percent_complete'],
    #                 stacked= True,
    #                 width= 1,
    #                 color= ['black', 'lightgray'],
    #                 figsize= figsize)
    #     )

    #     plt.xticks(rotation=rotation)
    #     plt.xlabel('Span number')
    #     plt.ylabel('Percentage missing')
    #     plt.legend(['Missing', 'Present'])
    #     plt.title(
    #         f'Percentage of missing values\nOver a repeating span of { span_every } ',
    #         loc='left', weight= 'bold', fontsize= font_size+3)
        
    #     plt.grid(False)
    #     plt.margins(0)
    #     plt.tight_layout(pad= 0)

#-func 17-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# si
    # """(1) Plots the nulls whose values appear together (in successive rows)
    #        - using an 'upsetplot'"""
    # def missing_upsetplot(self, cols: List[str] = None, **kwargs):
    #     #pandas version -> 2.2.1
    #     #upsetplot version -> 0.9.0
    #     if cols is None:
    #         cols = self._obj.columns.tolist()  # se convierten 'Series' a una lista

    #     # null values values count
    #     missing_data = self._obj.isna().value_counts(subset=cols)
        
    #     # remove the FutureWarnings
    #     with warnings.catch_warnings():
    #         warnings.simplefilter(action='ignore', category=FutureWarning)
    #         # upsetplot -> library
    #         upsetplot.plot(missing_data, **kwargs)
        
        
#-func 17-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# revisar función
    def scatter_imputation_plot(self, x, y,
                                imputation_suffix='_imp',
                                show_marginal= False,
                                **kwargs):

        x_imputed = f'{ x }{ imputation_suffix }'
        y_imputed = f'{ y }{ imputation_suffix }'

        plot_func = sns.scatterplot if not show_marginal else sns.jointplot

        return (
            self._obj[[x, y, x_imputed, y_imputed]]
            .assign(is_imputed=lambda df: df[x_imputed] | df[y_imputed])
            .pipe(lambda df: (plot_func(data= df, x= x, y= y,
                                        hue= 'is_imputed', **kwargs)))
        )

#-func 18-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# revisar función
    def missing_mosaic_plot(self, target_var: str,
                            x_categorical_var: str,
                            y_categorical_var: str,
                            ax = None ):
        return (
            self._obj
            .assign(                      #target_var = height
                **{f'{target_var}_bool': lambda df: df[target_var].isna().replace([True, False], ['NA', '!NA'])}
            )
            .groupby(
                [x_categorical_var, y_categorical_var, target_var],
                dropna=False,
                as_index=True )
            .size()
            .pipe(
                lambda df: mosaic(
                    data= df,
                    properties= lambda key: {'color': 'orange' if 'NA' in key else 'gray'},
                    ax= ax,
                    horizontal= True,
                    axes_label= True,
                    title= '',
                    #labelizer=lambda key: '',
                    )
                )
            )

#-func 19-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#  revisar función     
    """Creates a pd.Series with the proportion of nulls in a column
    - Args:
        - col (Series, str): column to be analyzed
        - proportion_bellow (float): proportion of nulls below the real range of the data [default: 0.10] 
            - where the nulls will be plotted
        - jitter (float): add noise for data visualization [default: 0.75]""" 
    def col_fill_with_dummies(self, col: pd.Series| str,
                              prop_bellow: float= 0.10,
                              jitter: float= 0.075) -> pd.Series:
        #not_used = self._obj#¿?
        column = col.copy(deep= True)# deep= True -> evitar que se modifique el original
        
        # Extract values metadata
        search_missing_vals = column.isnull()
        number_missing = search_missing_vals.sum() # number of missing values
        column_range = column.max() - column.min()
        
        # shift the values: (Desplazar los valores tal que se ubiquen por debajo de su valor original)
        # shift-> add a random value between 0 and 1
        column_shift = column.min() - column.min() * prop_bellow
        
        # noise around the original values
        np.random.seed(42)
        column_jitter = (np.random.rand(number_missing) - 2) * column_range * jitter
        
        #save new dummie data
        column[search_missing_vals] = column_shift + column_jitter
        
        return column
    
#-func 20-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# revisar función
    """Plots a scatter plot comparing the null values vs the complete values 
       in two columns.
       (Making a shadow matrix with the null values and filling them with dummies)
        - Args:
            - col_x / y (str): selected columns
            - prop_bellow (float): proportion of nulls below the real range of data [default: 0.05]
            - jitter (float): add noise for data visualization [default: 0.075]"""
    def null_vs_complete(self, x_col: str,
                         y_col:str,
                         prop_bellow: float= 0.05,
                         jitter: float= 0.075,
                         palette= None):
        
        #1.- select numeric columns
        num_cols = self._obj.select_dtypes( exclude=['category'] ).columns

        #2.- preserve only columns with missing values
        df_nums = (self._obj[num_cols]
                   .pipe(lambda df: df[ df.columns[df.isna().any(axis= 0)] ]))
        
        #3.- shadow matrix
        df_nums_shdw_mt = (df_nums
                           .missing
                           .concat_shadow_matrix(true_str= True, false_str= False))

        # 4.- fill nulls with dummies (apply)
        ## (a) columns with suffix '_NA' (no changes)
        ## (b) columns without suffix -> fill them with dummies
        ## (c) assign -> separate the 'columns_NA' and without suffix (for plotting)
        ### add a new column that indicates if the columns have nulls or not
        df_shdw_dummies = (df_nums_shdw_mt
                           .apply(lambda col: 
                                   col if '_NA' in col.name else self.col_fill_with_dummies(col,
                                                                                            prop_bellow= prop_bellow, 
                                                                                            jitter= jitter)) # noise
                           
                                ).assign(nullity= lambda df: df[f'{x_col}_NA'] | df[f'{y_col}_NA'])

        titl= f'Null data vs complete data\n in columns: {x_col} & {y_col}'

        df_shdw_dummies.pipe( lambda df: sns.scatterplot(data= df,
                                                         x= x_col,
                                                         y= y_col,
                                                         hue= 'nullity',
                                                         style= 'nullity',
                                                         palette= palette,
                                                         alpha= 0.6))
        
        plt.title(titl, fontsize= 15, fontweight= 'bold', loc= 'left')
        plt.show()
        


    #-func 22-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    """Fill missing values with a statistical measure"""
    def fill_na_with_stat(self, cols, stat):
        df = self._obj
        stats = {
            'mean': df[cols].mean(),
            'median': df[cols].median(),
            'mode': df[cols].mode().iloc[0],  # Usamos iloc[0] para obtener el primer valor del modo
            'std': df[cols].std()
        }
        # config: pandas for future versions
        pd.set_option('future.no_silent_downcasting', True)

        if stat not in stats.keys():
            raise ValueError('ERROR: "stat" must be one of the following: mean, median, mode, std')
        
        df_copy = df.copy()
        for col in cols:
            # fill missing values and then infer data types
            df_copy[col] = df_copy[col].fillna(stats[stat][col])
        
        # inferir tipos de datos depués de hacer el fill
        df_copy = df_copy.infer_objects(copy=False) 
        
        return df_copy
    
    #-func 23-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    def melt_filled_nans_with_stat(self, cols: List[str],
                                   stat: str= 'mean',
                                   plot: bool= True):
    # This function is for imputing missing values for  ONLY numerical columns
        df = self._obj
        #raises
        for col in cols:
            if col not in df.columns:
                raise ValueError(f'ERROR: "{col}" not in df.columns')
        if not any(df[col].dtype in ['int64', 'float64', 'int', 'float'] for col in cols):
            raise ValueError('ERROR: "cols" must be numerical')
        #for col in cols:
        #    df[col] = pd.to_numeric(df[col], errors='coerce') # discarded (optimization)
        if stat not in ['mean', 'median', 'mode', 'std']:
            raise ValueError('ERROR: "stat" must be one of the following: mean, median, mode, std')
        
        selected_cols = cols # columnas numéricas
        df_selected = df[selected_cols].copy()
        
        # aplicar shadow matrix 
        df_selected_shdw = df_selected.missing.concat_shadow_matrix(true_str= True, false_str= False)
        
        # rellenar nulos con la media 
        df_selected_fill_na = df_selected_shdw.missing.fill_na_with_stat(cols= selected_cols, stat= stat)
        
        # pivotar el df
        df_pivot =  df_pivot = (df_selected_fill_na
                                .pivot_longer(index = '*_NA')
                                .pivot_longer(index= ['variable', 'value'],
                                              names_to= 'variable_NA',
                                              values_to= 'imputed_value'))
        if plot:
            (
                df_pivot
                .assign(valid= lambda df: df.apply(
                axis= 'columns', func= lambda column: column['variable'] in column.variable_NA )) # filtramos las columnas que no son nulas
                ).query('valid == True').pipe(lambda df: sns.displot(data= df, 
                                                                    x= 'value',
                                                                    hue= 'imputed_value',
                                                                    col= 'variable',
                                                                    common_bins= False,
                                                                    facet_kws= {'sharex': False, 'sharey':False},
                                                                    palette= 'Set2'))
                
        else:
            return df_pivot
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
