# 
# path: ./modules/module_imputation.py

# libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree          import DecisionTreeClassifier
from typing import List
# own modules


class Imputation_module:
   
#-IMPUTATION PROCESS (MODE)-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """Imputa valores faltantes en las columnas especificadas utilizando la moda."""
    def impute_mode(self, df:pd.DataFrame, cols:str)-> pd.DataFrame:
        for col in cols:
            mode_value = df[col].mode(dropna=True) # <---------- Si hay muchos Nan, este sería la moda (y no se quiere eso)
        
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0]) #-- se asignan los valores imputados
                # iloc[0] --> "mode()" si devuelve varias modas (empate) tomamos la primera
        return df

#-IMPUTATION PROCESS (Decision tree)-#-#-#-#-#-#-#-#-#-#-#-#-#
    """Imputa la columna target_col en el DataFrame df utilizando un árbol de decisión.
    Solo aplica dentro de un subset estratificado.
        Args:    
            - df ------------------: DataFrame con el subset (income fijo)
            - target_col ----------: columna categórica a imputar
            - predictors_to_exclude: lista de columnas a no usar como predictores
              - - ejemplo:(la misma target_col, income, columnas con nulos)
        
        Returns: df con la "target_col" imputada
        """
    def impute_with_model(self, df: pd.DataFrame,
                          target_col: str,
                          predictors_to_exclude=[])-> pd.DataFrame:
        
        # (1) Filas con y sin nulos: target_col
        df_known   = df[df[target_col].notna()].copy() # Filas con valor completo / conocido
        df_unknown = df[df[target_col].isna()].copy()  # Filas con valor faltante / desconocido

        if df_unknown.empty:
            # Si no hay nulos en esta columna, no se hace nada
            return df

        # (2) predictoras
        predictors = df.columns.drop(predictors_to_exclude + [target_col])
        
        ## asegurarse de que NO haya NULOS en los predictores del subset df_known
        cols_with_nans = df_known[predictors].columns[df_known[predictors].isna().any()].tolist()
        predictors = [col for col in predictors if col not in cols_with_nans]
        
        # (3) encoding TEMPORAL para los predictores (one hot):
        # - (a) para las columnas categóricas en df_known/unknown
        # - (b) el one hot es sólo a las categóricas predictoras
        cat_cols = df_known[predictors].select_dtypes(include='object').columns.tolist()
 
        encoder = OneHotEncoder(sparse_output= False,
                                handle_unknown='ignore') # sparse= bool deprecado
        
        encoder.fit(df_known[cat_cols]) #----------------- Entrenamiento del encoder
        
        # (4) transformación: ambos df's
        known_cat_encoded   = encoder.transform(df_known[cat_cols])
        unknown_cat_encoded = encoder.transform(df_unknown[cat_cols])

        # df numérico ara el modelado
        ## (se combinan las cols numéricas con las del OHE)
        num_cols = [col for col in predictors if col not in cat_cols]
        
        X_known = (np.hstack([df_known[num_cols], known_cat_encoded])
                   if num_cols else known_cat_encoded)
        
        X_unknown = (np.hstack([df_unknown[num_cols], unknown_cat_encoded])
                     if num_cols else unknown_cat_encoded)
        
        y_known = df_known[target_col] #------------------- Variable objetivo

        # (5) modelo
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_known, y_known) #---------------------- Entrenamiento del modelo
        y_pred = model.predict(X_unknown) #---------------- Predicción de los valores faltantes

        df.loc[df[target_col].isna(), target_col] = y_pred # Asignamos imputaciones al df original

        return df


#- PLOTS -#–#-#-#-#-#-#-#–#-#-#-#-#-#-#–#-#-#-#-#-#-#–#-#-#-#-#
    """plot the distribution (count) of a column before & after the imputation process
       works better to compare 2 distributions (ORIGINAL vs IMPUTED)"""
    def before_vs_after_imputed(self, df_original: pd.DataFrame,
                                df_imputed: pd.DataFrame,
                                col:str, 
                                imp_strategy: None, title = None, bins = 10):
        
        def annotate_bars(ax):#-------------------------------------------- auxiliar func: value above each bar
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha= 'center', va= 'center', fontsize= 8,
                            color='black', xytext= (0, np.random.randint(2, 15)), #<-- para que no se superpongan
                            rotation= 0,
                            textcoords= 'offset points')
                
        plt.figure(figsize= (25,5))
        plt.hist(df_imputed[col], bins= bins, label= 'imputed', color= 'gold')
        plt.hist(df_original[col].dropna(), bins= bins, label= 'original', color= 'purple')

        plt.legend()
        if imp_strategy == 'mode':
            title_text = f'Distribution by {col.upper()} before vs after imputation (Imp by Mode)'
        if imp_strategy == 'tree':
            title_text = f'Distribution by {col.upper()} before vs after imputation (Imp by Decision Tree)'
        if title is not None:
            title_text = f'Distribution by {col.upper()}. {title}'
            
        plt.title(f'{title_text}', fontsize= 15, fontweight= 'bold', loc= 'left', color= 'gray')
        plt.xticks(rotation= 45, fontsize= 15)
        
        annotate_bars(plt.gca())     
        plt.show()
        
    #-#–#–#–#-#-#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#–#
    """plot the distribution (count) of a column before and after the imputation process
        
            - almost the same than 'before_vs_after_imputed' but allow to use different param colors and labels
            - used different param names to avoid confusion while using the first function to compare different df's"""
    def compare_col_dfs(self, df_1:pd.DataFrame,
                        df_2: pd.DataFrame,
                        col= str,
                        labels=List[str],
                        colors= List[str],
                        rotation = 0,
                        bins= 10,
                        title: str= None)-> None:
        
        title_text= f'Comparison by col:"{col.upper()}". {title}'
        
        def annotate_bars(ax):#-------------------------------------------- auxiliar func: value above each bar
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha= 'center', va= 'center', fontsize= 8,
                                color='black', xytext= (0, np.random.randint(2, 15)), #<-- para que no se superpongan
                                rotation= 0,
                                textcoords= 'offset points')
                        
        plt.figure(figsize= (25, 5))
        plt.hist(df_1[col], bins= bins, alpha= 0.8, color= colors[0], label= labels[0])
        plt.hist(df_2[col], bins= bins, alpha= 0.8, color = colors[1], label= labels[1])
        plt.legend()
        plt.xticks(rotation= rotation)
        plt.title(title_text, fontsize= 18, fontweight= 'bold', loc= 'left', color= 'gray')
        annotate_bars(plt.gca())
        plt.show()
                