# module: contains the methods that were discarded from the main code
# path: ./modules/discarted_methods.py

# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

import joblib
#from lightgbm                import  LGBMRegressor

from imblearn.under_sampling import  RandomUnderSampler
from imblearn.over_sampling  import  SMOTE

from sklearn.compose         import  ColumnTransformer
from sklearn.decomposition   import  PCA
from sklearn.ensemble        import (RandomForestClassifier,
                                     RandomForestRegressor,
                                     StackingRegressor)
from sklearn.linear_model    import  Ridge
from sklearn.metrics         import (f1_score, classification_report,
                                     make_scorer, mean_absolute_error,
                                     mean_squared_error, r2_score)
from sklearn.model_selection import (train_test_split as tts,
                                     cross_val_score  as cvs,
                                     GridSearchCV)
from sklearn.preprocessing   import (StandardScaler  as SS,
                                     OneHotEncoder   as OHE, LabelEncoder,
                                     FunctionTransformer)
from sklearn.pipeline        import  Pipeline
from sklearn.svm             import  SVR
from xgboost                 import  XGBRegressor
from typing                  import  Dict


def import_to_modules(directory: str= '.', show_content: bool= False):
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

import_to_modules(directory= 'scripts', show_content= False)

# own modules
from load_data import Loader
from module_imbalanced import ImbalancedModule

# class instances
loader      = Loader()
imb_methods = ImbalancedModule()

# original df
df_clean = loader.load_data(file_name= 'adult_imputed', dir= 'clean')


#----------------------------------------------------------------------
# DISCARDED METHODS
#----------------------------------------------------------------------

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

"""imbalanced method by subsets (discarted)
   -  the method is used to balance the target variable in the dataset
   -  strategy: undersampling(for majority class) and oversampling (for the minority class)
   why? -> the performance of the method is not as good as the SMOTE method"""
   
df_subsets = df_clean.copy()

# (1) Variables de entrenamiento y target
X = df_subsets.drop(columns='income')
y = df_subsets['income']

# (2) Encoding de categóricas: OHE
X_encoded, X_encoder, cat_cols = (imb_methods.encode_categoricals(X,
                                                                  return_encoder= True,
                                                                  return_cat_cols= True))
# returns-> numeric df, ohe encoder for reverse transformation

# (3) Codificación -> target : LabelEncoder
y_encoder = LabelEncoder() 
y_encoded = y_encoder.fit_transform(y) #----------------------- "no" -> 0, "yes" -> 1

# (4) split con las variables codificadas
X_train, X_test, y_train, y_test = tts(X_encoded, y_encoded,
                                       test_size=    0.3,
                                       random_state= 42,
                                       stratify=     y_encoded)
# stratify -> mantiene la proporción original de las clases en el test

no_count  = sum(y_train == 0) #-------------------------------- category count (original)
yes_count = sum(y_train == 1)
text_0 = f'Distribución original en train:\nNo:  {no_count}\nYes: {yes_count}'


# (5) size final
final_size = 51980
target_per_class = final_size // 2 # split the final size in half

# (6) Resample / balanceo
## Undersampling (if nedded)
if target_per_class < no_count:
    random_under_sampler = RandomUnderSampler(sampling_strategy={0: target_per_class, 1: yes_count},
                                                  random_state= 42)
    X_res, y_res = random_under_sampler.fit_resample(X_train, y_train)
else:
    X_res, y_res = X_train.copy(), y_train.copy()

## Oversampling (if nedded)
if target_per_class > sum(y_res == 1): #----------- estudiar procedimiento
    smote = SMOTE(sampling_strategy={1: target_per_class}, random_state= 42)
    X_res, y_res = smote.fit_resample(X_res, y_res)

text_1 = f"{'*-*-'*10}\nNuevo tamaño del dataset: {X_res.shape[0]} filas (balanceado)"
text_2 = f"Categorías:\nNo ->  {sum(y_res ==0)}\nYes -> {sum(y_res ==1)}\nFINAL SIZE: {final_size}"


# decodificación y creación de nuevo df
encoded_cols = X_encoder.get_feature_names_out(cat_cols)

X_res_num = X_res.drop(columns= encoded_cols).copy()
X_res_cat_encoded = X_res[encoded_cols]

original_categories = X_encoder.inverse_transform(X_res_cat_encoded)

df_balanced_subsets = X_res.drop(columns= encoded_cols).copy()
for i, col_name in enumerate(cat_cols):
    df_balanced_subsets[col_name] = original_categories[:, i]
        
y_res_original_sb = y_encoder.inverse_transform(y_res)
df_balanced_subsets['income'] = y_res_original_sb


if __name__ == '__main__':
    print('*-Balanced subsets method (discarted)-*\n')
    print(text_0)
    print(f'{text_1}\n{text_2}')
    print(df_balanced_subsets[:3])
    
    df_balanced_subsets['income'].value_counts().plot(kind= 'bar', figsize= (5, 3))
    plt.show()
    
#--------------------------------------------------------------------------------------    
### En este punto, obtenemos un dataset con un size ~ final_size con clases balanceadas
#--------------------------------------------------------------------------------------
###############################################################################
#- modelo descartado para predicción de fnlwgt -#
# model: RandomForestRegressor (con train_test)
# con OHE(categóricas) y SS(numéricas)

"""Entrena un Random Forest Regressor (descartado por bajo rendimiento en métricas)
    - Objetivo: predecir una variable cuya relacion es NO lineal con las demás
        
        Args:
        - df: DataFrame con los datos
        - target_col: nombre de la columna objetivo
        - param_grid: diccionario con los hiperparámetros a probar
        - output_model_path: ruta donde guardar el modelo entrenado"""
def training_forest_regressor(df: pd.DataFrame,
                              target_col: str,
                              param_grid: Dict,
                              output_model_path: str)-> Dict:
        # (1) X,y
        X = df.drop(columns= target_col)
        y = df[target_col]
        
        # (2) split
        X_train, X_test, y_train, y_test = tts(X,y, test_size= 0.3, random_state=42)
        # sin estratificar el problemas de regresión
        # "y" es una variable continua (sin clases)
        
        # (3) categóricas & numéricas
        cat_cols = X.select_dtypes(include= ['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include= ['int', 'float']).columns.tolist()
            
        # (4) preprocesamiento (SS -> num & OHE -> cat)
        preprocessor = ColumnTransformer(
                                transformers= [('num', SS(), num_cols),                        # Escalamos numéricas
                                            ('cat', OHE(sparse_output= False,
                                                        handle_unknown= 'ignore'), cat_cols)]) # encoding a categóricas
        # (5) modelo y pipeline
        forest_regressor_model = RandomForestRegressor(random_state=42)
 

        pipeline = Pipeline(steps= [('preprocessor', preprocessor),
                                    ('model', forest_regressor_model)])
        
        # (6) gridsearch
        grid_search = GridSearchCV(estimator= pipeline,
                                    param_grid= {'model__' + key: value for key, value in param_grid.items()}, # 'model__' en vez de  'classifier__'
                                    scoring= 'neg_mean_squared_error', 
                                    # Métrica principal (RMSE)
                                    # negativo porque queremos maximizar el score y sklearn lo requiere
                                    cv= 5,
                                    n_jobs= -1,
                                    verbose= 2 )
        
        # (7) entrenamiento
        print('Entrenando el modelo...')
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        best_params   = grid_search.best_params_
        best_score    = grid_search.best_score_
        
        y_train_pred = best_pipeline.predict(X_train) # predicción en train y test (comparar)
        y_test_pred  = best_pipeline.predict(X_test)
        
        # resultados
        print(f'Mejor modelo: {best_pipeline}')
        print(f'Mejores hipermarámetros: {best_params}')
        print(f'Mejor score: {best_score}')
        
        # (8) métricas (tanto en train como en test)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test  = r2_score(y_test, y_test_pred)
        
        print(f'\n- MAE (train): {mae_train:.4f}, MAE (test): {mae_test:.4f}')
        print(f'\n- RMSE (train: {rmse_train:.4f}, RMSE (test): {rmse_test:.4f})')
        print(f'\n- R^2 (train): {r2_train:.4f}, R^2 (test): {r2_test:.4f}')
        
        # (9) guardar el mejor modelo
        print(f'Guardando el modelo en : {output_model_path}')
        joblib.dump(best_pipeline, output_model_path)
        
        results = {'best_pipeline': best_pipeline,
                    'best_params'  : best_params,
                    'train metrics': {'mae': mae_train, 'rmse': rmse_train, 'R2': r2_train},
                    'test metrics' : {'mae': mae_test,  'rmse': rmse_test,  'R2': r2_test},
                    }
        
        return results
    
#-Func 6-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
"""Entrena un modelo XGBoost Regressor para predecir fnlwgt (DESCARTADO).

        Args:
            - df (pd.DataFrame): Dataset con las variables necesarias.
            - target_col (str): Nombre de la columna objetivo ('fnlwgt').
            - param_grid (dict): Diccionario con los hiperparámetros a optimizar.
            - output_model_path (str): Ruta para guardar el modelo final.

        Returns:
            - dict: Resultados del desempeño y el mejor modelo entrenado."""
def training_xgb_regressor(self, df, target_col, param_grid, output_model_path):

        # (1) X,y
        X = df.drop(columns=target_col)
        y = df[target_col]

        # (2) numéricas & categóricas
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()

        # (3) preprocesador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SS(), num_cols),                         # Escalar numéricas
                ('cat', OHE(sparse_output=False,
                            handle_unknown='ignore'), cat_cols)  # Codificar categóricas
            ]
        )

        # (4) modelo & pipeline
        xgb_model = XGBRegressor(random_state=42)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', xgb_model)
        ])

        # (5) gridsearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={'model__' + k: v for k, v in param_grid.items()},
            scoring='neg_mean_squared_error',  # Métrica principal: RMSE (negativo porque sklearn lo requiere así)
            cv=3,                              # menos folds para acelerar el proceso
            n_jobs=-1,
            verbose=2
        )

        # (6) dividir datos
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

        # (7) entrenamiento
        print("Entrenando modelo...")
        grid_search.fit(X_train, y_train)

        # (8) resultados
        best_pipeline = grid_search.best_estimator_
        print(f"Mejores hiperparámetros: {grid_search.best_params_}")

        # predicción & métricas
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred  = best_pipeline.predict(X_test)
        
        mae_train  = mean_absolute_error(y_train, y_train_pred)
        mae_test   = mean_absolute_error(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test  = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_train   = r2_score(y_train, y_train_pred)
        r2_test    = r2_score(y_test, y_test_pred)

        print(f"\n- MAE (train): {mae_train:.4f}, MAE (test): {mae_test:.4f}")
        print(f"\n- RMSE (train): {rmse_train:.4f}, RMSE (test): {rmse_test:.4f}")
        print(f"\n- R² (train): {r2_train:.4f}, R² (test): {r2_test:.4f}")

        # 9. guardar el modelo
        print(f"Guardando modelo en: {output_model_path}")
        joblib.dump(best_pipeline, output_model_path)

        return {
            'best_pipeline': best_pipeline,
            'best_params': grid_search.best_params_,
            'train_metrics': {'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train},
            'test_metrics': {'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test}
        }

    
#-Func 7-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
"""Entrena un Support Vector Regressor (SVR) - (DESCARTADO)
        - Objetivo: predecir una variable cuya relacion es NO lineal con las demás
        
        Args:
        - df: DataFrame con los datos
        - target_col: nombre de la columna objetivo
        - param_grid: diccionario con los hiperparámetros a probar
        - output_model_path: ruta donde guardar el modelo entrenado"""
def training_svr(self, df: pd.DataFrame,
                     target_col: str,
                     param_grid: Dict,
                     use_pca: bool,
                     output_model_path: str,
                     n_components: int = None) -> Dict:
        #(1) X,y
        X = df.drop(columns=target_col)
        y = df[target_col]

        #(2) categóricas y numéricas
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()

        #(3) transformers + PCA 
        transformers = []
        if num_cols:
            transformers.append(('num', SS(), num_cols))
        if cat_cols:
            transformers.append(('cat', OHE(sparse_output=False,
                                            handle_unknown='ignore'), cat_cols))
        if use_pca and n_components is not None:
            transformers.append(('pca', PCA(n_components=n_components)))

        if not transformers:
            raise ValueError("No hay columnas para procesar.")

        preprocessor = ColumnTransformer(transformers=transformers)

        #(4) split
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

        # validación: dimensionalidad antes del ajuste
        print(f"Dimensiones originales de X_train: {X_train.shape}")

        #(5) Configución de modelo & pipeline
        svr_model = SVR()
        pipeline  = Pipeline(steps=[('preprocessor', preprocessor), ('model', svr_model)])

        #(6) GridSearchCV
        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=param_grid,
                                   scoring='neg_mean_squared_error',
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=2
                                )

        #(7) entrenamiento
        print("Entrenando el modelo...")
        try:
            grid_search.fit(X_train, y_train)
        except ValueError as e:
            print(f"Error durante el ajuste del modelo: {e}")
            print(f"Dimensiones procesadas después de preprocesamiento: {X_train.shape}")
            raise

        #(8) evaluación: mejor modelo & parámetros
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        #(9) predicción: train & test
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)

        #(10) cálculo de métricas
        mae_train  = mean_absolute_error(y_train, y_train_pred)
        mae_test   = mean_absolute_error(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test  = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_train   = r2_score(y_train, y_train_pred)
        r2_test    = r2_score(y_test, y_test_pred)

        print(f"\n- MAE (train): {mae_train:.4f}, MAE (test): {mae_test:.4f}")
        print(f"\n- RMSE (train): {rmse_train:.4f}, RMSE (test): {rmse_test:.4f}")
        print(f"\n- R² (train): {r2_train:.4f}, R² (test): {r2_test:.4f}")

        #(11) guardar el mejor modelo
        print(f"Guardando el modelo en: {output_model_path}")
        joblib.dump(best_pipeline, output_model_path)

        return {
            'best_pipeline': best_pipeline,
            'best_params': best_params,
            'train_metrics': {'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train},
            'test_metrics': {'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test}
        }
        
    #-Func 8-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
"""
    Entrena un modelo para predecir fnlwgt con un dataset extendido.
    (DESCARTADO)
    
    Args:
        - df (pd.DataFrame): Dataset con las variables necesarias.
        - target_col (str): Nombre de la columna objetivo ('fnlwgt').
        - param_grid (dict): Diccionario con los hiperparámetros a optimizar.
        - output_model_path (str): Ruta para guardar el modelo entrenado.
        - use_pca (bool): Indica si se usa PCA.
        - n_components (int): Número de componentes principales (si PCA está activo).

    Returns:
        - dict: Resultados del desempeño y el mejor modelo entrenado.
"""
def training_fnlwgt_enhanced(self, df: pd.DataFrame,
                            target_col: str,
                            param_grid: Dict,
                            output_model_path: str,
                            use_pca: bool = False,
                            n_components: int = None) -> Dict:
    
        # (1) X,y
        X = df.drop(columns=target_col)
        y = df[target_col]

        # (2) numéricas & categóricas
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()

        # (3) preprocesador con PCA o sin PCA
        transformers = []
        if num_cols:
            transformers.append(('num', SS(), num_cols))
        if cat_cols:
            transformers.append(('cat', OHE(handle_unknown='ignore', sparse_output=False), cat_cols))
        if use_pca and n_components:
            transformers.append(('pca', PCA(n_components=n_components)))
        if not transformers:
            raise ValueError("No hay columnas para procesar.")

        preprocessor = ColumnTransformer(transformers=transformers)

        # (4) modelo + pipeline
        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])

        # (5) gridsearchCV
        grid_search = GridSearchCV(
                        estimator= pipeline,
                        param_grid={'model__' + key: value for key, value in param_grid.items()},
                        scoring='neg_mean_squared_error',
                        cv=5,
                        n_jobs=-1,
                        verbose=2)

        # (6) split
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

        # (7) entrenamiento
        print("Entrenando el modelo...")
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # (8) evaluación
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)

        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        print(f"MAE (train): {mae_train:.4f}, MAE (test): {mae_test:.4f}")
        print(f"RMSE (train): {rmse_train:.4f}, RMSE (test): {rmse_test:.4f}")
        print(f"R² (train): {r2_train:.4f}, R² (test): {r2_test:.4f}")

        # (9) guardar el mejor modelo
        print(f"Guardando el modelo en: {output_model_path}")
        joblib.dump(best_pipeline, output_model_path)

        return {
            'best_pipeline': best_pipeline,
            'best_params': best_params,
            'train_metrics': {'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train},
            'test_metrics': {'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test}
        }
        
#-Func 9-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
    

"""
    (DESCARTADO)
    Analiza el impacto de la nueva columna 'capital_net' en el modelo de predicción de fnlwgt.

    Args:
        - df (pd.DataFrame): Dataset con las variables necesarias.
        - target_col (str): Nombre de la columna objetivo ('fnlwgt').
        - param_grid (dict): Diccionario con los hiperparámetros a optimizar.
        - output_model_path (str): Ruta para guardar el modelo final.
        - use_pca (bool): Si se debe aplicar PCA en el preprocesamiento.
        - n_components (int): Número de componentes principales para PCA (si se usa).

    Returns:
        - dict: Resultados del modelo, métricas y pipeline entrenado.
"""
def analyze_capital_net_impact(self, df,
                                   target_col,
                                   param_grid,
                                   output_model_path,
                                   use_pca=False,
                                   n_components=None):
    
        # (1) X,y
        X = df.drop(columns=target_col)
        y = df[target_col]

        # (2) Categóricas & numéricas
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()

        ## Todas las columnas necesarias deberían están presentes
        if 'capital_gain' not in num_cols or 'capital_loss' not in num_cols:
            raise ValueError("Las columnas 'capital_gain' y 'capital_loss' son necesarias para calcular 'capital_net'.")

        # (2) Función interna para calcular 'capital_net'
        def add_capital_net(X):
            X = X.copy()
            X['capital_net'] = X['capital_gain'] - X['capital_loss']
            return X.drop(columns=['capital_gain', 'capital_loss'])

        # (3) Preprocesamiento
        transformers = [
            ('num', SS(), [col for col in num_cols if col not in ['capital_gain', 'capital_loss']] 
                           + ['capital_net']), #--------- Escalar numéricas + capital_net
            ('cat', OHE(handle_unknown='ignore',
                        sparse_output=False), cat_cols)  # Codificar categóricas
        ]
        if use_pca and n_components is not None:
            transformers.append(('pca', PCA(n_components=n_components)))

        preprocessor = ColumnTransformer(transformers=transformers)

        # (4) Modelo + pipeline
        model = XGBRegressor(random_state=42)
        pipeline = Pipeline(steps=[
                    ('add_capital_net', FunctionTransformer(add_capital_net)),
                    ('preprocessor', preprocessor),
                    ('model', model)
                  ])

        # (5) gridsearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={'model__' + k: v for k, v in param_grid.items()},
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        # (6) split
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

        # (7) training
        print("Entrenando el modelo con 'capital_net'...")
        grid_search.fit(X_train, y_train)

        ## Mejor pipeline
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # (8) predicciones
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)

        # Métricas
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        print(f"\n- MAE (train): {mae_train:.4f}, MAE (test): {mae_test:.4f}")
        print(f"- RMSE (train): {rmse_train:.4f}, RMSE (test): {rmse_test:.4f}")
        print(f"- R² (train): {r2_train:.4f}, R² (test): {r2_test:.4f}")

        # (9) Guardar el mejor modelo
        print(f"Guardando el modelo en: {output_model_path}")
        joblib.dump(best_pipeline, output_model_path)

        return {
            'best_pipeline': best_pipeline,
            'best_params': best_params,
            'train_metrics': {'mae': mae_train, 'rmse': rmse_train, 'r2': r2_train},
            'test_metrics': {'mae': mae_test, 'rmse': rmse_test, 'r2': r2_test}
        }
        
#-Func 10-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
"""stacked model (DESCARTADO)"""
def train_stacking_pipeline(self,df,
                            target_col,
                            param_grids,
                            use_pca= False,
                            n_components= None,
                            output_model_path="stacked_model.pkl"):
        # (1) X,y
        X = df.drop(columns=target_col)
        y = df[target_col]
        
        # (2) split
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)
        
        # (3) preprocesamiento
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
        
        # (4) transformers + PCA
        transformers = []
        
        if num_cols:
            transformers.append(('num', SS(), num_cols))
        if cat_cols:
            transformers.append(('cat', OHE(handle_unknown='ignore', sparse_output=False), cat_cols))
        if use_pca and n_components:
            transformers.append(('pca', PCA(n_components=n_components)))
            
        preprocessor = ColumnTransformer(transformers)
        
        # (5) modelos base
        base_models = [
            ('xgb', GridSearchCV(XGBRegressor(random_state=42),
                                 param_grid=param_grids['xgb'],
                                 cv=5,
                                 scoring='neg_mean_squared_error',
                                 n_jobs=-1)
             ),
        #     ('lgbm', GridSearchCV(LGBMRegressor(random_state=42),
        #                           param_grid=param_grids['lgbm'],
        #                           cv=5,
        #                           scoring='neg_mean_squared_error',
        #                           n_jobs=-1)
        #      ),
        #     ('rf', GridSearchCV(RandomForestRegressor(random_state=42),
        #                         param_grid=param_grids['rf'],
        #                         cv=5,
        #                         scoring='neg_mean_squared_error',
        #                         n_jobs=-1)
        #     )
         ]#NOTE: para activar estos modelos, importar LGBMRegressor y RandomForestRegressor
               # no debería de haber problemas con su implementación
        
        # (5) metamodelo
        meta_model = Ridge()
        
        # (6) pipeline completo
        stack = StackingRegressor(estimators=base_models,
                                  final_estimator=meta_model,
                                  passthrough=True)
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('stacking', stack)])
        
        # (7) entrenamiento
        print("Entrenando el modelo...")
        pipeline.fit(X_train, y_train)
        
        # (8) evaluación: train vs test para overfitting
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        metrics = {
            'train': {
                'mae': mean_absolute_error(y_train, y_train_pred),
                'rmse': mean_squared_error(y_train, y_train_pred, squared=False),
                'r2': r2_score(y_train, y_train_pred)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_test_pred),
                'rmse': mean_squared_error(y_test, y_test_pred, squared=False),
                'r2': r2_score(y_test, y_test_pred)
            }
        }
        
        # (9) guardar el mejor modelo
        print(f"Guardando el modelo en: {output_model_path}")
        joblib.dump(pipeline, output_model_path)
        
        return metrics


        
        
    



