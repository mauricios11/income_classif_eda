# module for modeling methods
# path: ./modules/modeling_methods.py

#libraries
import joblib # guardar el modelo final
import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd

from sklearn.compose           import  ColumnTransformer
#from sklearn.decomposition     import PCA
from sklearn.ensemble          import (RandomForestClassifier,
                                       StackingClassifier)
from sklearn.feature_selection import  mutual_info_classif
from sklearn.neighbors         import  NearestNeighbors
from sklearn.pipeline          import  Pipeline
from sklearn.preprocessing     import (OneHotEncoder as OHE,
                                       StandardScaler as SS,
                                       LabelEncoder,
                                       FunctionTransformer)

from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import (train_test_split as tts,
                                       cross_val_score as cvs,
                                       GridSearchCV)

from sklearn.metrics           import (make_scorer,
                                       f1_score, 
                                       classification_report,
                                       recall_score,
                                       precision_score)

from xgboost                   import XGBClassifier

from typing import List, Dict


class ModelingMethods:
    #
    #-Func 1-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
    """codificar categóricas por medio de OneHotEncoding"""
    def encode_categoricals(self, df: pd.DataFrame,
                            return_encoder:  bool= False,
                            return_cat_cols: bool= False)-> pd.DataFrame:
        
        # (1) category & numeric cols
        cat_cols = df.select_dtypes(include= ['object', 'category']).columns.tolist()          
        num_cols = df.select_dtypes(include= ['int', 'float']      ).columns.tolist()  
        
        # (2) encoding
        encoder = OHE(sparse_output= False, handle_unknown='ignore')  
        encoded_cat = encoder.fit_transform(df[cat_cols]) #Encoding for categorical
        
        # (3) encoded DF
        encoded_cat_df = pd.DataFrame(encoded_cat,
                                      columns= encoder.get_feature_names_out(cat_cols),
                                      index=   df.index)
        
        # DF: fusion of numerical and encoded categorical
        df_encoded = pd.concat([df[num_cols], encoded_cat_df], axis= 1)
        
        #returns
        if return_encoder and return_cat_cols == False:
            return df_encoded, encoder
        
        elif return_cat_cols and return_encoder == False:
            return df_encoded, cat_cols
        
        elif return_encoder and return_cat_cols:
            return df_encoded, encoder, cat_cols
        
        else:
            return df_encoded
        
    #-Func 2-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
    """Encoding de variables categóricas del DF. 
       - Maneja casos con y sin columna objetivo (`target_col`).

       Args:
        - df (pd.DataFrame)--: DataFrame a procesar.
        - target_col (str)---: Nombre de la columna objetivo (puede ser None para predicción).
       """
    def encode_vars_Xy(self, df: pd.DataFrame,
                       target_col: str = None, # 'None' para predicción de nuevos datos
                       return_originals : bool= False,
                       return_encoded   : bool= False,
                       return_y_encoder : bool= False)-> tuple:

        if df is None:
            raise ValueError('You must provide a DataFrame')
        
        if target_col and target_col in df.columns:
            X = df.drop(columns= target_col)
            y = df[target_col]
            encoder_y = LabelEncoder()
            y_encoded = encoder_y.fit_transform(y) # enconding solo si target es declarado            
     
        X_encoded, encoder_X = self.encode_categoricals(X, return_encoder= True)
        
        return_list = ()
        if return_originals:
            return_list += (X, y)
            
        if return_encoded:
            return_list += (X_encoded, y_encoded)
            
        if return_y_encoder:
             return_list += (encoder_y) # si es necesario retornar encoder de X agregarlo (x, y)
        
        return return_list # revisar función
        
    #-Func 3-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
    """Por medio de un ciclo, se evalúan dos datasets (con y sin outliers)
       - usando Random Forest.
        Args:
            - df_1 (pd.DataFrame): Dataset with outliers.
            - df_2 (pd.DataFrame): Dataset without outliers.
            - target_col (str): Nombre de la columna objetivo.
            - features (list): Columnas seleccionadas como relevantes.

        Returns: (dict) Resultados para ambos datasets"""
    def evaluate_datasets(self, df_with_outliers: pd.DataFrame,
                          df_without_outliers: pd.DataFrame,
                          target_col: str,
                          return_results: bool= False)-> dict | None:
    

        results = {}
        datasets = {'with_outliers': df_with_outliers, 'without_outliers': df_without_outliers}
        
        for name, df in datasets.items():
            print(f"\nEvaluando dataset: {name}")
            
            # X,y
            X, y, X_encoded, y_encoded = self.encode_vars_Xy(df,
                                                             target_col= target_col,
                                                             return_originals = True,
                                                             return_encoded   = True,
                                                             return_y_encoder = False)
                
            # split
            X_train, X_test, y_train, y_test = tts(X_encoded,
                                                   y_encoded,
                                                   test_size=0.3,
                                                   stratify = y,
                                                   random_state=42)
            
            # training
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            # predicción + métricas
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict= True)
            f1_test = f1_score(y_test, y_pred, pos_label= 1)
            
            # CV
            f1_cv_scores = cvs(model, X_train, y_train, cv= 5, scoring= 'f1_macro')
            f1_cv_mean = np.mean(f1_cv_scores)
            
            # resultados
            print(f"F1-score en test: {f1_test:.4f}")
            print(f"F1-score (CV): {f1_cv_mean:.4f}")
            
            df_report = pd.DataFrame(report)
            print(df_report)
            print('-*-' *20,'\n')
            
            # se guardar los resultados
            results[name] = {'model': model,
                             'f1_score_test': f1_test,
                             'f1_score_cv': f1_cv_mean,
                             'classification_report': report }
        
        return results if return_results else None
    
    #-Func 4-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
    """Entrena un modelo Random Forest optimizado con GridSearchCV y guarda el pipeline final.
        Args:
            - df (pd.DataFrame)------: Dataset balanceado con las columnas seleccionadas
            - target_col (str)-------: Nombre de la columna objetivo
            - params_grid (dict)-----: Diccionario con los hiperparámetros a optimizar
            - output_model_path (str): Ruta para guardar el modelo entrenado
    
        Returns: (dict) Contiene el mejor modelo, resultados y métricas clave"""
    def training_rf_bagging(self, df, target_col, param_grid, output_model_path):
        
        # (1) X,y
        X = df.drop(columns= target_col)
        y = df[target_col]
        
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError('Hay valores nulos en el dataset, verificarlos antes de seguir')
        
        #(2) split
        X_train, X_test, y_train, y_test = tts(X, y,
                                               test_size= 0.3,
                                               stratify= y,
                                               random_state= 42)
        # (3) categóricas|numéricas
        cat_cols = X.select_dtypes(include= ['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include= ['int','float']).columns.tolist()
        
        # (4) preprocesamiento (encoding y escalado)
        preprocessor = ColumnTransformer(
                        transformers=[('num', SS(), num_cols),
                                      ('cat', OHE(sparse_output= False,
                                                  handle_unknown='ignore'), cat_cols)
                                      ] )
        
        # (5) modelo y pipeline
        random_forest_model = RandomForestClassifier(bootstrap= True,  # bagging
                                                     oob_score= True,  # out-of-bag (con muestras no usadas)
                                                     warm_start= True, # actualización progresiva
                                                     random_state=42) 
        
        pipeline = Pipeline(steps= [('preprocessor', preprocessor),       # preprocesamiento
                                    ('classifier'  , random_forest_model) # modelo de clasificación
                                    ])
        
        # (6) gridsearch
        grid_search = GridSearchCV(estimator= pipeline, 
                                   param_grid= {'classifier__' + key: value for key, value in param_grid.items()},
                                   cv = 5,
                                   scoring= 'f1_macro', 
                                   n_jobs= -1, # paralelizar: todos los núcleos
                                   verbose= 2)
        # (7) training
        print('Entrenando modelo optimizado con GridSearchCV...')
        grid_search.fit(X_train, y_train)
        
        # (8) resultados: best modelo
        best_pipeline = grid_search.best_estimator_
        print(f'Mejores hiperparámetros: {grid_search.best_params_}')
        
        # (9) predicción + evaluación
        y_pred = best_pipeline.predict(X_test)
        f1_test = f1_score(y_test, y_pred, average= 'macro')
        print(f'F1-score en test: {f1_test:.4f}')
        
        # (10) métricas
        metrics = {'f1_macro': f1_test,
                   'recall': recall_score(y_test, y_pred, average= 'macro'),
                   'precision': precision_score(y_test, y_pred, average= 'macro'),
                   'classification_report': classification_report(y_test, y_pred)}
        print(f'Classification report:\n{metrics["classification_report"]}')
        
        # (11) guardar el mejor modelo
        print(f'Guardando modelo en: {output_model_path}')
        joblib.dump(best_pipeline, output_model_path)
        
        # juntamos los resultados
        return {'best_pipeline': best_pipeline,
                'best_params'  : grid_search.best_params_,
                'metrics'      : metrics}
        
    #-Func 5-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
    """Entrena un modelo Stacking combinando Random Forest y XGBoost, 
        optimizados previamente con GridSearch."""
    def training_stacking(self, df:pd.DataFrame,
                          target_col:str,
                          output_model_path:str):
        # (1) conversión: labels -> numéricas
        df[target_col] = df[target_col].map({'no': 0, 'yes': 1})
        if df[target_col].isnull().any():
            raise ValueError("Las etiquetas no tienen valores válidos. Verifica la columna objetivo.")

        # (2) X,y
        X = df.drop(columns=target_col)
        y = df[target_col]

        # (3) split
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, stratify=y, random_state=42)

        # (4) categóricas & numéricas
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()

        # (5) preprocesamiento
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SS(), num_cols),
                ('cat', OHE(handle_unknown='ignore'), cat_cols)
            ]
        )

        # (6) pipeline 1: optimización de randomforest
        print("Optimizando Random Forest...")
        rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(random_state=42,
                                                                         bootstrap=True))]
                            )
        rf_grid = {
            'classifier__n_estimators': [100, 150, 300],
            'classifier__max_depth': [10, 15, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        rf_search = GridSearchCV(estimator = rf_model,
                                 param_grid= rf_grid,
                                 scoring= 'f1_macro',
                                 cv= 3,
                                 n_jobs= -1,
                                 verbose= 2)
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        print(f"Mejor Random Forest: {rf_search.best_params_}")

        # (7) pipeline 2: optimización de xgboost
        print("Optimizando XGBoost...")
        xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', XGBClassifier(eval_metric='logloss',
                                                                 random_state=42))]
                             )
        xgb_grid = {'classifier__learning_rate': [0.01, 0.1],
                    'classifier__n_estimators': [100, 150],
                    'classifier__max_depth': [3, 5]
                    }
        xgb_search = GridSearchCV(estimator=  xgb_model,
                                  param_grid= xgb_grid,
                                  scoring=   'f1_macro',
                                  cv= 3,
                                  n_jobs= -1,
                                  verbose= 2)
        
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        print(f"Mejor XGBoost: {xgb_search.best_params_}")

        # (8) Meta-modelo
        meta_model = LogisticRegression(random_state=42)

        # (9) stacking classifier (con los mejores modelos)
        stacking_clf = StackingClassifier(estimators=[
                                                ('random_forest', best_rf),
                                                ('xgboost', best_xgb)],
                                          final_estimator=meta_model,
                                          cv=5,
                                          n_jobs=-1
                            )

        # (10) entrenamiento: pipeline completo
        print("Entrenando modelo Stacking...")
        stacking_pipeline = Pipeline(steps=[('stacking', stacking_clf)])
        stacking_pipeline.fit(X_train, y_train)

        # (11) predicción & métricas
        y_test_pred = stacking_pipeline.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))

        # (12) guardar el mejor modelo
        print(f"Guardando modelo en: {output_model_path}")
        joblib.dump(stacking_pipeline, output_model_path)

        return {
            'pipeline': stacking_pipeline,
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
        }

    #-Func 6-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#–#–#–#–#-#-#
    """ Valida un pipeline de entrenamiento con cross-validation y métricas adicionales
        Args:
            - pipeline (Pipeline): Modelo entrenado con preprocesamiento
            - X: datos de entrada (columnas seleccionadas)
            - y: target
            - cv: número de folds para el cross validation
        returns: (dict) métricas de desempeño y puntaje de CV"""
    def validate_pipeline(self, pipeline, df, target_col, cv= 5):
        print('Validando el modelo con CV...')
        
        # (1) X,y
        X = df.drop(columns= target_col)
        y = df[target_col]

        # (2) CV
        f1_scores = cvs(pipeline, X, y, cv= cv, scoring = 'f1_macro', n_jobs= -1)
        f1_mean = np.mean(f1_scores) # para tener un solo valor
        print(f'F1-score (CV): {f1_mean:.4f}')
        
        # (3) predicción & métricas (accuracy, f1_macro)
        y_pred = pipeline.predict(X)
        print(f'Clasiification report:\n{classification_report(y, y_pred)}')


        accuracy = pipeline.score(X, y)
        f1_macro = f1_score(y, y_pred, average= 'macro')
        print(f'Accuracy: {accuracy:.4f}\nF1 macro: {f1_macro:.4f}')
        
        return {'f1_score_cv': f1_mean,
                'accuracy': accuracy,
                'f1_macro': f1_macro}
    
    #-Func 6-#-#-#-#–#-#-#-#-#–#–#–#–#-#-#–#-#–#-#-#-#-#–#–#–#–#-#-#
   