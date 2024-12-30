# module to handle  all the methods for 02.ipynb (imbalance data, and EDA)
# path: ./modules/module_imbalanced.py

# libraries
import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd

from imblearn.over_sampling    import  SMOTE
from imblearn.under_sampling   import  RandomUnderSampler

from sklearn.compose           import  ColumnTransformer
from sklearn.decomposition     import  PCA
from sklearn.neighbors         import  NearestNeighbors
from sklearn.pipeline          import  Pipeline
from sklearn.preprocessing     import (OneHotEncoder as OHE,
                                       StandardScaler as SS,
                                       LabelEncoder)
from sklearn.ensemble          import  RandomForestClassifier
from sklearn.feature_selection import  mutual_info_classif
from sklearn.model_selection   import (train_test_split as tts,
                                       cross_val_score as cvs)

from sklearn.metrics           import (make_scorer,
                                       classification_report,
                                       f1_score, 
                                       accuracy_score,
                                       precision_score,
                                       recall_score)
from typing import List, Dict


#-#-#-#–#–#-#-#-#–#–#-#-#-#–#–#-#-#-#–#–#-#-#-#–#–#-#-#-#

class ImbalancedModule:
    #----------------------------------------------------------------------------
    # funciones para encoding, análisis de ruido y balanceo de datos
    #----------------------------------------------------------------------------
    #- func 1-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """codificar categóricas por medio de OneHotEncoding"""
    def encode_categoricals(self, df:pd.DataFrame,
                            return_encoder:  bool= False,
                            return_cat_cols: bool= False):
        
        # (1) category & numeric cols
        cat_cols = df.select_dtypes(include= ['object', 'category']).columns.tolist()          
        num_cols = df.select_dtypes(include= ['int', 'float']      ).columns.tolist()  
        
        encoder = OHE(sparse_output= False, handle_unknown='ignore')  
        encoded_cat = encoder.fit_transform(df[cat_cols]) #Encoding for categorical
        
        # (2) encoded DF
        encoded_cat_df = pd.DataFrame(encoded_cat,
                                      columns= encoder.get_feature_names_out(cat_cols),
                                      index=   df.index)
        
        # (3) DF: fusion of numerical and encoded categorical
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

    #- func 2-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """Find synthetic data over a balanced DF to analyze noise"""
    def find_synthetic_data(self, X_original, X_balanced):
        
        # if not in X_original then it is synthetic
        synthetic_indices = (np.where(~X_balanced       # "~" negation
                                      .isin(X_original) # if not in X_original
                                      .all(axis= 1))    # all columns
                            [0])                        # we take the first element -> index
        
        return synthetic_indices
    
    #- func 3-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """analize noise: measuring distances (original vs synthetic)
       - using KNN to calculate distances"""
    def analyze_noise(self, X_original, X_balanced):
        # (1) encoding para KNN
        X_original_encoded = self.encode_categoricals(X_original)
        X_balanced_encoded = self.encode_categoricals(X_balanced)
        
        # (1.5) tranformación porque los resultados salen raros
        scaler = SS()
        
        ## ajuste y transformación del original
        X_original_scaled = scaler.fit_transform(X_original_encoded)
        
        ## transformación del balanceado  
        X_balanced_scaled = scaler.transform(X_balanced_encoded)      
        
        # (2) identificar datos sintéticos
        synthetic_indices = self.find_synthetic_data(X_original_encoded, X_balanced_encoded)
        synthetic_data    = X_balanced_scaled[synthetic_indices] # sin iloc
        
        # (3) cálculo de distancias -> KNN
        knn = NearestNeighbors(n_neighbors= 5)
        knn.fit(X_original_scaled)                     # Ajuste con originales (scaled)
        distances, _ = knn.kneighbors(synthetic_data)  # Calcular distancias (sintéticos vs originales)
        
        print(f"Distancia promedio entre datos sintéticos y originales: {distances.mean():.4f}")
        return distances
        
    #- func 4-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    def find_best_size(self, df:pd.DataFrame,
                       target_col:str,
                       final_sizes_list= List[int])-> str:
        # (0) X,y
        X = df.drop(columns= target_col)  
        y = df['income']
        X_encoded = self.encode_categoricals(X) #-------------- encoding a categóricas

        # (1) tts
        X_train, X_test, y_train, y_test = tts(X_encoded, y,
                                               test_size= 0.3,
                                               random_state= 42,
                                               stratify= y )

        no_count  = sum(y_train == "no") #------------------- conteos iniciales
        yes_count = sum(y_train == "yes")

        print("Distribución original en entrenamiento:")
        print(f"No: {no_count}, Yes: {yes_count}")

        # (2) rango de tamaños finales a evaluar 
        # sizes con f1 bajos: 25000, 30000, 35000 (descartados)
        final_sizes = final_sizes_list #--------------------- decisión arbitraria en función del contexto del dataset

        results = []
        scorer = make_scorer(f1_score, pos_label= 'yes')
        #Esto arregla el ERROR: "ValueError:
        # pos_label=1 is not a valid label...['no', 'yes']"

        for final_size in final_sizes:
            # (3) determinamos el size de cada clase (balanceada)
            ## división de enteros entre: income= "no" y "yes"
            target_per_class = final_size // 2
            
            # Comprobar límites:
            # - Si target_per_class es mayor al número original de "yes" -> oversampling
            # - Si target_per_class es menor al número original de "no"  -> undersampling
            
            # (4) pipeline para resampling
            # - (a)submuestro de clase mayoritaria: (no)
            if target_per_class < no_count:
                ## reducimos mayoritaria a -> target_per_class
                rus = RandomUnderSampler(sampling_strategy={"no" : target_per_class,
                                                            "yes": yes_count},
                                         random_state=42)
                X_res, y_res = rus.fit_resample(X_train, y_train)
            else:
                ## Si target_per_class >= no_count, NO es necesario submuestrear "no"
                X_res, y_res = X_train.copy(), y_train.copy()
            
            # - (b) submuestro de clase minoritaria: (yes)
            ## Si target_per_class > yes_count, es necesitamos oversampling
            if target_per_class > sum(y_res == "yes"):
                sm = SMOTE(sampling_strategy={"yes": target_per_class}, random_state=42)
                ## Si target_per_class <= yes_count, NO es necesario el oversampling
                X_res, y_res = sm.fit_resample(X_res, y_res)
            
        #--------------------------------------------------------------------------------------    
        ### En este punto, obtenemos un dataset con un size ~ final_size con clases balanceadas
        #--------------------------------------------------------------------------------------

            # (5) Entrenamiento de prueba (RandomForest) para evaluar
            random_forest_clf = RandomForestClassifier(n_estimators= 100, random_state= 42)
            
            # CV: evaluar
            scores = cvs(random_forest_clf,
                         X_res,
                         y_res,
                         cv= 5,
                         scoring= scorer)
            
            mean_score = np.mean(scores)
            results.append((final_size, mean_score))
            print(f"Final size: {final_size}, F1-score (CV): {mean_score:.4f}")

        # (6) se comparan resultados 
        best_config = max(results, key=lambda x: x[1])

        print("\nMejor configuración encontrada:")
        print(f"Tamaño final: {best_config[0]} con F1-score promedio: {best_config[1]:.4f}")
    
    #- func 5-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """Encode categorical columns for model training
       returns one of the following: 
        - (X,y) encoded -> (DataFrame): if  return_only_encoders = True
        - (X, y, X_encoded, y_encoded, y_encoder) -> (DataFrame, array, LabelEncoder)
        - if return_only_encoders = False""" 
    def encode_vars_Xy(self, df: pd.DataFrame = None,
                      return_only_encoders: bool = False)-> tuple:

        if df is not None:
            X = df.drop(columns= 'income')
            y = df['income']
            
            X_encoded = self.encode_categoricals(X)
            y_encoder = LabelEncoder()
            y_encoded = y_encoder.fit_transform(y)
            
        else:
            raise ValueError('You must provide a DataFrame')
        
        if return_only_encoders:
            return X_encoded, y_encoded
        else:
            return X, y, X_encoded, y_encoded, y_encoder
        
    #- func 6-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """Función para evaluar datasets balanceados con un modelo Random Forest. 
        Args:
            X (pd.DataFrame)-: Dataset de características.
            y (pd.Series)----: Columna objetivo.
            name_method (str): Nombre del método de balanceo ('smote' o 'subsets').
        
        Returns:
            dict: Contiene el reporte de clasificación, F1-score y el modelo entrenado"""
    def evaluate_balanced_data(self, df: pd.DataFrame, name_method:str)-> dict:

        # (1) encoding 
        X, y, X_encoded, y_encoded, y_encoder = self.encode_vars_Xy(df= df)

        # split
        X_train, X_test, y_train, y_test = tts(X_encoded, y_encoded,
                                               test_size=    0.3,
                                               random_state= 42,
                                               stratify=     y)
        # (2) Random Forest
        random_forest_clf = RandomForestClassifier(n_estimators= 100, random_state= 42)

        # (3) CV con F1-score
        scorer = make_scorer(f1_score, pos_label=1)
        scores = cvs(random_forest_clf, X_train, y_train, cv= 5, scoring= scorer)
        mean_f1 = np.mean(scores)
        print(f'F1-score (CV) con 51980k: {mean_f1:.4f}')

        # (4) training con valores balanceados
        random_forest_clf.fit(X_train, y_train)

        # (5) predición: valores test
        y_pred = random_forest_clf.predict(X_test)

        f1_test = f1_score(y_test, y_pred, pos_label=1)
        print(f'F1-score en test: {f1_test:.4f}')

        # (6) etiquetas originales
        y_pred_original = y_encoder.inverse_transform(y_pred)
        y_test_original = y_encoder.inverse_transform(y_test)

        # (7) evaluación
        report = classification_report(y_test_original, y_pred_original, output_dict=True)
        print(f'Classification Report ({name_method.upper()}):')
        print(pd.DataFrame(report))
        print('\n','*-*-' *10, '\n')
        
        return {'method': name_method, 'f1_score_cv': mean_f1,
                'f1_score_test': f1_test,'classification_report': report,
                'model': random_forest_clf}
        
    #----------------------------------------------------------------------------
    # funciones para analizar importancia de variables
    #---------------------------------------------------------------------------- 
       
    #- func 7-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """Calculate feature importance with Random Forest, grouping by original columns
        - Args:
            - df (pd.DataFrame)-------: DataFrame with the data
            - target_col (str)--------: Name of the target column
            - rf_estimators (int)-----: Number of estimators for the Random Forest
            - plot (bool)-------------: If True, plot the top 10 features"""
    def calculate_rf_importance(self, df:pd.DataFrame,
                                target_col:str,
                                rf_estimators:int = 100,
                                plot: bool = True) -> pd.DataFrame:
        
        # (1) encoding de X(entrenamiento) e y(target)
        X_encoded, y_encoded = self.encode_vars_Xy(df= df, return_only_encoders= True)

        rf_clf = RandomForestClassifier(n_estimators= rf_estimators, random_state=42)
        rf_clf.fit(X_encoded, y_encoded)
        
        # (2) feature importances
        importance = rf_clf.feature_importances_
        feature_importance = (pd.DataFrame({'Feature': X_encoded.columns,
                                           'Importance': importance})
                              .sort_values(by='Importance', ascending=False))
        
        # (3) agrupamos por cols originales
        original_cols = df.drop(columns= target_col).columns
        grouped_importance = []
        for col in original_cols:
            # (a) se filtran las cols derivadas del OHE que contienen el nombre de la col original
            # (b) solo a estas, se les calcula la importancia total
            # (c) en un diccionario, se agraga, la col en cuestión y su importancia
            
            related_cols = [c for c in X_encoded.columns if col in c]
            total_importance = (feature_importance
                                .loc[feature_importance['Feature'].isin(related_cols), 'Importance']
                                .sum())
            grouped_importance.append({'Feature': col, 'Importance': total_importance})

        # (4) df final
        df_grouped_importance = (pd.DataFrame(grouped_importance)
                                .sort_values(by='Importance', ascending= False))
            
        # (5) gráfico 
        if plot:
            #se muestra el top 10 de cols más importantes, ordenadas de mayor a menor
            plt.figure(figsize= (25, 4))
            plt.barh(df_grouped_importance['Feature'][:10][::-1],
                     df_grouped_importance['Importance'][:10][::-1])
            # Explicación: 
            #-# df['Feature'][:10] -> selecciona las primeras 10 columnas más importantes
            #-#             [::-1] -> invierte el orden de las columnas (de mayor a menor)
            
            plt.title(f'Top 10 Feature Importance (Random Forest) {rf_estimators} estimators',
                      fontsize= 18, fontweight= 'bold', loc= 'left', color= 'gray')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.show()
    
        return df_grouped_importance
    
    #- func 8-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#-#
    
    def calculate_mutual_info(self, df: pd.DataFrame, target_col:str)-> pd.DataFrame:
        # (1) encoding
        X_encoded, y_encoded = self.encode_vars_Xy(df= df, return_only_encoders= True)
        original_cols = df.drop(columns= target_col).columns
        
        # (2) mutual information
        mutual_info = mutual_info_classif(X_encoded, y_encoded, random_state=42)
        ## este es un np.array, no un df
        df_mutual_info = pd.DataFrame({'Feature': X_encoded.columns,
                                       'Mutual Information': mutual_info})
        
        # (3) agrupamos por cols originales
        grouped_mutual_info = []
        for col in original_cols:
        # (a) se filtran las cols derivadas del OHE que contienen el nombre de la col original
        # (b) solo a estas, se les calcula la importancia total
        # (c) en un diccionario, se agraga, la col en cuestión y su importancia 
            related_cols = [c for c in X_encoded.columns if col in c]
            total_mutual_info = (df_mutual_info
                                .loc[df_mutual_info['Feature'].isin(related_cols),
                                    'Mutual Information'].sum())
            
            grouped_mutual_info.append({'Feature': col,
                                        'Mutual Information': total_mutual_info})

        # Resultados como df
        df_m_info = pd.DataFrame(grouped_mutual_info)
        df_m_info = df_m_info.sort_values(by='Mutual Information', ascending=False)
        return df_m_info
    
    #----------------------------------------------------------------------------
    # funciones para modelos ML (evaluación de performance) NO es entrenamiento final
    #----------------------------------------------------------------------------
    
    #- func 9-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """ Entrenamiento de un modelo de prueba de desmpeño.
        - por medio de un pipeline. que incluye PCA después de balancear los datos con SMOTE.
        Args:
        - df_balanced_smote (pd.DataFrame): Dataset balanceado con SMOTE.
        - target_col (str): Nombre de la columna objetivo.
        - n_components (int): Número de componentes principales que retendrá el PCA.

        Returns:
        - dict: Contiene el modelo entrenado, métricas de desempeño y el pipeline completo."""
    def train_model_with_smote_pca(self, df:pd.DataFrame, target_col:str,
                                   n_components:int= 5):

            # (1) X,y
            X = df.drop(columns=[target_col])  
            y = df[target_col] #-------------------------------------------- target
            
            # (2) categóricas & numéricas
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features   = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # (3) preprocesamiento: OHE -> cat, StandardScaler -> num
            preprocessor = ColumnTransformer(
                            transformers=[('cat', OHE(sparse_output=  False,
                                                    handle_unknown= 'ignore'), categorical_features),
                                           # handle_unknown -> valores poco frecuentes
                                          ('num', SS(), numerical_features)]
                                            )
            # (4) PCA + modelo (Random Forest)
            pca = PCA(n_components=n_components)
            model_random_forest = RandomForestClassifier(random_state=42)    # Modelo base
            
            # (5) pipeline: preprocesamiento + PCA + modelo clasif
            pipeline = Pipeline(steps=[ ('preprocessing', preprocessor),     # (OHE + SS)
                                        ('pca', pca),                        # creamos el PCA
                                        ('classifier', model_random_forest)])# entrenamiento
            
            # (6) split
            X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # (7) entrenamiento: pipeline con datos de train
            pipeline.fit(X_train, y_train)
            
            # (8) evaluación d
            y_pred  = pipeline.predict(X_test) #------------------------------ Predicciones
            report  = classification_report(y_test, y_pred, output_dict=True)# métricas (desempeño)
            f1_test = f1_score(y_test, y_pred, pos_label='yes') #------------- ¿cómo se comporta el modelo con datos de test?
            
            # (c) CV con f1 score (capacidad de generalización)
            scorer = make_scorer(f1_score, pos_label= 'yes')
            f1_cv_scores = cvs(pipeline, X_train, y_train, cv=5, scoring=scorer)
            f1_cv_mean = np.mean(f1_cv_scores) #------------------------------ para tener un solo valor
            
            print(f'F1-score en test: {f1_test}\nF1-score (CV) SMOTE + PCA ({n_components} componentes):    {f1_cv_mean}')
            print("\nClassification Report:\n", pd.DataFrame(report))
            
            return {'pipeline': pipeline,
                    'f1_score_test': f1_test,
                    'f1_score_cv': f1_cv_mean,
                    'classification_report': report}
            
    #- func 10-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#-#
    """Evalúa el desempeño de varios datasets en un modelo Random Forest con y sin PCA.
    Args:
        - df: Diccionario con nombre del dataset y el DataFrame asociado.
        - target_col (str): Nombre de la columna objetivo ('income').
        - use_pca (bool): Indica si se aplica PCA o no.
        - n_components (int): Número de componentes PCA (None para desactivar PCA).

    Returns:
        - results (dict): Diccionario con métricas de evaluación para cada dataset."""
    def evaluate_performance(self, df: pd.DataFrame,
                             target_col: str,
                             use_pca: bool = False,
                             n_components: int= None,
                             show_report:bool = True):
        # (1) X,y
        X = df.drop(columns=target_col)
        y = df[target_col]
        
        # (2) split 
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, stratify=y, random_state=42)
        
        # (3) categóricas & numéricas
        num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # (4) preprocesamiento -> OHE + SS
        transformers = []
        if num_cols:
            transformers.append(('num', SS(), num_cols))
        if cat_cols:
            transformers.append(('cat', OHE(sparse_output=False, handle_unknown='ignore'), cat_cols))
    
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # (5) modelo & pipeline
        model_rfc = RandomForestClassifier(random_state=42)
        pipeline  = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('pca', PCA(n_components= n_components)),
                                    ('model', model_rfc)])
        
        # (6) entrenamiento & predicción
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
          
        # (7) Métricas de evaluación
        metrics = {
            'train': {
                'accuracy'       : accuracy_score(y_train, y_train_pred),
                'f1_macro'       : f1_score(y_train, y_train_pred, average='macro'),
                'f1_weighted'    : f1_score(y_train, y_train_pred, average='weighted'),
                'precision_macro': precision_score(y_train, y_train_pred, average='macro'),
                'recall_macro'   : recall_score(y_train, y_train_pred, average='macro')
                },
            'test': {
                'accuracy'       : accuracy_score(y_test, y_test_pred),
                'f1_macro'       : f1_score(y_test, y_test_pred, average='macro'),
                'f1_weighted'    : f1_score(y_test, y_test_pred, average='weighted'),
                'precision_macro': precision_score(y_test, y_test_pred, average='macro'),
                'recall_macro'   : recall_score(y_test, y_test_pred, average='macro')}
            }    
        metrics_df = pd.DataFrame(metrics)
        report= print(classification_report(y_test, y_test_pred, output_dict=False))
                
        return report if show_report else metrics_df
    
    #- func 11-#-#-#-#–#-#–#-#-#-#–#–#–#-#–#-#-#-#-#-#–#–#-#-#-#-#-#-#-#-#-#-#-#
    """Evalúa el desempeño de varios datasets en un modelo Random Forest con y sin PCA."""
    def evaluate_dfs(self, dfs: Dict,
                     target_col: str,
                     n_components: int= None)-> Dict:
        results = {}
        for name, df in dfs.items():
            print(f'Evaluando dataset: {name} without PCA')
            results[f'{name}_without_PCA'] = self.evaluate_performance(df= df,
                                                                target_col= target_col,
                                                                use_pca= False)
            if n_components:
                print(f'Evaluando dataset: {name} with PCA')
                results[f'{name}_with_PCA'] = self.evaluate_performance(df= df,
                                                                target_col= target_col,
                                                                use_pca= True,
                                                                n_components= n_components)
            else:
                print(f'PCA desactivado para: {name}')
                results[f'{name}_with_PCA'] = None
        
        return results


    
    


    