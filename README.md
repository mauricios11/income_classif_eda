# 💸 Predicción de Ingresos Anuales a partir de Datos demográficos 💸

## 📝 Descripción
Este proyecto se centra en la predicción de ingresos anuales usando un dataset del *Census Bureau* de los EEUU. Contiene información demográfica y económica de más de $45,000$ sujetos. El objetivo principal es predecir si una persona tiene un ingreso mayor a $USD $50$ k al año basado en sus características personales y laborales.
* **objetivos**: A parte de entrenar un `RandomForest` como modelo final para realizar predicciones, el principal objetivo del proyecto es evitar tomar decisiones arbitrarias. A lo largo del análisis, nos enfocamos en justificar cada elección mediante evaluaciones estadísticas
* *(bitácora al final del README)*

### Descripción de variables
* **(1) age**: edad del sujeto
* **(2) workclass**: Tipo de trabajo / sector al que pertenece el individuo *(gobierno, privado, sin empleo, etc.)*.
* **(3) fnlwgt**: Peso final del individup en la encuesta (descripción de esta variable más abajo) 
* **(4) education_num**: Nivel educativo del individuo *(formato numérico)* correspondiente a los años de educación completados
* **(5) marital_status**: Estado civil del individuo *(soltero, casado, etc.)*
* **(6) occupation**: A qué se dedica *(ejecutivo, obrero, empleado de gobierno, etc)*.
* **(7) relationship**: Rol **familiar** que el individuo asume dentro del hogar *(jefe de hogar, esposo/a)*.
* **(8) ethnicity**: Etnia del individuo *(Blanco, negro, asiático)*.
* **(9) genre**: género del individuo
* **(10) capital_gain**: Ganancias de vapital adicionales, fuera del salario o ingresos laborales
* **(11) capital_loss**: Pérdidas de capital adicionales
* **(12) hours_per_week**: Horas trabajadas por semana
* **(14) native_country**: País de origen del indiviuo
* **(15) income**: Variable objetivo.
    * no $=$ salario USD$<= 50k$ anuales **(individuo gana 50k anuales o menos)**
    * yes  $=$ salario USD$> 50k$ anuales **(individuo gana más de 50k anuales)**

## 🛠️ Deployment del modelo
El modelo de clasificación entrenado durante este proyecto ha sido desplegado en **streamlit.io**.
* Puedes ingresar a: "[Income Prediction APP](https://mauricios11-random-forest-deployment.streamlit.app)" para probarlo. *(está en inglés porque ando muy bilingüe 🤭 :D)*
* El repositorio del deployment está en el siguiente enlace: "[Random Forest Deployment](https://github.com/mauricios11/random_forest_deployment)"

### 📚 Aclaración de entornos implementados
Trabajamos con dos entornos distintos debido a problemas de compatibilidad entre versiones:.
* **Entorno 1 (EDA)**: Downgrade a`scikit-learn==1.5.2` para evitar advertencias al usar `xgboost`.
* **Entorno 2 (Imbalanced)**: Downgrade a `scikit-learn==1.2.2` para usar `imblearn` durante el balanceo de datos.

### ℹ️ instalación del proyecto:
✂️ En un nuevo directorio para el proyecto, ejecuta el siguiente comando:
```
git clone https://github.com/mauricios11/income_classif_eda.git
```
✂️ **entorno 1: (eda)**:

anaconda
```
conda env create -f eda_environment.yml
conda activate eda
```
pip
```
python -m venv eda
source eda/bin/activate  # Windows: eda\Scripts\activate
pip install -r eda_requirements.txt
```
✂️ **entorno 2: (imbalanced)**

anaconda *(instalar imblearn con pip)*
```
conda env create -f imbalanced_environment.yml
conda activate imbalanced
pip install imblearn
```
pip
```
python -m venv imbalanced
source imbalanced/bin/activate  # Windows: imbalanced\Scripts\activate
pip install -r imbalanced_requirements.txt
```
Si tienen algún problema con la instalación de librerías $\rightarrow$ just hit me up :D *(recomiendo usar anaconda)*

### 📝 Estructura del proyecto
```
├── data
│   ├── clean
│   └── raw
│       ├── adult.csv
│       ├── adult_complete.csv
│       ├── adult_test.test
│       └── adult_validation.csv
├── description.txt
├── modules                           # Módulos auxiliares (listas, diccionarios, métodos)
│   ├── list_and_dicts.py             # listas y diccionarios
│   ├── module_discarted_methods.py   # registro de procesos descartados durante el análisis (bajo desempeño)
│   ├── module_imbalanced.py          # métodos implementados durande el procedo de balanceo
│   ├── module_imputation.py          # para el proceso de imputación
│   ├── module_modeling.py            # entrenamiento de varios modelos
│   └── import_modules.py             # Archivo para manejar todas las importaciones al nb
├── models                            # Modelos entrenados (leer txt)
├── notebooks
│   ├── 00_initial_EDA.ipynb          # EDA inicial
│   ├── 01_EDA_missing.ipynb          # EDA e imputación para valores nulos
│   ├── 02_imbalanced.ipynb           # balanceo de target con dos diferentes métodos
│   ├── 03_model_training.ipynb       # entrenamiento del modelo final (diferentes métodos)
│   ├── 03_training_input.txt         # muestra de uno de los inputs del entrenamiento
│   └── import_modules.py             # Archivo para manejar todas las importaciones al nb
├── scripts
│   ├── load_data.py                  # Funciones para cargar los datos
│   ├── utils.py                      # Funciones generales
│   ├── utils_categorical_plots.py    # Gráficos para variables categóricas
│   ├── utils_classif_models_plots.py # Gráficos relacionados con modelos de clasificación
│   ├── utils_initial_exploration.py  # Exploración inicial
│   ├── utils_missing_extension.py    # Extensiones para manejo de nulos en pandas
│   ├── utils_missing_extension_plots.py # Gráficos para manejo de nulos
│   └── utils_probability_funcs.py    # Funciones para cálculos probabilísticos
```

### 📰 Bitácora de procesos
#### 🕵️‍♀️ Exploración de Datos (EDA)
Iniciamos estudiando el comportamiento general de las variables. No se encontraron relaciones lineales significativas, lo que llevó a investigar posibles relaciones no lineales.

* **Manejo de valores nulos**: Se detectaron nulos implícitos como 'unknown'. Columnas con prefijos como "_other" no fueron modificadas, ya que representaban casos reales pero poco comunes.

* **Imputación de valores**: Se probaron dos métodos de imputación: por moda y un `DecisionTreeClassifier`, siendo este último el de mejor desempeño.

* **Balanceo de datos**: Se utilizaron dos técnicas `SMOTE` *(oversampling)* y una combinación de oversampling + undersampling con un RandomForestClassifier. SMOTE resultó ligeramente superior. 
    * Ajustamos la cantidad de muestras balanceadas para maximizar el f1-score.

* 🔍 **Selección de variables**: Identificamos las columnas más importantes usando:
    * `Feature Importances` + `RandomForest`.
    * `Mutual Information` *(análisis de dependencia)*.
    * En adición, evaluamos el uso de `PCA` para reducir dimensionalidad, pero los resultados fueron inferiores, ya que la mayoría de las variables eran categóricas.

### ❌ Dificultades con fnlwgt
Durante el EDA, la columna `fnlwgt` destacó como una variable significativa para la predicción. Sin embargo, detectamos que en un contexto práctico, a la hora de desplgegar el modelo el usuario no tendría acceso a este valor.

* **Estrategia fallida**: Aceptamos el desafío de predecir `fnlwgt` usando modelos como: *Random Forest, XGBoost, LightGBM, SVR* y un *ensemble stacking*.
    * A pesar de optimizar hiperparámetros y agregar variables derivadas (como `capital_net`), los resultados fueron insatisfactorios
    * Métricas tales como R², MAE y RMSE reflejaron que el modelo era incapaz de generalizar.

* **Reflexión**: Ignorar señales iniciales del EDA *(y estar de necios)* nos llevó a invertir tiempo en un reto poco práctico.
    * A veces, es mejor replantear estrategias en lugar de insistir en soluciones poco factibles.

* **Nueva estrategia**: Se descartó `fnlwgt` y se probó con diferentes configuraciones de variables:
    * *df_no_capital*: Sin columnas adicionales (7 variables)
    * *df_capital_gain*: Incluyendo capital_gain (8 variables).
    * *df_capital_net*: Con la derivada capital_net (8 variables)

### 🤖 Entrenamiento del Modelo

* **Primera iteración**: Entrenamos un `RandomForestClassifier` +  `GridSearchCV`, pero tuvo un desempeño inferior.
* **Modelo final**: Implementamos un `StackingClassifier` combinando $\rightarrow$ 
    * `RandomForestClassifier` +  `XGBoostClassifier` (cada uno optimizado con gridsearch)
    * Administramos el pipeline completo para garantizar escalabilidad y reproducibilidad

### 🌐 Despliegue del Modelo

* **Exportación del modelo**: Tras seleccionar el mejor modelo, lo exportamos para su despliegue en Streamlit.
* **Desafíos durante el despliegue**: El modelo excedía el límite de GitHub. Para arreglarlo usamos `Git LFS` para manejar archivos pesados.
    * otra alternativa hubiera sido comprimir el modelo.

* **Resultado final**: El modelo está desplegado en Streamlit. Los usuarios pueden predecir ingresos anuales en función de los parámetros introducidos