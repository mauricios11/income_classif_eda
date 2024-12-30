# 💸 Predicción de Ingresos Anuales a partir de Datos demográficos 💸

## 📝 Descripción del proyecto
Este proyecto se centra en la predicción de ingresos anuales usando un dataset del *Census Bureau* de los EEUU. Contiene información demográfica y económica de más de $45,000$ sujetos. El objetivo principal es predecir si una persona tiene un ingreso mayor a $USD $50$ k al año basado en sus características personales y laborales.
* *(bitácora al final del README)*

incluye las siguientes variables:
(*"$n$ tipos distintos"* $\rightarrow$ equivale al número de elementos/categorías diferentes que hay en cada columna **categórica**): 
* **(1) age**: edad del sujeto
* **(2) workclass**: Tipo de trabajo / sector al que pertenece el individuo *(gobierno, privado, sin empleo, etc.)*. $9$ tipos distintos
* **(3) fnlwgt**: Peso final del individup en la encuesta (descripción de esta variable más abajo) 
* **(4) education_num**: Nivel educativo del individuo *(en formato numérico)* correspondiente a los años de educación completados
* **(5) marital_status**: Estado civil del individuo *(soltero, casado)*
* **(6) occupation**: A qué se dedica *(ejecutivo, obrero, empleado de gobierno, etc)*. $15$ tipos distintos
* **(7) relationship**: Rol **familiar** que el individuo asume dentro del hogar *(jefe de hogar, esposo/a)*. $6$ tipos distintos
* **(8) ethnicity**: Etnia del individuo *(Blanco, negro, asiático)*. $5$ tipos distintos
* **(9) genre**: género del individuo. $2$ tipos distintos
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
Durante el análisis exploratorio hemos trabajado con dos entornos distintos. La razón de esto es porque la librería encargada de hacer el balanceo de datos necesita un versión específica de *sklearn*, una versión anterior cuya compatibilidad con *seaborn* y otras librerías da algunos problemas.
* teóricamente, la diferencia principal entre ambos entornos es que el que está hecho para datos desbalanceados usa `scikit-learn==1.2.2` + `imblearn`
* en adición, se hizo un downgrade en el entorno *eda* a `scikit-learn==1.5.2` por un festival de warnings que salían al usar `xgboost`

### ℹ️ instalación del proyecto:
✂️ En un nuevo directorio para el proyecto, ejecuta el siguiente comando:
```


```
✂️ entorno 1: (eda):
```
#anaconda
conda env create -f eda_environment.yml

#pip
pip install --upgrade pip
pip install -r eda_requirements.txt

```
✂️ entorno 2: (imbalanced) 
```
#anaconda (instalar imblearn con pip)
conda env create -f imbalanced_environment.yml
pip install imblearn

#pip
pip install --upgrade pip
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

se comenzó con un EDA, analizando el comportamiento de las variables en general, sin encontrar muchas columnas relación lineal significativa (posteriormente se hizo un análisis para encontrar relaciones no lineales)

se detectaron valores nulos implícitos como 'unknnown', en adición se ancontró otro tipo de columnas con prefijo "_other" las cuales se decidió no tocar dado que se trataba de registros reales pero poco comunes (por eso no figuraban como categoría con un nombre específico)

imputación por la moda e imputación por medio de un DecisionTreeClassifier (el segundo con mejor desempeño)
balanceo por SMOTE (oversampling a minoritarias) y se encontró el mejor punto para tener la cantidad de muestras tal que diera el mejor f1-score. después se hizo un  balanceo por subsets (oversampling a minoritarias + undersampling con RandomForestClassifier)
el smote termino siendo ligeramente mejor

continuamos con la selección de las columnas más importantes mediante feature importances y mutual information

comparamos el desempeño con un PCA, pero resultó ser menor (tiene sentido porque no hay muchas columnas, y las que hay en su mayoría son categóritas)
    <p>Durante el análisis exploratorio en pasos anteriores encontramos que una de las variables más importantes en el análisis inicial fue <i>fnlwgt</i>. Sin embargo, al planear el despliegue del modelo surgió una preocupación:</p> 
    <ul>
        <li>El usuario no tendría acceso a esta cifra <i>(fnlwgt)</i> sin los datos y procesos específicos del dataset original. Por esta razón, aceptamos el reto de crear un modelo capaz de predecir este valor, dada su importancia en el desempeño del modelo principal.</li>
    </ul>
    <p><b>Estrategia</b>:</p>
    <ul>
        <li>Se exploraron varios algoritmos: <i>Random Forest Regressor, XGBoost, LightGBM, SVR</i> junto con una estrategia de Modelos de ensamble<i>(stacking)</i> combinando varios de los anteriores.</li>
    </ul>
    <p>Todos estos métodos fueron optimizados mediante la búsqueda de hiperparámetros (GridSearchCV) y ajustes adicionales como:</p>
    <ul>
        <li>PCA para reducción de dimensionalidad.
        <li>Incorporación de variables derivadas como <code>capital_net</code> <i>(basado en capital_gain y capital_loss).</i></li>
        <li>Variantes con distintos conjuntos de columnas consideradas importantes.</li>
    </ul>
    <p>A pesar de las múltiples iteraciones con distintos métodos, los resultados en todos los casos tuvieron un desmempeño poco aceptable</p>
    <ul>
        <li>La métricas: <b>R², MAE y RMSE</b> (haciendo incapié en la primera) reflejaron un modelo incapaz de generalizar la complejidad de <i>fnlwgt</i></li>
    </ul>
<b>reflexión</b>: A pesar de haber tenido indicios de que esta variable podría ser problemática desde el principio del análisis exploratorio, ignoramos tal evidencia y continuamos con el análisis, aceptando el reto de enfrentarse a predecir esta variable.

* Esta es una prueba sobre la importancia sobre evitar ignorar indicios que los datos nos dan al princpio del EDA, y que no todos los valores importantes en un análisis inicial son predicibles en un contexto práctico. fnlwgt, aunque relevante para predecir income, no fue posible de modelar con la precisión deseada <>(incluso tras extensos ajustes)
* A veces es más eficiente reevaluar y rediseñar estrategias en lugar de insistir en una solución que no es factible en el contexto práctico.

 <b style="font-size: 1.5em;">💭 Nueva estrategia</b>
    <p>Se descarta <code>fnlwgt</code>como variable en el modelo final que busca predecir <code>income</code>. Para compensar su ausencia, se agregarán más columnas presentes en las <b>feature importances</b> | <b>mutual information</b></p>
    <br>
    <p>Para contrarrestar la pérdida de <code>fnlwgt</code> evaluaremos diferentes instancias del df con las columnas más importantes:</p>
    <ul>
        <li> <b>df_no_capital</b>: sin agregar más columnas (total de 7)</li>
        <li> <b>df_capital_gain</b>: agregando otra feature importance con menos relevancia: <code>capital_gain</code> (total de 8)
        <li> <b>df_capital_net</b>: haciendo una columna nueva: <code>capital_net</code> <i>(direfencia entre capital_gain y capital_loss)</i> (total de 8)</li>
    </ul>
</div>

finalmente entrenamos un RandomForest classifier con un gridsearch para encontrar los mejores parámetros, aunque se descartó por tener un bajo desempeño, el modelo que ha funcionado es un stacking de random forest classifier + xgboost classifier (cada uno con un grid search) administrando el proceso por un pipeline

después de encontrar el mejor modelo, exportamos el código necesario para desplegar el modelo a streamlit. 
* durante el despliegue nos encontramos con que el modelo en cuestión era demasiado pesado, gracias a Git LFS. aunuqe otra alternativa hubiera sido comprimir el modelo.



