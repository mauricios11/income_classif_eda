# Module focused on probability functions (such as PMF, CDF, PDF)
# current path: utils/utils_probability.py

# libraries
import empiricaldist as ed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

# own modules

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

class ProbabilityFuncs:
    def __init__(self) -> None:
        
        #self.df_obj = df 
        pass
    
#-Func 01-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #(1) hacer una función que calcule un pmf de una columna de un df
    """ docstring"""

    def pmf_calc(self,df: pd.DataFrame,
                 xcol: str,
                 use_seaborn: bool = False,
                 hue: str = None,
                 sns_stat = 'probability',
                 ed_normalize = True,
                 ed_probability_calc: int= None,
                 ed_color: str = None):
        
        #df = self.df_obj
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("ERROR: df param must be a DataFrame")
        if not isinstance(xcol, str) and xcol in df.columns:
            raise ValueError("ERROR: xcol param must string and must be a column of df")
        if ed_probability_calc  and not isinstance(ed_probability_calc, int):
            raise ValueError("ERROR: ed_probability_calc param must be an integer")
        
        if use_seaborn:
            if not hue:
                sns.histplot(data= df, x= xcol,
                             kde= True,
                             binwidth= 1,
                             stat= sns_stat)
            else:
                sns.histplot(data= df, x= xcol,
                             hue= hue, kde= True,
                             binwidth= 1,
                             stat= sns_stat)
            
            plt.title(f'PMF (Probability Mass Function)\n column: {xcol}',
                      fontsize= 14, weight= 'bold')
            
        if not use_seaborn and ed_probability_calc:
        # el gráfico de barras, debería tener un  rango de entre 0 y 1

            pmf_df = ed.Pmf.from_seq(df[xcol], normalize= ed_normalize)
            probability = pmf_df(ed_probability_calc)
            
            pmf_df.bar(label= 'real data', color= ed_color, alpha= 0.9)  # real data
            
            plt.scatter(x= ed_probability_calc,                          # value
                        y= probability,
                        color= 'turquoise',  
                        label= 'calc value', zorder= 4)
            
            plt.ylabel('Probability'), plt.xlabel('Values')              # varify label names
            
            plt.hlines(y = probability,
                       xmin= pmf_df.qs[0],
                       xmax= ed_probability_calc,
                       color= 'orange',
                       linestyle= '--', alpha= 0.9)
            
            plt.vlines(x = ed_probability_calc,
                       ymin= 0,
                       ymax= probability,
                       color = 'orange',
                       linestyle= '--', alpha= 0.9)
            
            plt.title(f'PMF (Probability Mass Function)\ncolumn: {xcol} for value: {ed_probability_calc}',
                      fontsize= 14, weight= 'bold')
            plt.legend()
            plt.show()
            print (f'PMF (Probability Mass Function)\n - The probability to find values == {ed_probability_calc} in "{xcol}" is:\n {round(probability *100, 4)}%')

#-Func 02-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #(2)cdf + iqr
    """ docstring """
    def cdf_calc(self, df: pd.DataFrame, col: str, quantity: int, iqr: bool= False):
        
        #df = self.df_obj
        if not isinstance(df, pd.DataFrame) or not isinstance(col, str) or not isinstance(quantity, int):
            raise TypeError('ERROR: The args must be:\n- df: pd.DataFrame\n- col: str\n- quantity: int')
        
        cdf_df = ed.Cdf.from_seq(df[col], normalize= True)
        if not iqr:
            cdf_df.plot(marker= 'o')
        
        probability = cdf_df.forward(quantity)
        # Plotting:
        plt.plot(quantity, probability,
                 marker= 'o',
                 color= 'orange',
                 zorder= 4, label= f'calc value: {quantity}')
        
        plt.vlines(x= quantity,
                   ymin= 0,
                   ymax= probability,
                   color= 'green', linestyle= '--')
        
        plt.hlines(y= probability,
                   xmin= cdf_df.qs[0],
                   xmax= quantity,
                   color= 'green', linestyle= '--')
        
        if iqr:
            cdf_df.step(marker= 'o', alpha= 0.7, label= 'other values')
            prob_1 = 0.25
            prob_2 = 0.75
            ps = (prob_1, prob_2)
            qs = cdf_df.inverse(ps) 
            # inverse(): devuelve el valor de la variable aleatoria que corresponde a una probabilidad dada
            
            plt.vlines(x= qs,
                       ymin= 0, ymax= ps,
                       color= 'magenta', linestyle= ':')
            
            plt.hlines(y= ps,
                       xmin= cdf_df.qs[0], xmax= qs,
                       color= 'magenta', linestyle= ':')
            
            plt.scatter(x= qs, y= ps, color= 'magenta', zorder= 3, label= 'IQR')
            
            plt.ylabel('Probability')
            plt.legend()
            plt.title(f'CDF (Cumulative Distribution Func)\n"{col}" for value: {quantity} and IQR', fontsize= 14,
                      weight= 'bold')
            plt.show()
            
        #comprobar si el resultado está dentro de iqr
        if iqr:
            if quantity >= qs[0] and quantity <= qs[1]:
                print(f'The probability result for the value "{quantity}" is inside the IQR')
            else:
                if quantity < qs[0]:
                    not_iqr_result= '(under q1)'
                if quantity > qs[1]:
                    not_iqr_result= '(over q3)'
                print(F'The value "{quantity}" is an OUTLIER: {not_iqr_result}')
                
        print(f'Result: Probability to find a value {quantity} or less: {round(probability *100, 3)}%')
        

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #(3) observar qué tanto se ajustan los datos reales a una distribución normal
    """
    """

    def cdf_pdf(self, df: pd.DataFrame, col: str, value: float = None,
                main_color: str = 'orange',
                show_index: bool = True,
                pdf_bandwidth: float = 0.08):
        #df = self.df_obj
        
        if col not in df.columns:
            raise ValueError(f"La columna {col} no se encuentra en el DataFrame")
        if not df[col].dtype in [np.float64, np.int64]:
            raise ValueError(f"La columna {col} no es numérica")
        
        stats= df[col].describe()
        xs= np.linspace(stats['min'], stats['max'])
        
        #plot settings
        size = 14 #            <- labels an title size
        fig, axes= plt.subplots(1,2 , figsize= (25,6))
        
        # CDF
        ys_cdf= scipy.stats.norm(stats['mean'], stats['std']).cdf(xs)

        if value is not None: 
            closest_index= np.abs(xs - value).argmin()
                # encontrar el valor más cercano a 'value'
                # np.abs ->  distancia entre dos números sin considerar la dirección
            prob_value= ys_cdf[closest_index]
            if show_index:
                print(f'Index for the value {round(value, 4)} is: {closest_index}') 
        
        #PDF
        ys_pdf= scipy.stats.norm(stats['mean'],stats['std']).pdf(xs)
        
        # CDF plot
        text_result =f'\n\nThe probability to obtain {value} or less\nis: {round(prob_value,4) *100}%'
        text_cdf = f'CDF vs real data - column: {col}'    
        cdf_calc = ed.Cdf.from_seq(df[col], normalize= True)
        
        axes[0].plot(cdf_calc, 
                     marker= 'o',
                     label= 'real data',                    # real data
                     alpha= 0.5, zorder= 1)
             
        axes[0].plot(xs, ys_cdf,
                     color = main_color,                    # CDF probability  
                     marker= 'o', linewidth= 1.9,
                     zorder= 2, label= 'CDF probability')
             
        axes[0].scatter(value, prob_value,                   # graficar value
                        color= 'purple', marker= 'D',
                        s= 60, zorder= 4, label= f'Value: {value}') 
        
        axes[0].vlines(value,
                       ymin= 0, ymax= prob_value,
                       color= 'purple',linestyle= '--', alpha= 0.5)
        
        axes[0].hlines(prob_value,
                       xmin= xs[0], xmax= value,
                       color= 'purple', linestyle= '--', alpha= 0.5)
        
        axes[0].set_xlabel('Values', size= size)
        axes[0].set_ylabel('Probability', size= size)
        axes[0].set_title(f'{text_cdf}{text_result}', size= size + 2, weight= 'bold')
        axes[0].legend()
        
        # PDF plot
        text_pdf = f'PDF vs KDE - column: {col}'
        
        axes[1].plot(xs, ys_pdf,
                     color= main_color, marker= 'o',
                     linewidth= 1.9, zorder= 2, label= 'PDF probability')
        
        sns.kdeplot(data= df, x= col,
                    marker= '^',
                    color = '#005fb9',
                    bw_method= pdf_bandwidth,
                    zorder= 1, alpha= 0.3, label= 'real data', ax= axes[1])
        
        if value is not None:
            xs_pdf_value = scipy.stats.norm(stats['mean'], stats['std']).pdf(value)
            
            axes[1].scatter(value, xs_pdf_value,         # graficar value
                            marker= 'D', color= 'purple',
                            s= 60, zorder= 4,    
                            label= f'Value: {value}') 
            
            axes[1].vlines(x= value,
                           ymin= 0, ymax= xs_pdf_value,
                           color= 'purple', linestyle= '--', alpha = 0.5)
            
        axes[1].set_xlabel('Values', size= size)
        axes[1].set_ylabel('Density', size= size)
        axes[1].set_title(f'{text_pdf}{text_result}', weight= 'bold', size= size + 2)
        axes[1].legend()
        
        plt.show()
 
#-Func -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # (4) teoría del límite central: calcular la probabilidad de obtener un valor (PMF)
    #dado que esta columna es categórica y  contiene datos binomiales, (dos valores posibles) deberíamos pasarlos a números
    #reemplazar los valores de la columna "sex" por 0 y 1
    
    def pmf_categorical_col(self, df: pd.DataFrame, categorical_col: str,
                            sample_size: int= 35,
                            number_samples: int= 1000,
                            color: str | dict = None):
        #df = self.df_obj
        # Validations
        if not isinstance(df, pd.DataFrame):
            raise TypeError('ERROR: df param must be a DF')
        if not isinstance(categorical_col, str) or categorical_col not in df.columns:
            if df[categorical_col].dtype not in ['object', 'category']:
                raise ValueError('ERROR: categorical_col must be a column of -> df')
            else:
                raise ValueError('ERROR: categolical_col param must be:\n- a string\n- a column of df')
        if not isinstance(sample_size, int) or sample_size <= 0 or sample_size > len(df):
            raise ValueError('ERROR: sample_size must be an integer > 0 and < len(df)')
        if number_samples <= 0:
            raise ValueError('ERROR: number_samples must be an integer > 0')
        
        # One-hot encoding
        one_hot_encoded = pd.get_dummies(df[categorical_col])       
        
        fig, axes= plt.subplots(one_hot_encoded.shape[1], 2,                        # row number == amount of cols in one_hot
                                figsize= (25, 6 * one_hot_encoded.shape[1]))
        
        for index, category in enumerate(one_hot_encoded.columns):
            # vals for the current col -> (1 for current, 0 for the rest)
            category_values = one_hot_encoded[category]
            
            samples_list = []
            np.random.seed(42)
            
            for i in range(1, number_samples +1):
                category_sample = category_values.sample(sample_size,
                                                         replace= True).to_numpy()   # sample size
                samples_list.append(
                                pd.Series(category_sample,
                                          name= f'sample_{i}'))                      # store each sample in Series
            
            df_samples = pd.concat(samples_list, axis= 1)                            # concat all samples in a new df (col -> sample_i)
            category_mean = df_samples.mean().mean()
            
            print(f'Estimated percentage for: {category} (in total population): {category_mean * 100:.2f}%')
            
            # plot: KDE for samples (mean)
            samples_means = pd.DataFrame(df_samples.mean(), columns= ['mean'])
            
            sns.kdeplot(data= samples_means,
                        x= 'mean',
                        fill= True,
                        linewidth= 2.5,
                        label= f'PMF: sample mean for {category}',
                        ax= axes[index, 0]) # axes[index, 0]¿?
            
            axes[index , 0].axvline(x= category_values.mean(),
                                    color= 'orange', linestyle= '--',
                                    linewidth= 1.9, label= f'{category} mean (real data)')
            axes[index , 0].legend()
            axes[index , 0].set_title(f'Sampling Distribution of the Mean - PMF\ncolumn: {categorical_col}, value: {category}', size= 17)
            
            # cumulative mean estimation (diferent sample sizes)
            estimaded_means = [df_samples.iloc[:, : i].mean().mean() for i in range(1, number_samples +1)]
            
            sample_size_experiment = pd.DataFrame({'sample size': range(1, number_samples +1), 'estimated mean': estimaded_means})
            
            # plot: relationship between sample size vs estimated mean
            sns.scatterplot(data= sample_size_experiment,
                            x= 'sample size',
                            y= 'estimated mean',
                            alpha= 0.3,
                            ax= axes[index, 1])
            
            axes[index, 1].axhline(y= category_values.mean(),
                                   color= 'red', linestyle= '--',
                                   linewidth= 1.9, label= f'{category} mean')
            
            axes[index, 1].set_title(f'Sample size vs Estimated Mean\ncolumn: {categorical_col}, value: {category}', size= 17)
            axes[index, 1].legend()
            
        plt.tight_layout()
        plt.show()
        # modificación: hacer que el plot del scatter plot tenga un mayor espacio y los puntos estén más alejados de las orillas del eje y 