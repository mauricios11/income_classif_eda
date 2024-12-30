#
# current path: ./utils/utils_classif_models_plots.py


#libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Tuple
from sklearn.tree import plot_tree
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score)



class ClassifModelsPlots:
    def __init__(self) -> None:
        pass
    
    #-func 1-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    """barplot of the feature importances
        - need to calculate the feature importances before apply this function"""
    def plot_feature_importances(self, data,
                                 series: bool = False,
                                 cols: List[str]= None,
                                 x_col: str= 'feature',
                                 y_col: str= 'importance',
                                 figsize: Tuple[int,int]=(15,8),
                                 importance_type: str= 'tree',
                                 palette = 'viridis'):
        
        plt.figure(figsize= figsize)
        if series:
            sns.barplot(x= cols, y= data,
                        hue= cols,
                        dodge= False,
                        palette= palette, edgecolor= 'gray')
            
        else:
            sns.barplot(x= data[x_col],
                        y= data[y_col],
                        hue= data[x_col],
                        dodge= False,
                        palette= palette, edgecolor= 'gray')
        
        
        plt.title(f'Feature importances ({importance_type})', fontsize= 15,
                   weight= 'bold', loc= 'left')
        
        plt.xlabel('Features', fontsize= 12, weight= 'bold')
        plt.ylabel('Importance', fontsize= 12, weight= 'bold')
        plt.show()