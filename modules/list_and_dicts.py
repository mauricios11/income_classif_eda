# module to store lists and dictionaries for the data cleaning process
# path: ./modules/list_and_dicts.py

class ListAndDicts:
    def __init__(self): 
        self.initial_name_cols = ['age', 'workclass', 'fnlwgt', 'education',
                                  'education-num', 'marital-status', 'occupation',
                                  'relationship', 'race', 'sex', 'capital-gain',
                                  'capital-loss', 'hours-per-week', 'native-country',
                                  'income']
        
        self.replace_edu_names = {'Preschool':   'preschool',
                                  'Bachelors':   'bachelors',
                                  'HS-grad':     'hs_grad',         
                                  'Masters':     'masters',
                                  'Some-college':'some_college',
                                  'Assoc-acdm':  'assoc_acdm',
                                  'Assoc-voc':   'assoc_voc',
                                  'Doctorate':   'doctorate',
                                  'Prof-school': 'prof_school',
                                  'Preschool':   'preschool'
                                  } 
                                    
        self.replace_country_names = {'?':'unknown',
                                      'United-States':      'USA',
                                      'Outlying-US(Guam-USVI-etc)': 'Outlying_US',
                                      'Trinadad&Tobago':    'Trinidad_Tobago',
                                      'Holand-Netherlands': 'Netherlands',
                                      'El-Salvador':        'Salvador',
                                      'Dominican-Republic': 'Dominican_Rep',
                                      'Puerto-Rico':        'Puerto_Rico',
                                      'Hong':               'Hong_Kong'
                                      }
        
        self.replace_workclass_names = {'State-gov':        'state_gov',
                                        'Self-emp-not-inc': 'self_emp_not_inc',
                                        'Federal-gov':      'federal_gov',
                                        'Local-gov':        'local_gov',
                                        '?':                'unknown',
                                        'Self-emp-inc':     'self_emp_inc',
                                        'Without-pay':      'without_pay',
                                        'Never-worked':     'never_worked'
                                        }
        
        self.replace_marital_stat_names = {'Never-married':        'never_married',
                                          'Married-civ-spouse':    'married_civ_spouse',
                                          'Divorced':              'divorced',
                                          'Married-spouse-absent': 'married_spouse_absent',
                                          'Separated':             'separated',
                                          'Married-AF-spouse':     'married_AF_spouse',
                                          'Widowed':               'widowed'
                                         }
        
        self.replace_occupation_names = {'Adm-clerical':       'adm_clerical',
                                         'Exec-managerial':    'exec_managerial',
                                         'Handlers-cleaners':  'handlers_cleaners',
                                         'Prof-specialty':     'prof_specialty',
                                         'Other-service':      'other_service',
                                         'Sales':              'sales',
                                         'Craft-repair':       'craft_repair',
                                         'Transport-moving':   'transport_moving',
                                         'Farming-fishing':    'farming_fishing',
                                         'Machine-op-inspct':  'machine_op_inspct',
                                         'Tech-support':       'tech_support',
                                         '?':                  'unknown', 
                                         'Protective-serv':    'protective_serv',
                                         'Armed-Forces':       'armed_forces',
                                         'Priv-house-serv':    'priv_house_serv'
                                         }
        
        self.replace_family_names = {'Not-in-family':     'not_in_family',
                                     'Husband':           'husband',
                                     'Wife':              'wife',
                                     'Own-child':         'own_child',
                                     'Unmarried':         'unmarried',
                                     'Other-relative':    'other_relative'
                                     }
        
        self.replace_ethnicity_names = {'White':               'white',
                                        'Black':               'black',
                                        'Asian-Pac-Islander':  'asian_pac_islander',
                                        'Amer-Indian-Eskimo':  'amer_indian_eskimo',
                                        'Other':               'other'
                                        }
     
        self.replace_iter_cols = {'native_country': self.replace_country_names,
                                  'education':      self.replace_edu_names,
                                  'workclass':      self.replace_workclass_names,
                                  'marital_status': self.replace_marital_stat_names,
                                  'occupation':     self.replace_occupation_names,
                                  'relationship':   self.replace_family_names ,
                                  'ethnicity':      self.replace_ethnicity_names}

        



