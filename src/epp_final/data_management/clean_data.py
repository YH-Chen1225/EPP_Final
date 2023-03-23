import pandas as pd
import numpy as np


# def clean_data(data, data_info):
  
#     data = data.drop(columns=data_info["columns_to_drop"])
#     data = data.dropna()
#     for cat_col in data_info["categorical_columns"]:
#         data[cat_col] = data[cat_col].astype("category")
#     data = data.rename(columns=data_info["column_rename_mapping"])

#     numerical_outcome = pd.Categorical(data[data_info["outcome"]]).codes
#     data[data_info["outcome_numerical"]] = numerical_outcome
#     return data


#Cleaning Dataset before doing any analysis, in this process, many variable would be added into the dataset.
def clean_mydata(df,genre,assum):
    ''' Create the comprehensive dataset for analysis
          Args:
            df:The dataset,following variable are includes in the dataset
                buttonpresses:How much times does participants press the buttoms on the keyboard. Usually called "effort" in my code.
                treatment: There are different treatment would be conducted to see which treament really motivates participants to making effor.
                           Treatment variable in the dataset is a number represent each treatment.
                treatment name: The name of each treatment
            genre: The list, include many treatment,  I used for creating the variable easily.
            assum: The dict, I used for assign the payoff(piece rate) for each treatment.

                '''
    # Creating several varaiables in the dataset for estimation, same as original code did, but in a efficient way.
    for i in genre:
        if i != 'prob':
            df[F"{i}"] = 0
        else:
            df[F"{i}"] = 1
        
    # Create piece-rate payoffs per 100 button presses (p)
    # create payoff per 100 to charity and dummy charity (alpha/a) 
    # create payoff per 100 delayed by 2 weeks and dummy delay
    # probability weights to back out curvature and dummy for gift exchange    
    for task,payoff in assum.items():
        for key in payoff:
            df.loc[df.treatment == key, task] = payoff[key]
    
    # Generating effort and log effort. authors round buttonpressed to nearest 100 value. If 0 set it to 25.
    df['buttonpresses'] += 0.1 # python rounds 50 to 0, while stata to 100. by adding a small value we avoid this mismatch
    df['buttonpresses_nearest_100'] = round(df['buttonpresses'],-2)
    df.loc[df.buttonpresses_nearest_100 == 0, 'buttonpresses_nearest_100'] = 25
    df['logbuttonpresses_nearest_100']  = np.log(df['buttonpresses_nearest_100'])

    # Create dummies for thses specification(who receive these treatment)
    df['dummy1'] = (df['treatment'].isin(['1.1', '1.2','1.3'])).astype(int)# isin can be applied in pandas dataframe
    df['samplenw'] = (df['treatment'].isin(['1.1','1.2','1.3','3.1','3.2','4.1','4.2','10'])).astype(int)
    df['samplepr'] = (df['treatment'].isin(['1.1','1.2','1.3','6.1','6.2'])).astype(int)

    return df
