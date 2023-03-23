#import statsmodels.formula.api as smf
#from statsmodels.iolib.smpickle import load_pickle
#Import all the necessary package
import numpy as np
import pandas as pd
from sklearn.utils import resample
import random
from decimal import Decimal
#from IPython.display import display
import warnings
import random
from epp_final.config import mini_dist_exp_author
from epp_final.config import mini_dist_power_author


# def fit_logit_model(data, data_info, model_type):
#     """Fit a logit model to data.

#     Args:
#         data (pandas.DataFrame): The data set.
#         data_info (dict): Information on data set stored in data_info.yaml. The
#             following keys can be accessed:
#             - 'outcome': Name of dependent variable column in data
#             - 'outcome_numerical': Name to be given to the numerical version of outcome
#             - 'columns_to_drop': Names of columns that are dropped in data cleaning step
#             - 'categorical_columns': Names of columns that are converted to categorical
#             - 'column_rename_mapping': Old and new names of columns to be renamend,
#                 stored in a dictionary with design: {'old_name': 'new_name'}
#             - 'url': URL to data set
#         model_type (str): What model to build for the linear relationship of the logit
#             model. Currently implemented:
#             - 'linear': Numerical covariates enter the regression linearly, and
#             categorical covariates are expanded to dummy variables.

#     Returns:
#         statsmodels.base.model.Results: The fitted model.

#     """
#     outcome_name = data_info["outcome"]
#     outcome_name_numerical = data_info["outcome_numerical"]
#     feature_names = list(set(data.columns) - {outcome_name, outcome_name_numerical})

#     if model_type == "linear":
#         # smf.logit expects the binary outcome to be numerical
#         formula = f"{outcome_name_numerical} ~ " + " + ".join(feature_names)
#     else:
#         message = "Only 'linear' model_type is supported right now."
#         raise ValueError(message)

#     model = smf.logit(formula, data=data).fit()
#     return model


# def load_model(path):
#     """Load statsmodels model.

#     Args:
#         path (str or pathlib.Path): Path to model file.

#     Returns:
#         statsmodels.base.model.Results: The stored model.

#     """
#     model = load_pickle(path)
#     return model



def model_esti(df):
    Table5Exp,Table5Power,sd_exp,sd_power = model_estimation(df)
    table_result = model_table(Table5Power, sd_power, Table5Exp, sd_exp)
    return table_result


#Bootstraping process 
def mybootstrap(dataset, number):
    """
        args:
            dataset
            number:The number of times to conducting bootstrap
    """   
    E11, E12, E13, E31, E32, E10, E41, E42 = [], [], [], [], [], [], [], []
    box={'1.1':E11,'1.2':E12,'1.3':E13,'3.1':E31,'3.2':E32,'10':E10,'4.1':E41,'4.2':E42}
        
    for i in range(1,number+1):
        for a,b in box.items():
            db = dataset[dataset.treatment==a]
            bootsample = resample(db['buttonpresses'],replace=True,)
            b.append(np.round(np.mean(bootsample)))
    return E11, E12, E13, E31, E32, E10, E41, E42



#Using the formula to estimate each parameters for both exp cost function and power cost function
def mymindisest(params):
        '''
        params: params is the dictionary can be the mean effort value for each treatment 
                or the original bootstrap outcome(Resampling each treatment, calculate the mean, do it for 2000 times, so that the outcome would be a 2000 obs array).
                Besides, this function also include a key called "specification", we can choose "Exp" or "Power" as our cost function.
        '''
        # Define constants payoff p
        P = [0, 0.01, 0.1] # P is a vector containing the different piece-rates
        expr = {
            'Exp': {
                'log_k': lambda E11, E12: (np.log(P[2]) - np.log(P[1]) * E12 / E11) / (1 - E12 / E11),
                'log_gamma': lambda log_k, E11: np.log((np.log(P[1]) - log_k) / E11),
                'log_s': lambda log_gamma, E13, log_k: np.exp(log_gamma) * E13 + log_k,
                'EG31': lambda E31, g: np.exp(E31 * g),'EG32': lambda E32, g: np.exp(E32 * g),'EG10': lambda E10, g: np.exp(E10 * g),'EG41': lambda E41, g: np.exp(E41 * g),'EG42': lambda E42, g: np.exp(E42 * g),
            


            },
            'Power': {
                'log_k': lambda E11, E12: (np.log(P[2]) - np.log(P[1]) * np.log(E12) / np.log(E11)) / (1 - np.log(E12) / np.log(E11)),
                'log_gamma': lambda log_k, E11: np.log((np.log(P[1]) - log_k) / np.log(E11)),
                'log_s': lambda log_gamma, E13, log_k: np.exp(log_gamma) * np.log(E13) + log_k,
                'EG31': lambda E31, g: E31 ** g,'EG32': lambda E32, g: E32 ** g,'EG10': lambda E10, g: E10 ** g,'EG41': lambda E41, g: E41 ** g,'EG42': lambda E42, g: E42 ** g
            }
        }

        # Extract arguments from params dictionary
        E11 = np.array(params['E11'])
        E12 = np.array(params['E12'])
        E13 = np.array(params['E13'])
        E31 = np.array(params['E31'])
        E32 = np.array(params['E32'])
        E10 = np.array(params['E10'])
        E41 = np.array(params['E41'])
        E42 = np.array(params['E42'])
        specification = params['specification']

        # Calculate k, gamma, alpha, a, s_ge, delta, beta
        log_k = expr[specification]['log_k'](E11, E12)
        log_gamma = expr[specification]['log_gamma'](log_k, E11)
        log_s = expr[specification]['log_s'](log_gamma, E13, log_k)
        k = np.exp(log_k)
        g = np.exp(log_gamma)
        s = np.exp(log_s)
        EG31 = expr[specification]['EG31'](E31,g)
        EG32 = expr[specification]['EG32'](E32,g)
        EG10 = expr[specification]['EG10'](E10,g)
        EG41 = expr[specification]['EG41'](E41,g)
        EG42 = expr[specification]['EG42'](E42,g)
        alpha = 100/9*k*(EG32-EG31)
        a = 100*k*EG31-100*s-alpha
        s_ge = k*EG10 - s
        delta = np.sqrt((k*EG42-s)/(k*EG41-s))
        beta  = 100*(k*EG41-s)/(delta**2)
      
      
        return k, g, s, alpha, a, s_ge, beta, delta


    #Estimate the result
    ##Table 5 minimum distance estimates: columns (1) (3) panel A and columns (1) (4) panel B
def model_estimation(df):
     emp_moments = np.array(np.round(df.groupby("treatment").mean("buttonpresses")))
     E11, E12, E13, E31, E32, E10, E41, E42 = mybootstrap(df,2000)
     
     #Now create a params with the mean effort for all treaments.
     params = {'E11':emp_moments[0],'E12':emp_moments[1],'E13':emp_moments[2],'E31':emp_moments[6],'E32':emp_moments[7],
            "E10":emp_moments[4],'E41':emp_moments[8],'E42':emp_moments[9],'specification':'Exp'}
     Table5Exp = np.array(mymindisest(params)).flatten()
     params['specification'] = 'Power'
     Table5Power = np.array(mymindisest(params)).flatten()
     vmindisest = np.vectorize(mymindisest)
     params['specification'] = "Exp"

     #Define the new dictionary, now ,for each treatment, there is a array that length equal to 2000, which we obtain from boostrap process.
     nw_params = {"E11":E11,"E12":E12,"E13":E13,"E31":E31,"E32":E32,"E10":E10,"E41":E41,"E42":E42,"specification":"Exp"}
    #Obtain the estimation result
    #Function for compute the mean and standard error of estimates for the cost function using the Bootstrap procedure
     def cal_res(params,type):
        
        '''
            Args:
            params : params is a dictionary, which include each bootstraping result for each parameters
            type : Type can be Exp or Power, depends on whcih cost function you want to apply.'''
        if type == "Exp":
            params['specification'] = "Exp"
            estimates = vmindisest(params)
        else:
            params['specification'] = "Power"
            estimates = vmindisest(params)
        res = np.zeros((8,2))
        #Calculating the mean and the std for each treatment
        for i in range(0,8):
            res[i,0],res[i,1] = np.mean(estimates[i][~np.isnan(estimates[i])&~np.isinf(estimates[i])]), np.std(estimates[i][~np.isnan(estimates[i])&~np.isinf(estimates[i])])
        return res
        
     warnings.filterwarnings('ignore') # This is to avoid showing RuntimeWarning in the notebook regarding overflow. For a couple of cases in our 1000 new samples
                                  # we cannot find the results because of overflow. Losing 2-3 observations out of thousands should not change the overall mean
                                  # for the parameters
     # Store mean and standard error of estimates for the cost function using the Bootstrap procedure
     exp_res = cal_res(nw_params,type="Exp")
     power_res = cal_res(nw_params,type="Power")
     sd_exp   = exp_res[0:8,1]
     sd_power = power_res[0:8,1] 
     return Table5Exp,Table5Power,sd_exp,sd_power

def model_table(Table5Power, sd_power, Table5Exp, sd_exp):
    '''
        Args:
            Table5Power: The power cost function parameters for minimum distance method
            sd_power: The standard error of power cost function parameters for minimum distance method
            Table5Exp: The exponential cost function parameters for minimum distance method
            sd_exp: The standard error of exponential cost function parameters for minimum distance method

    '''

    params_name = ["Level k of cost of effort", "Curvature γ of cost function","Intrinsic motivation s","Social preferences α",
                "Warm glow coefficient a","Gift exchange Δs", "Present bias β","(Weekly) discount factor δ"]
    columns = [Table5Power, sd_power, Table5Exp, sd_exp]
    vs = []
    for col in columns:
        col = ['{0:.2e}'.format(Decimal(col[0])), round(col[1],3), '{0:.2e}'.format(Decimal(col[2])),
            round(col[3],3), round(col[4],3), '{0:.2e}'.format(Decimal(col[5])), round(col[6],2), round(col[7],2)]
        vs.append(col)
        
    Table5Results = pd.DataFrame({'Parameters name': params_name,
                                'Minimum dist est on average effort power cost function estimates': vs[0],
                                'Minimum dist est on average effort power cost function from authors': mini_dist_power_author,
                                'Minimum dist est on average effort power standard errors': vs[1],
                                'Minimum dist est on average effort exp cost function estimates': vs[2],
                                'Minimum dist est on average effort exp cost function from authors': mini_dist_exp_author,
                                'Minimum dist est on average effort exp standard errors': vs[3]})
    return Table5Results 
