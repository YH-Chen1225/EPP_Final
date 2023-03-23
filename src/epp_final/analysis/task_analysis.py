import pytask
from epp_final.config import BLD
from epp_final.config import GROUPS
from epp_final.config import SRC

import pandas as pd
import numpy as np
#from epp_final.analysis.model import fit_logit_model
#from epp_final.analysis.model import load_model
#from epp_final.analysis.predict import predict_prob_by_age
from epp_final.utilities import read_yaml
from   scipy.stats import norm
import scipy.optimize as opt
import math
from decimal import Decimal
from IPython.display import display
from epp_final.config import bp52_aut
from epp_final.config import st_values_exp
from epp_final.config import st_values_power
from epp_final.config import stvale_spec
from epp_final.analysis.NLS import predict_table_first
from epp_final.analysis.NLS import predict_table_first_r2
from epp_final.analysis.NLS import predict_table_two
from epp_final.analysis.NLS import predict_table_three
#from epp_final.analysis.predict import predict_table_ss
#Gmm
from epp_final.analysis.mini_dist import model_esti
from epp_final.analysis.mini_dist import model_estimation
from epp_final.analysis.mini_dist import model_table
import random

#
from epp_final.data_management.clean_data import clean_mydata
from epp_final.config import genre
from epp_final.config import assum



@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)


@pytask.mark.produces(BLD / "python" / "data" / "table_1.csv")
def task_predict_parameter_r1(depends_on, produces):
    df = pd.read_pickle(depends_on["data"])
    table1 = predict_table_first(df)
    table1.to_csv(produces, index=False)


@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)

@pytask.mark.produces(BLD / "python" / "data" / "table_1_r2.csv")
def task_predict_parameter_r2(depends_on, produces):
    df = pd.read_pickle(depends_on["data"])
    table1_r2 = predict_table_first_r2(df)
    table1_r2.to_csv(produces, index=False)


@pytask.mark.depends_on(
    {
    #"data" : BLD / "python" / "data" / "data_modify.csv"
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "table_2.csv")
def task_clean_data_python2(depends_on, produces):
    df = pd.read_pickle(depends_on["data"])
    table2 = predict_table_two(df)
    table2.to_csv(produces, index=False)
    #table2.to_pickle(produces)


@pytask.mark.depends_on(
    {
    #"data" : BLD / "python" / "data" / "data_modify.csv"
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "table_3.csv")
def task_clean_data_python3(depends_on, produces):
    df = pd.read_pickle(depends_on["data"])
    table3 = predict_table_three(df)
    display(table3)
    table3.to_csv(produces, index=False)
    #table3.to_pickle(produces)

@pytask.mark.depends_on(
    {
        "data":SRC / "data" / "mturk_clean_data_short.dta"

    }
)    
@pytask.mark.produces(BLD / "python" / "data" / "table_gmm.csv")
def task_predict_model_gmm(depends_on,produces):
    random.seed(666)# set the seed for reproducible reason
    df = pd.read_stata(depends_on["data"])
    table4 = model_esti(df)
    table4.to_csv(produces, index=False)






#@pytask.mark.depends_on(
    #{
        #"scripts": ["model.py", "predict.py"],
        #"data": BLD / "python" / "data" / "data_clean.csv",
        #"data_info": SRC / "data_management" / "data_info.yaml",
    #}
#)
#@pytask.mark.produces(BLD / "python" / "models" / "model.pickle")
#def task_fit_model_python(depends_on, produces):
    #data_info = read_yaml(depends_on["data_info"])
    #data = pd.read_csv(depends_on["data"])
    #model = fit_logit_model(data, data_info, model_type="linear")
    #model.save(produces)


#for group in GROUPS:

    #kwargs = {
        #"group": group,
        #"produces": BLD / "python" / "predictions" / f"{group}.csv",
    #}

    #@pytask.mark.depends_on(
        #{
            #"data": BLD / "python" / "data" / "data_clean.csv",
            #"model": BLD / "python" / "models" / "model.pickle",
        
        #}
    #)
    #@pytask.mark.task(id=group, kwargs=kwargs)
    #def task_predict_python(depends_on, group, produces):
        #model = load_model(depends_on["model"])
        #data = pd.read_csv(depends_on["data"])
        #predicted_prob = predict_prob_by_age(data, model, group)
       #predicted_prob.to_csv(produces, index=False)

