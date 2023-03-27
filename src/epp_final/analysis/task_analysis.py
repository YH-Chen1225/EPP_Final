import random

import pandas as pd
import pytask
from epp_final.analysis.mini_dist import model_esti
from epp_final.analysis.NLS import predict_table_first
from epp_final.analysis.NLS import predict_table_first_r2
from epp_final.analysis.NLS import predict_table_three
from epp_final.analysis.NLS import predict_table_two
from epp_final.config import BLD
from epp_final.config import SRC
from IPython.display import display

# Gmm
#


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.produces(BLD / "python" / "data" / "table_1.csv")
def task_predict_parameter_r1(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    table1 = predict_table_first(data)
    table1.to_csv(produces, index=False)


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.produces(BLD / "python" / "data" / "table_1_r2.csv")
def task_predict_parameter_r2(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    table1_r2 = predict_table_first_r2(data)
    table1_r2.to_csv(produces, index=False)


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.produces(BLD / "python" / "data" / "table_2.csv")
def task_clean_data_python2(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    table2 = predict_table_two(data)
    table2.to_csv(produces, index=False)


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.produces(BLD / "python" / "data" / "table_3.csv")
def task_clean_data_python3(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    table3 = predict_table_three(data)
    display(table3)
    table3.to_csv(produces, index=False)


@pytask.mark.depends_on({"data": SRC / "data" / "mturk_clean_data_short.dta"})
@pytask.mark.produces(BLD / "python" / "data" / "table_gmm.csv")
def task_predict_model_gmm(depends_on, produces):
    random.seed(666)  # set the seed for reproducible reason
    data = pd.read_stata(depends_on["data"])
    table4 = model_esti(data)
    table4.to_csv(produces, index=False)


# @pytask.mark.depends_on(
# @pytask.mark.produces(BLD / "python" / "models" / "model.pickle")
# def task_fit_model_python(depends_on, produces):


# for group in GROUPS:


# @pytask.mark.depends_on(

# @pytask.mark.task(id=group, kwargs=kwargs)
# def task_predict_python(depends_on, group, produces):
