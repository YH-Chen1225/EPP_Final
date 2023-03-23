import pytask
from epp_final.config import BLD
from epp_final.config import SRC

import pandas as pd
#from epp_final.data_management import clean_data
from epp_final.data_management.clean_data import clean_mydata
from epp_final.utilities import read_yaml
from epp_final.config import genre
from epp_final.config import assum
#import the necessary package
import numpy as np
import pandas as pd


# @pytask.mark.depends_on(
#     {
#         "scripts": ["clean_data.py"],
#         "data_info": SRC / "data_management" / "data_info.yaml",
#         "data": SRC / "data" / "data.csv",
#     }
# )
# @pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
# def task_clean_data_python(depends_on, produces):
#     data_info = read_yaml(depends_on["data_info"])
#     data = pd.read_csv(depends_on["data"])
#     data = clean_data(data, data_info)
#     data.to_csv(produces, index=False)


#Clean the dataset and store it as pickle file to make sure than it would not be changed when storing process.
@pytask.mark.depends_on(
    {
        "data":SRC / "data" / "mturk_clean_data_short.dta"
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "data_modify.pickle")
def task_clean_data_python(depends_on, produces):
    df = pd.read_stata(depends_on["data"])
    data_modify = clean_mydata(df,genre,assum)
    data_modify.to_pickle(produces)




