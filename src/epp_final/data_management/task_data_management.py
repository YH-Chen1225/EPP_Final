import pandas as pd
import pytask
from epp_final.config import assume
from epp_final.config import BLD
from epp_final.config import genre
from epp_final.config import SRC
from epp_final.data_management.clean_data import clean_mydata

# import the necessary package


# @pytask.mark.depends_on(
# @pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
# def task_clean_data_python(depends_on, produces):


# Clean the dataset and store it as pickle file
# to make sure than it would not be changed when storing process.
@pytask.mark.depends_on({"data": SRC / "data" / "mturk_clean_data_short.dta"})
@pytask.mark.produces(BLD / "python" / "data" / "data_modify.pickle")
def task_clean_data_python(depends_on, produces):
    data = pd.read_stata(depends_on["data"])
    data_modify = clean_mydata(data, genre, assume)
    data_modify.to_pickle(produces)
