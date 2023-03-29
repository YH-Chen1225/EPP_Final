# Import all the necessary package
import operator
import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
from epp_final.config import mini_dist_exp_author
from epp_final.config import mini_dist_power_author
from sklearn.utils import resample


def model_esti(df):
    table5exp, table5power, sd_exp, sd_power = model_estimation(df)
    table_result = model_table(table5power, sd_power, table5exp, sd_exp)
    return table_result


# Bootstraping process
def mybootstrap(dataset, number):
    """Creating bootstrap function.

    args:
        dataset
        number:The number of times to conducting bootstrap

    """
    if number <0:
        raise ValueError("The number of boostraping times should not be smaller than zero")

    e11, e12, e13, e31, e32, e10, e41, e42 = [], [], [], [], [], [], [], []
    box = {
        "1.1": e11,
        "1.2": e12,
        "1.3": e13,
        "3.1": e31,
        "3.2": e32,
        "10": e10,
        "4.1": e41,
        "4.2": e42,
    }

    for _i in range(1, number + 1):
        for a, b in box.items():
            db = dataset[dataset.treatment == a]
            bootsample = resample(
                db["buttonpresses"],
                replace=True,
            )
            b.append(np.round(np.mean(bootsample)))
    return e11, e12, e13, e31, e32, e10, e41, e42


def mymindisest(params):
    """Estimating each parameters for cost function.

    Args:
            params: params is the dictionary
            which can be the mean effort value for each treatment
            or the original bootstrap outcome
            (Resampling each treatment,
            calculate the mean, do it for 2000 times,
            so that the outcome would be a 2000 obs array).
            Besides, this function also include a key called "specification",
            we can choose "Exp" or "Power" as our cost function.

    """
    # Define constants payoff p
    p = [0, 0.01, 0.1]  # P is a vector containing the different piece-rates
    expr = {
        "Exp": {
            "log_k": lambda e11, e12: (np.log(p[2]) - np.log(p[1]) * e12 / e11)
            / (1 - e12 / e11),
            "log_gamma": lambda log_k, e11: np.log((np.log(p[1]) - log_k) / e11),
            "log_s": lambda log_gamma, e13, log_k: np.exp(log_gamma) * e13 + log_k,
            "EG31": lambda e31, g: np.exp(e31 * g),
            "EG32": lambda e32, g: np.exp(e32 * g),
            "EG10": lambda e10, g: np.exp(e10 * g),
            "EG41": lambda e41, g: np.exp(e41 * g),
            "EG42": lambda e42, g: np.exp(e42 * g),
        },
        "Power": {
            "log_k": lambda e11, e12: (
                np.log(p[2]) - np.log(p[1]) * np.log(e12) / np.log(e11)
            )
            / (1 - np.log(e12) / np.log(e11)),
            "log_gamma": lambda log_k, e11: np.log(
                (np.log(p[1]) - log_k) / np.log(e11)
            ),
            "log_s": lambda log_gamma, e13, log_k: np.exp(log_gamma) * np.log(e13)
            + log_k,
            "EG31": lambda e31, g: operator.pow(e31, g),  # lambda e31, g: e31**g,
            "EG32": lambda e32, g: operator.pow(e32, g),  # lambda e32, g: e32**g,
            "EG10": lambda e10, g: operator.pow(e10, g),  # lambda e10, g: e10**g,
            "EG41": lambda e41, g: operator.pow(e41, g),  # lambda e41, g: e41**g,
            "EG42": lambda e42, g: operator.pow(e42, g),  # lambda e42, g: e42**g,
        },
    }

    # Extract arguments from params dictionary
    e11 = np.array(params["E11"])
    e12 = np.array(params["E12"])
    e13 = np.array(params["E13"])
    e31 = np.array(params["E31"])
    e32 = np.array(params["E32"])
    e10 = np.array(params["E10"])
    e41 = np.array(params["E41"])
    e42 = np.array(params["E42"])
    specification = params["specification"]

    # Calculate k, gamma, alpha, a, s_ge, delta, beta
    log_k = expr[specification]["log_k"](e11, e12)
    log_gamma = expr[specification]["log_gamma"](log_k, e11)
    log_s = expr[specification]["log_s"](log_gamma, e13, log_k)
    k = np.exp(log_k)
    g = np.exp(log_gamma)
    s = np.exp(log_s)
    eg31 = expr[specification]["EG31"](e31, g)
    eg32 = expr[specification]["EG32"](e32, g)
    eg10 = expr[specification]["EG10"](e10, g)
    eg41 = expr[specification]["EG41"](e41, g)
    eg42 = expr[specification]["EG42"](e42, g)
    alpha = 100 / 9 * k * (eg32 - eg31)
    a = 100 * k * eg31 - 100 * s - alpha
    s_ge = k * eg10 - s
    delta = np.sqrt((k * eg42 - s) / (k * eg41 - s))
    beta = 100 * (k * eg41 - s) / (delta**2)

    return k, g, s, alpha, a, s_ge, beta, delta


# Estimate the result
def model_estimation(df):
    emp_moments = np.array(np.round(df.groupby("treatment").mean("buttonpresses")))
    e11, e12, e13, e31, e32, e10, e41, e42 = mybootstrap(df, 2000)

    # Now create a params with the mean effort for all treaments.
    params = {
        "E11": emp_moments[0],
        "E12": emp_moments[1],
        "E13": emp_moments[2],
        "E31": emp_moments[6],
        "E32": emp_moments[7],
        "E10": emp_moments[4],
        "E41": emp_moments[8],
        "E42": emp_moments[9],
        "specification": "Exp",
    }
    table5exp = np.array(mymindisest(params)).flatten()
    params["specification"] = "Power"
    table5power = np.array(mymindisest(params)).flatten()
    vmindisest = np.vectorize(mymindisest)
    params["specification"] = "Exp"

    # Define the new dictionary, for each treatment,
    # there is a array that length equal to 2000,
    # which we obtain from bootstrap process.
    nw_params = {
        "E11": e11,
        "E12": e12,
        "E13": e13,
        "E31": e31,
        "E32": e32,
        "E10": e10,
        "E41": e41,
        "E42": e42,
        "specification": "Exp",
    }
    # Obtain the estimation result
    def cal_res(params, func_type) -> float:
        """Computing mean and standard error of parameters from Bootstrap.

        Args:
            params : params is a dictionary,
                    which include each bootstraping result for each parameters
            func_type : Type can be Exp or Power,
                    depends on which cost function you want to apply.

        """
        if func_type not in("Exp","Power"):
            raise NameError("This function is not included")

        if func_type == "Exp":
            params["specification"] = "Exp"
            estimates = vmindisest(params)
        else:
            params["specification"] = "Power"
            estimates = vmindisest(params)
        res = np.zeros((8, 2))
        # Calculating the mean and the std for each treatment
        for i in range(0, 8):
            res[i, 0], res[i, 1] = np.mean(
                estimates[i][~np.isnan(estimates[i]) & ~np.isinf(estimates[i])]
            ), np.std(estimates[i][~np.isnan(estimates[i]) & ~np.isinf(estimates[i])])
        return res

    warnings.filterwarnings("ignore")

    # Store mean and standard error of estimation
    # of the cost function from the Bootstrap procedure
    exp_res = cal_res(nw_params, func_type="Exp")
    power_res = cal_res(nw_params, func_type="Power")
    sd_exp = exp_res[0:8, 1]
    sd_power = power_res[0:8, 1]
    return table5exp, table5power, sd_exp, sd_power


def model_table(table5power, sd_power, table5exp, sd_exp):
    """Table 5 minimum distance estimates in paper.

    Args:
        Table5Power: The power cost function parameters
                        for minimum distance method
        sd_power: The standard error of
                        power cost function parameters for minimum distance method
        Table5Exp: The exponential cost function parameters
                        for minimum distance method
        sd_exp: The standard error of exponential cost function parameters
                         for minimum distance method

    """
    if any(sd_power <0):
        raise ValueError("SD shouldn't be smaller than zero")
    if any(sd_exp<0):
        raise ValueError("SD shouldn't be smaller than zero")
    
    params_name = [
        "Level k of cost of effort",
        "Curvature y of cost function",
        "Intrinsic motivation s",
        "Social preferences a",
        "Warm glow coefficient a",
        "Gift exchange Δs",
        "Present bias β",
        "(Weekly) discount factor δ",
    ]
    columns = [table5power, sd_power, table5exp, sd_exp]
    vs = []
    for col in columns:
        col = [
            f"{Decimal(col[0]):.2e}",
            round(col[1], 3),
            f"{Decimal(col[2]):.2e}",
            round(col[3], 3),
            round(col[4], 3),
            f"{Decimal(col[5]):.2e}",
            round(col[6], 2),
            round(col[7], 2),
        ]
        vs.append(col)

    table5results = pd.DataFrame(
        {
            "Parameters name": params_name,
            "power cost function_est": vs[0],
            "power cost function_authors": mini_dist_power_author,
            "p_se": vs[1],
            "exp cost function_est": vs[2],
            "exp cost function_authors": mini_dist_exp_author,
            "e_se": vs[3],
        }
    )
    return table5results
