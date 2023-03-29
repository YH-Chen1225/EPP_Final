from decimal import Decimal

import numpy as np
import pandas as pd
import scipy.optimize as opt
from epp_final.config import aut_exp
from epp_final.config import aut_power
from epp_final.config import bp52_aut
from epp_final.config import e_est_autp1
from epp_final.config import e_est_autp2
from epp_final.config import e_est_autp3
from epp_final.config import p_est_autp1
from epp_final.config import p_est_autp2
from epp_final.config import p_est_autp3
from epp_final.config import st_values_exp
from epp_final.config import st_values_power
from epp_final.config import stvale_spec


# The result similar to the first table in the original code,
# which compare different method
# when estimate the benchmarks parameters for power cost function.
def predict_table_first(df):
    be54, se54, aut, bp52, sp52, sse, sse_aut = predict_func(df)
    bp52_least_squaree, bp52_optt, be52_ls, be52_optt = predict_other_func(df)
    result1 = predict_table(bp52_least_squaree, bp52_optt, aut, bp52, sse, sse_aut)
    return result1


def predict_table_first_r2(df):
    be54, se54, aut, bp52, sp52, sse, sse_aut = predict_func(df)
    bp52_least_squaree, bp52_optt, be52_ls, be52_optt = predict_other_func(df)
    result_r2 = predict_table_r2(be52_ls, be54, be52_optt)
    return result_r2


# This table can be corresponded to "TABLE 5" NLS on individual part in the paper.
def predict_table_two(df):
    be54, se54, aut, bp52, sp52, sse, sse_aut = predict_func(df)
    be56, se56, bp53, se53, sse_our, sse_aut = predict_other_treatment(df)
    result2 = predict_table1(be54, be56, se54, se56, bp52, bp53, sp52, se53)
    return result2


# This table can be corresponded to "TABLE 6" in the paper
def predict_table_three(df):
    be64, se64, be65, se65, bp61, sp61, bp62, sp62 = predict_prob_01(df)
    be66, se66, bp63, sp63 = predict_prob_02(df)
    result3 = predict_table2(
        be64, se64, be65, se65, bp61, sp61, bp62, sp62, be66, se66, bp63, sp63
    )
    return result3


# Estimate the benchmark parameters for exp and power cost function
def predict_func(df):
    def opt_param(effort, k_scaler, s_scaler, st_values, func_type) -> float:
        """Find the optimal parameters for cost function.

        args:
            effort: How many times the participants press the buttons
            k_scaler: This is a tool to prvent from very large number
                        show up in the calculation process,
                        and it would not affect on the result.
                        The k_scaler for "exp" and "power" can be different.
            s_scaler:  This is a tool to prvent from very large number
                        show up in the calculation process,
                        and it would not affect on the result.
                        The s_scaler for "exp" and "power" can be different.
            st_values: This is the initial guess for the parameters.
            type: type can be "exp" or "power",
                    to decide which cost function we are going to apply.

        """
        if k_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")
        if s_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")
        if func_type not in ("exp", "power"):
            raise NameError("This type of function is not included")

        def benchmark(pay100, g, k, s) -> float:
            """The cost function with benchmark treatment.

            args:
               pay100: This is the payoff for basic assumption
               g,k,s: These are original parameters in both exp and power cost function.

            """
            if func_type == "exp":
                check1 = k / k_scaler
                check2 = s / s_scaler + pay100
            else:
                check1 = max(
                    k / k_scaler, 1e-115
                )  # since check1 will enter log it must be greater than zero
                check2 = np.maximum(
                    s / s_scaler + pay100, 1e-10
                )  # np.maximum computes the max element wise.
                # We do not want a negative value inside log

            f_x = -1 / g * np.log(check1) + 1 / g * np.log(
                check2
            )  # f(x,θ) written above
            return f_x

        sol = opt.curve_fit(
            benchmark,
            df.loc[df["dummy1"] == 1].payoff_per_100,
            df.loc[df["dummy1"] == 1, effort],
            maxfev=5000,
            p0=st_values,
        )

        se = np.sqrt(np.diagonal(sol[1]))
        solo = [
            i / j for i, j in zip(sol[0], [1, k_scaler, s_scaler], strict=True)
        ]  # Change it back to original scale
        se = [
            i / j for i, j in zip(se, [1, k_scaler, s_scaler], strict=True)
        ]  # Change it back to original scale

        # Following code are for making nicer, comparable and understandable result
        sol_result = [0] * 3
        se_result = [0] * 3

        for i in range(0, len(sol[0])):
            if i == 0:
                sol_result[i] = round(solo[i], 4)
                se_result[i] = round(se[i], 5)
            else:
                sol_result[i] = f"{Decimal(solo[i]):.2e}"
                se_result[i] = f"{Decimal(se[i]):.2e}"

        # if func_type == "exp":

        sse = np.round(
            np.sum(
                (
                    benchmark(df.loc[df["dummy1"] == 1].payoff_per_100, *sol[0])
                    - df.loc[df["dummy1"] == 1, effort]
                )
                ** 2
            ),
            3,
        )  # Calculate the squared for our result
        bp52_aut = [20.546, 5.12e-13, 3.17]
        sse_aut = np.round(
            np.sum(
                (
                    benchmark(df.loc[df["dummy1"] == 1].payoff_per_100, *bp52_aut)
                    - df.loc[df["dummy1"] == 1, effort]
                )
                ** 2
            ),
            3,
        )  # Calculate the squared for authors result
        return sol_result, se_result, sse, sse_aut

    # The estimation result and standard error.
    be54, se54, noneed1, noneed2 = opt_param(
        effort="buttonpresses_nearest_100",
        k_scaler=1e16,
        s_scaler=1e6,
        st_values=st_values_exp,
        func_type="exp",
    )

    aut = [0] * 3
    for i in range(0, len(bp52_aut)):
        if i == 0:
            aut[i] = round(bp52_aut[i], 3)
        else:
            aut[i] = f"{Decimal(bp52_aut[i]):.2e}"

    # The estimation result and standard error.
    bp52, sp52, sse, sse_aut = opt_param(
        effort="logbuttonpresses_nearest_100",
        k_scaler=1e57,
        s_scaler=1e6,
        st_values=st_values_power,
        func_type="power",
    )
    return be54, se54, aut, bp52, sp52, sse, sse_aut


# Here are using other method to obtain the optimal parameters
# for both exp and power cost function.
# def predict_other_func(df):
def other_opi(df, effort, k_scaler, s_scaler, func_type, opt_type, st_values) -> float:
    """Using other way to estimate the parameters.

    Args:
        effort: How many times the participants press the button.

        k_scaler: This is a tool to prvent from very large number
                    show up in the calculation process,
                    and it would not affect on the result.
                    The k_scaler for "exp" and "power" can be different.
        s_scaler:  This is a tool to prvent from very large number
                    showup in the calculation process,
                    and it would not affect on the result.
                    The s_scaler for "exp" and "power" can be different.
        st_values: This is the initial guess for the parameters.

        opt_type: There are two optimization method can be applied here,
                    opt.minimize and opt.least_squares.
                    So Here opt_type can be "mini" or "ls"

        type: type can be "exp" or "power",
                to decide which cost function we are going to apply.

    """
    if k_scaler < 0:
        raise ValueError("The scaler shouldn't be smaller than zero")

    if s_scaler < 0:
        raise ValueError("The scaler shouldn't be smaller than zero")

    if func_type not in ("exp", "power"):
        raise NameError("This type of function is not included")

    if opt_type not in ("ls", "mini"):
        raise NameError("This type of method is not included")

    def benchmark_other(params) -> float:
        """The cost function with benchmark treatment.

        Args:
        params: This include g,k,s.

        """
        pay100 = np.array(df.loc[df["dummy1"] == 1].payoff_per_100)
        buttonpresses = np.array(df.loc[df["dummy1"] == 1, effort])
        g, k, s = params
        if func_type == "exp":
            check1 = k / k_scaler
            check2 = s / s_scaler + pay100
        elif func_type == "power":
            check1 = max(k / k_scaler, 1e-100)
            check2 = np.maximum(s / s_scaler + pay100, 1e-10)
        if opt_type == "ls":
            f_x = (
                0.5
                * ((-1 / g * np.log(check1) + 1 / g * np.log(check2)) - buttonpresses)
                ** 2
            )  # Now the f_x is different from the one previous function.
        else:
            f_x = np.sum(
                0.5
                * ((-1 / g * np.log(check1) + 1 / g * np.log(check2)) - buttonpresses)
                ** 2
            )
        return f_x

    sol_result = [0] * 3
    if opt_type == "ls":  # least squared method
        sol = opt.least_squares(
            benchmark_other,
            x0=st_values,
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            method="lm",
        )

        # Following code are for making nicer, comparable and understandable result
        solo = [
            i / j for i, j in zip(sol.x, [1, k_scaler, s_scaler], strict=True)
        ]  # change back to the original scale
        sse = np.round(
            (2 * benchmark_other(sol.x)).sum(), 3
        )  # calculate the sum of squared error
        for i in range(0, 3):
            if i == 0:
                sol_result[i] = round(solo[i], 3)
            else:
                sol_result[i] = f"{Decimal(solo[i]):.2e}"

    else:  # elif opt_type == "mini"
        sol = opt.minimize(
            benchmark_other,
            x0=st_values,
            method="Nelder-Mead",
            options={"maxiter": 2500},
        )
        solo = [
            i / j for i, j in zip(sol.x, [1, k_scaler, s_scaler], strict=True)
        ]  # change back to the original scale
        sse = np.round(
            (2 * benchmark_other(sol.x)), 3
        )  # calculate the sum of squared error
        for i in range(0, 3):
            if i == 0:
                sol_result[i] = round(solo[i], 3)
            else:
                sol_result[i] = f"{Decimal(solo[i]):.2e}"
    return sol_result, sse


def predict_other_func(df):
    bp52_least_squaree = other_opi(
        df,
        "logbuttonpresses_nearest_100",
        k_scaler=1e57,
        s_scaler=1e6,
        func_type="power",
        opt_type="ls",
        st_values=st_values_power,
    )
    bp52_optt = other_opi(
        df,
        "logbuttonpresses_nearest_100",
        k_scaler=1e57,
        s_scaler=1e6,
        func_type="power",
        opt_type="mini",
        st_values=st_values_power,
    )
    be52_ls = other_opi(
        df,
        "buttonpresses_nearest_100",
        k_scaler=1e16,
        s_scaler=1e6,
        func_type="exp",
        opt_type="ls",
        st_values=st_values_exp,
    )
    be52_optt = other_opi(
        df,
        "buttonpresses_nearest_100",
        k_scaler=1e16,
        s_scaler=1e6,
        func_type="exp",
        opt_type="mini",
        st_values=st_values_exp,
    )
    return bp52_least_squaree, bp52_optt, be52_ls, be52_optt


def predict_table(bp52_least_squaree, bp52_optt, aut, bp52, sse, sse_aut):
    """Creating table 1.

    Args:
    bp52_least_squaree : Benchmark parameters for power cost function
                            calculated by least squared method
    bp52_optt : Benchmark parameters for power cost function calculated
                            by Nelder-Mead minimized method
    aut : author's power cost parameters estimation result showing in the paper.
    bp52 : Benchmark parameters for power cost function calculated by curve fit method
    sse : The sum of squared error for each method
    sse_aut : The sum of squared error for the author.

    """
    pn = [
        "Curvature y of cost function",
        "Level k of cost of effort",
        "Intrinsic motivation s",
        "Min obj. function",
    ]
    r1 = pd.DataFrame(
        {
            "parameters": pn,
            "curve_fit": [*bp52, sse],
            "least_square": [*bp52_least_squaree[0], bp52_least_squaree[1]],
            "minimize_nd": [*bp52_optt[0], bp52_optt[1]],
            "authors": [*aut, sse_aut],
        }
    )
    return r1


def predict_table_r2(be52_ls, be54, be52_optt):
    """Creating table1_r2.

    Args:
    be52_ls : Benchmark parameters for exponential cost function
                calculated by least squared method
    be52_optt : Benchmark parameters for exponential cost function
                calculated by Nelder-Mead minimized method
    be54 : Benchmark parameters for exponential cost function
                calculated by curve fit method.

    """
    # Here I also adding the outcome of exponential cost of effort,
    # also adding the authors result for comparison
    pn = [
        "Curvature y of cost function",
        "Level k of cost of effort",
        "Intrinsic motivation s",
    ]
    be52_aut = [0.0156, 1.71e-16, 3.72e-06]
    r2 = pd.DataFrame(
        {
            "parameters": pn,
            "curve_fit": [*be54],
            "least_square": [*be52_ls[0]],
            "minimize_nd": [*be52_optt[0]],
            "authors": [
                round(be52_aut[0], 3),
                (f"{Decimal(be52_aut[1]):.2e}"),
                (f"{Decimal(be52_aut[2]):.2e}"),
            ],
        }
    )
    return r2


def predict_other_treatment(df):
    def treatment_a(effort, k_scaler, s_scaler, func_type, st_values) -> float:
        """Several treatment was added into cost function.

        effort: How many times the participants press the button.

        k_scaler: This is a tool to prvent from very large number show up
                    in the calculation process, and it would not affect on the result.
                    The k_scaler for "exp" and "power" can be different.
        s_scaler:  This is a tool to prvent from very large number
                    show up in the calculation process,
                    and it would not cause affect on the result.
                    The s_scaler for "exp" and "power" can be different.

        func_type: type can be "exp" or "power",
                to decide which cost function we are going to apply.

        st_values: This is the initial guess for the parameters k,gamma,s.

        """
        if k_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")

        if s_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")

        if func_type not in ("exp", "power"):
            raise NameError("This type of function is not included")

        # Define the f(x,θ) to estimate all parameters but the probability weight
        def noweight(xdata, g, k, s, alpha, a, gift, beta, delta) -> float:
            """Cost function assumption.

            xdata: This the vector containing the explanatory variables.

            g, k, s: These are the same parameters from before

            alpha: This the pure altruism coefficient

            a: This is the warm glow coefficient

            gift: This is the gift exchange coefficient Δs

            beta: This is the present bias parameter

            delta: This is the (weekly) discount factor

            """
            pay100 = xdata[0]
            gd = xdata[1]  # gd is gift dummy
            dd = xdata[2]  # dd is delay dummy
            dw = xdata[3]  # dw is delay weeks
            paychar = xdata[4]  # paychar is pay in charity treatment
            dc = xdata[5]  # dc is dummy charity
            if func_type == "exp":
                check1 = k / k_scaler
                check2 = (
                    s / s_scaler
                    + gift * 0.4 * gd
                    + (beta**dd) * (delta**dw) * pay100
                    + alpha * paychar
                    + a * 0.01 * dc
                )
            elif func_type == "power":
                check1 = max(k / k_scaler, 1e-115)
                check2 = np.maximum(
                    s / s_scaler
                    + gift * 0.4 * gd
                    + (beta**dd) * (delta**dw) * pay100
                    + alpha * paychar
                    + a * 0.01 * dc,
                    1e-10,
                )
            f_x = -1 / g * np.log(check1) + 1 / g * np.log(check2)
            return f_x

        # Find the solution to the problem by non-linear least squares

        st_valuesnoweight = np.concatenate((st_values, stvale_spec))  # starting values

        args = [
            df.loc[df["samplenw"] == 1].payoff_per_100,
            df.loc[df["samplenw"] == 1].gift_dummy,
            df.loc[df["samplenw"] == 1].delay_dummy,
            df.loc[df["samplenw"] == 1].delay_wks,
            df.loc[df["samplenw"] == 1].payoff_charity_per_100,
            df.loc[df["samplenw"] == 1].dummy_charity,
        ]

        sol = opt.curve_fit(
            noweight, args, df.loc[df["samplenw"] == 1, effort], st_valuesnoweight
        )

        sol_result = [0] * 8
        se_result = [0] * 8
        se = np.sqrt(np.diagonal(sol[1]))
        for i in range(0, len(sol[0])):
            if i == 5:
                sol_result[i] = f"{Decimal(sol[0][i]):.2e}"
                se_result[i] = f"{Decimal(se[i]):.2e}"
            else:
                sol_result[i] = round(sol[0][i], 3)
                se_result[i] = round(se[i], 3)
        # if func_type == "power":
        nwest_aut = [
            20.546,
            5.12e-70,
            3.17e-06,
            0.0064462,
            0.1818249,
            0.0000204,
            1.357934,
            0.7494928,
        ]
        sse_our = np.sum(
            (noweight(args, *sol[0]) - df.loc[df["samplenw"] == 1, effort]) ** 2
        )
        sse_aut = np.sum(
            (noweight(args, *nwest_aut) - df.loc[df["samplenw"] == 1, effort]) ** 2
        )
        return sol_result, se_result, sse_our, sse_aut

    # Following are the estimation result
    be56, se56, no_need1, no_need2 = treatment_a(
        "buttonpresses_nearest_100",
        k_scaler=1e16,
        s_scaler=1e6,
        func_type="exp",
        st_values=st_values_exp,
    )
    bp53, se53, sse_our, sse_aut = treatment_a(
        "logbuttonpresses_nearest_100",
        k_scaler=1e57,
        s_scaler=1e6,
        func_type="power",
        st_values=st_values_power,
    )
    return be56, se56, bp53, se53, sse_our, sse_aut


def predict_table1(be54, be56, se54, se56, bp52, bp53, sp52, se53):
    # Create and save the dataframe for table 5 NLS estimates.
    """For creating table2 in the data file.

    Args:
    be54: benchmark parameters estimation result for exp cost function
    be56: other parameters estimation result for exp cost function
    se54: standard error for exp cost function benchmark parameters estimation
    se56: standard error for exp cost function for other parameters estimation
    bp52: benchmark parameters estimation result for power cost function
    bp53: other parameters estimation result for power cost function
    sp52: standard error for power cost function benchmark parameters estimation
    se53: standard error for power cost function
            for other parameters estimation.

    """
    params_name = [
        "Curvature y of cost function",
        "Level k of cost of effort",
        "Intrinsic motivation s",
        "Social preferences a",
        "Warm glow coefficient a",
        "Gift exchange Δs",
        "Present bias β",
        "(Weekly) discount factor δ",
    ]

    be5 = be54[0:3] + be56[3:8]
    se5 = se54[0:3] + se56[3:8]
    bp5 = bp52[0:3] + bp53[3:8]
    sp5 = sp52[0:3] + se53[3:8]
    t5 = pd.DataFrame(
        {
            "parameters": params_name,
            "power_est": bp5,
            "power_aut": aut_power,
            "se_p": sp5,
            "exp_est": be5,
            "se_e": se5,
            "exp_aut": aut_exp,
        }
    )
    return t5  # Table 5: non-linear-least-squares estimates of behavioural parameters


def predict_prob_01(df):  # dis____func
    def treatment_prob(
        effort, k_scaler, s_scaler, func_type, st_values, curve
    ) -> float:
        """Estimate the parameters with probability treatment and curve is pre-defined.

        Args:
        effort: How many times the participants press the button.

        k_scaler: This is a tool to prvent from very large number
                    show up in the calculation process,
                    and it would not affect on the result.
                    The k_scaler for "exp" and "power" can be different.
        s_scaler:  This is a tool to prvent from very large number
                    show up in the calculation process,
                    and it would not cause affect on the result.
                    The s_scaler for "exp" and "power" can be different.
        type: type can be "exp" or "power",
                to decide which cost function we are going to apply.

        st_values: This is the initial guess for the parameters k,gamma,s.

        curve: the curvature of the value function

        """
        if k_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")

        if s_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")

        if func_type not in ("exp", "power"):
            raise NameError("This type of function is not included")

        def probweight(xdata, g, k, s, p_weight) -> float:
            """probability weight cost function.

            Args:
            xdata: This the vector containing the explanatory variables.

            g, k, s: These are the same parameters from before

            p_weight: p_weight is the probability weighting coefficient

            """
            pay100 = xdata[0]
            wd = xdata[1]
            prob = xdata[2]

            if func_type == "exp":
                check1 = k / k_scaler
                check2 = s / s_scaler + p_weight**wd * prob * pay100**curve
            else:
                check1 = max(k / k_scaler, 1e-115)
                check2 = np.maximum(
                    s / s_scaler + p_weight**wd * prob * pay100**curve, 1e-10
                )

            f_x = -1 / g * np.log(check1) + 1 / g * np.log(check2)
            return f_x

        prob_weight_init = [0.2]
        st_valuesprobweight = np.concatenate((st_values, prob_weight_init))
        args = [
            df.loc[df["samplepr"] == 1].payoff_per_100,
            df.loc[df["samplepr"] == 1].weight_dummy,
            df.loc[df["samplepr"] == 1].prob,
        ]
        sol = opt.curve_fit(
            probweight, args, df.loc[df["samplepr"] == 1, effort], st_valuesprobweight
        )

        # Transform to easily understanding and comparable format
        solo = [i / j for i, j in zip(sol[0], [1, k_scaler, s_scaler, 1], strict=True)]
        se = [
            i / j
            for i, j in zip(
                np.sqrt(np.diagonal(sol[1])), [1, k_scaler, s_scaler, 1], strict=True
            )
        ]
        sol_result = [0] * 4
        se_result = [0] * 4
        for i in range(0, len(solo)):
            if i in (1, 2):  # i == 1 or i == 2
                sol_result[i] = f"{Decimal(solo[i]):.2e}"
                se_result[i] = f"{Decimal(se[i]):.2e}"
            else:
                sol_result[i] = round(solo[i], 4)
                se_result[i] = round(se[i], 4)

        return np.append(sol_result, curve), np.append(se_result, 0)

    be64, se64 = treatment_prob(
        "buttonpresses_nearest_100",
        k_scaler=1e16,
        s_scaler=1e6,
        func_type="exp",
        st_values=st_values_exp,
        curve=1,
    )
    be65, se65 = treatment_prob(
        "buttonpresses_nearest_100",
        k_scaler=1e16,
        s_scaler=1e6,
        func_type="exp",
        st_values=st_values_exp,
        curve=0.88,
    )
    bp61, sp61 = treatment_prob(
        "logbuttonpresses_nearest_100",
        k_scaler=1e57,
        s_scaler=1e6,
        func_type="power",
        st_values=st_values_power,
        curve=1,
    )
    bp62, sp62 = treatment_prob(
        "logbuttonpresses_nearest_100",
        k_scaler=1e57,
        s_scaler=1e6,
        func_type="power",
        st_values=st_values_power,
        curve=0.88,
    )
    return be64, se64, be65, se65, bp61, sp61, bp62, sp62


def predict_prob_02(df):
    def treatment_prob_curve(effort, k_scaler, s_scaler, func_type, st_values) -> float:
        """Estimate Probability weight parameters, and the curve value is undefined.

        Args:
        effort: How many times the participants press the button.

        k_scaler: This is a tool to prvent from very large number
                    show up in the calculation process,
                    and it would not cause affect on the result.
                    The k_scaler for "exp" and "power" can be different.
        s_scaler:  This is a tool to prvent from very large number
                    show up in the calculation process,
                    and it would not cause affect on the result.
                    The s_scaler for "exp" and "power" can be different.

        type: type can be "exp" or "power",
                to decide which cost function we are going to apply.

        st_values: This is the initial guess for the parameters k,gamma,s.

        """
        if k_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")

        if s_scaler < 0:
            raise ValueError("The scaler shouldn't be smaller than zero")

        if func_type not in ("exp", "power"):
            raise NameError("This type of function is not included")

        def probweight(xdata, g, k, s, p_weight, curve) -> float:
            """Cost function with probability weight and curve parameters.

            Args:
            xdata: This the vector containing the explanatory variables
            g, k, s: These are the same parameters from before
            p_weight: p_weight is the probability weighting coefficient
            curve: the curvature of the value function.

            """
            pay100 = xdata[0]
            wd = xdata[1]
            prob = xdata[2]

            if func_type == "exp":
                check1 = k / k_scaler
                check2 = s / s_scaler + p_weight**wd * prob * pay100**curve
            else:
                check1 = max(k / k_scaler, 1e-115)
                check2 = np.maximum(
                    s / s_scaler + p_weight**wd * prob * pay100**curve, 1e-10
                )

            f_x = -1 / g * np.log(check1) + 1 / g * np.log(check2)
            return f_x

        prob_weight_init = [0.2]
        curv_init = [0.5]
        st_valuesprobweight = np.concatenate((st_values, prob_weight_init, curv_init))
        args = [
            df.loc[df["samplepr"] == 1].payoff_per_100,
            df.loc[df["samplepr"] == 1].weight_dummy,
            df.loc[df["samplepr"] == 1].prob,
        ]
        sol = opt.curve_fit(
            probweight, args, df.loc[df["samplepr"] == 1, effort], st_valuesprobweight
        )
        solo = [
            i / j for i, j in zip(sol[0], [1, k_scaler, s_scaler, 1, 1], strict=True)
        ]
        se = [
            i / j
            for i, j in zip(
                np.sqrt(np.diagonal(sol[1])), [1, k_scaler, s_scaler, 1, 1], strict=True
            )
        ]
        sol_result = [0] * 5
        se_result = [0] * 5
        for i in range(0, len(solo)):
            if i in (1, 2):  # i == 1 or i == 2
                sol_result[i] = f"{Decimal(solo[i]):.2e}"
                se_result[i] = f"{Decimal(se[i]):.2e}"
            else:
                sol_result[i] = round(solo[i], 4)
                se_result[i] = round(se[i], 4)
        return sol_result, se_result

    be66, se66 = treatment_prob_curve(
        "buttonpresses_nearest_100",
        k_scaler=1e16,
        s_scaler=1e6,
        func_type="exp",
        st_values=st_values_exp,
    )
    bp63, sp63 = treatment_prob_curve(
        "logbuttonpresses_nearest_100",
        k_scaler=1e57,
        s_scaler=1e6,
        func_type="power",
        st_values=st_values_power,
    )
    return be66, se66, bp63, sp63


def predict_table2(
    be64, se64, be65, se65, bp61, sp61, bp62, sp62, be66, se66, bp63, sp63
):
    """Create the table3 in data file.

    Args:
    be64: Estimation result of parameters of exp cost function
            when curve = 1, compare to "Table6" of the paper
    se64: Estimation result of standard error of parameters
            for exp cost function when curve = 1, compare to "Table6" of the paper
    be65: Estimation result of parameters of exp cost function
            when curve = 0.88, compare to "Table6" of the paper
    se65: Estimation result of standard errors of parameters
            for exp cost function when curve = 0.88, compare to "Table6" of the paper
    be66: Estimation result of parameters of exp cost function
            when curve is optimal, compare to "Table6" of the paper
    se66: Estimation result of standard errors of parameters
            for exp cost function when curve is optimal,
            please compare to "Table6" of the paper
    bp61,sp61,bp62,sp62,bp63,sp63 are defined similarly but with power cost function.

    """
    # Replicating the table 6 in the paper
    pnames = [
        "Curvature y of cost function",
        "Level k of cost of effort",
        "Intrinsic motivation s",
        "Probability weighting π (1%) (in %)",
        "Curvature of utility over piece rate",
    ]
    t6 = pd.DataFrame(
        {
            "parameters": pnames,
            "p_est1": bp61,
            "p_se1": sp61,
            "p_aut1": p_est_autp1,
            "p_est2": bp62,
            "p_se2": sp62,
            "p_aut2": p_est_autp2,
            "p_est3": bp63,
            "p_se3": sp63,
            "p_aut3": p_est_autp3,
            "e_est4": be64,
            "e_se4": se64,
            "e_aut1": e_est_autp1,
            "e_est5": be65,
            "e_se5": se65,
            "e_aut2": e_est_autp2,
            "e_est6": be66,
            "e_se6": se66,
            "e_aut3": e_est_autp3,
        }
    )
    return t6
