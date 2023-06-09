import pandas as pd
import pytask
from epp_final.config import BLD
from epp_final.config import SRC
from epp_final.final.plot import another_params_comparison
from epp_final.final.plot import ci_graph
from epp_final.final.plot import distribution_graph
from epp_final.final.plot import method_comparison
from epp_final.final.plot import mini_dist
from epp_final.final.plot import params_comparison


kwargs = {"produces": BLD / "python" / "figures" / "benchmark_treatment.html"}


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_benchmark_treatment(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(data, "1.1", "1.2", "1.3", "2")
    fig.update_layout(title="benchmark_treatment_distribution_graph")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "charity_treatment.html"}


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_charity_treatment(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(data, "1.1", "1.1", "3.1", "3.2")
    fig.update_layout(title="charity_treatment_distribution_graph")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "delay_treatment.html"}


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_delay_treatment(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(data, "1.1", "4.1", "1.1", "4.2")
    fig.update_layout(title="delay_treatment_distribution_graph")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "gain_lose_treatment.html"}


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_gain_lose_treatment(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(data, "1.1", "5.1", "5.2", "5.3")
    fig.update_layout(title="gain_lose_treatment_distribution_graph")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "probability_weight_treatment.html"}


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_probability_weight_treatment(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(data, "1.1", "6.1", "6.1", "6.2")
    fig.update_layout(title="probability_weight_treatment_distribution_graph")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "social_treatment.html"}


@pytask.mark.depends_on({"data": BLD / "python" / "data" / "data_modify.pickle"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_social_treatment(depends_on, produces):
    data = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(data, "1.1", "7", "8", "9")
    fig.update_layout(title="social_treatment_distribution_graph")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "confidence_interval.html"}


@pytask.mark.depends_on({"data": SRC / "data" / "mturk_clean_data_short.dta"})
@pytask.mark.task(kwargs=kwargs)
def task_plot_confidence_interval(depends_on, produces):
    data = pd.read_stata(depends_on["data"])
    fig = ci_graph(data)
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "method_comparison.html"}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "table_1.csv",
        "data1": BLD / "python" / "data" / "table_1_r2.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_method_comparison(depends_on, produces):
    r1 = pd.read_csv(depends_on["data"])
    r2 = pd.read_csv(depends_on["data1"])
    fig = method_comparison(r1, r2)
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "params_comparison_exp.html"}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "table_2.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_params_comparison_exp(depends_on, produces):
    t5 = pd.read_csv(depends_on["data"])
    fig = params_comparison(t5, "exp")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "params_comparison_power.html"}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "table_2.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_params_comparison_power(depends_on, produces):
    t5 = pd.read_csv(depends_on["data"])
    fig = params_comparison(t5, "power")
    fig.write_html(produces)


kwargs = {"produces": BLD / "python" / "figures" / "another_params_comparison_exp.html"}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "table_3.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_another_params_comparison_exp(depends_on, produces):
    t6 = pd.read_csv(depends_on["data"])
    fig = another_params_comparison(t6, "e_est", "e_aut", "exponential")
    fig.write_html(produces)


kwargs = {
    "produces": BLD / "python" / "figures" / "another_params_comparison_power.html"
}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "table_3.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_another_params_comparison_power(depends_on, produces):
    t6 = pd.read_csv(depends_on["data"])
    fig = another_params_comparison(t6, "p_est", "p_aut", "power")
    fig.write_html(produces)


kwargs = {
    "produces": BLD / "python" / "figures" / "mini_dist_params_comparison_power.html"
}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "table_gmm.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_mini_dist_params_comparison_power(depends_on, produces):
    gmm = pd.read_csv(depends_on["data"])
    fig = mini_dist(df=gmm, a="power cost function")
    fig.write_html(produces)


kwargs = {
    "produces": BLD / "python" / "figures" / "mini_dist_params_comparison_exp.html"
}


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "table_gmm.csv",
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_mini_dist_params_comparison_exp(depends_on, produces):
    gmm = pd.read_csv(depends_on["data"])
    fig = mini_dist(df=gmm, a="exp cost function")
    fig.write_html(produces)
