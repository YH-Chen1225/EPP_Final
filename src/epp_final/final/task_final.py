import pytask
from epp_final.config import BLD
from epp_final.config import GROUPS
from epp_final.config import SRC

import pandas as pd
from epp_final.final.plot import distribution_graph
from epp_final.final.plot import ci_graph
from epp_final.final.plot import method_comparison
from epp_final.final.plot import params_comparison
from epp_final.final.plot import another_params_comparison
from epp_final.final.plot import mini_dist




kwargs = {"produces": BLD / "python" / "figures" / "benchmark_treatment.png"}
@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_benchmark_treatment(depends_on,produces):
    df = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(df,"1.1","1.2","1.3","2")
    fig.update_layout(title = "benchmark_treatment_distribution_graph")
    fig.show()
    fig.write_image(produces)



kwargs = {"produces": BLD / "python" / "figures" / "charity_treatment.png"}
@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_charity_treatment(depends_on,produces):
    df = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(df,"1.1","1.1","3.1","3.2")
    fig.update_layout(title = "charity_treatment_distribution_graph")
    fig.show()
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "delay_treatment.png"}
@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_delay_treatment(depends_on,produces):
    df = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(df,"1.1","4.1","1.1","4.2")
    fig.update_layout(title = "delay_treatment_distribution_graph")
    fig.show()
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "gain_lose_treatment.png"}
@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_gain_lose_treatment(depends_on,produces):
    df = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(df,"1.1","5.1","5.2","5.3")
    fig.update_layout(title = "gain_lose_treatment_distribution_graph")
    fig.show()
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "probability_weight_treatment.png"}
@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_probability_weight_treatment(depends_on,produces):
    df = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(df,"1.1","6.1","6.1","6.2")
    fig.update_layout(title = "probability_weight_treatment_distribution_graph")
    fig.show()
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "social_treatment.png"}
@pytask.mark.depends_on(
    {
    "data" : BLD / "python" / "data" / "data_modify.pickle"
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_social_treatment(depends_on,produces):
    df = pd.read_pickle(depends_on["data"])
    fig = distribution_graph(df,"1.1","7","8","9")
    fig.update_layout(title = "social_treatment_distribution_graph")
    fig.show()
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "confidence_interval.png"}
@pytask.mark.depends_on(
    {
    "data":SRC / "data" / "mturk_clean_data_short.dta"
    }
)
@pytask.mark.task(kwargs=kwargs)
def task_plot_confidence_interval(depends_on,produces):
    df = pd.read_stata(depends_on["data"])
    fig = ci_graph(df)
    fig.show()
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "method_comparison.png"}
@pytask.mark.depends_on(
    {
    "data":BLD / "python" / "data" / "table_1.csv",
    "data1":BLD / "python" / "data" / "table_1_r2.csv"
    }
)

@pytask.mark.task(kwargs=kwargs)
def task_plot_method_comparison(depends_on,produces):
    r1 = pd.read_csv(depends_on["data"])
    r2 = pd.read_csv(depends_on["data1"])
    fig = method_comparison(r1,r2)
    fig.show()
    fig.write_image(produces)



kwargs = {"produces": BLD / "python" / "figures" / "params_comparison_exp.png"}
@pytask.mark.depends_on(
    {
    "data":BLD / "python" / "data" / "table_2.csv",
    }
)

@pytask.mark.task(kwargs=kwargs)
def task_plot_params_comparison(depends_on,produces):
    t5 = pd.read_csv(depends_on["data"])
    fig = params_comparison(t5,"exp")
    fig.show()
    fig.write_image(produces)


kwargs = {"produces": BLD / "python" / "figures" / "params_comparison_power.png"}
@pytask.mark.depends_on(
    {
    "data":BLD / "python" / "data" / "table_2.csv",
    }
)

@pytask.mark.task(kwargs=kwargs)
def task_plot_params_comparison(depends_on,produces):
    t5 = pd.read_csv(depends_on["data"])
    fig = params_comparison(t5,"power")
    fig.show()
    fig.write_image(produces)




kwargs = {"produces": BLD / "python" / "figures" / "another_params_comparison_exp.png"}
@pytask.mark.depends_on(
    {
    "data":BLD / "python" / "data" / "table_3.csv",
    }
)

@pytask.mark.task(kwargs=kwargs)
def task_plot_another_params_comparison(depends_on,produces):
    t6 = pd.read_csv(depends_on["data"])
    fig = another_params_comparison(t6,"e_est","e_aut","exponential")
    fig.show()
    fig.write_image(produces)



kwargs = {"produces": BLD / "python" / "figures" / "another_params_comparison_power.png"}
@pytask.mark.depends_on(
    {
    "data":BLD / "python" / "data" / "table_3.csv",
    }
)

@pytask.mark.task(kwargs=kwargs)
def task_plot_another_params_comparison(depends_on,produces):
    t6 = pd.read_csv(depends_on["data"])
    fig = another_params_comparison(t6,"p_est","p_aut","power")
    fig.show()
    fig.write_image(produces)



kwargs = {"produces": BLD / "python" / "figures" / "mini_dist_params_comparison_power.png"}
@pytask.mark.depends_on(
    {
    "data":BLD / "python" / "data" / "table_gmm.csv",
    }
)

@pytask.mark.task(kwargs=kwargs)
def task_plot_mini_dist_params_comparison_power(depends_on,produces):
    gmm = pd.read_csv(depends_on["data"])
    fig = mini_dist(df=gmm,a = "power cost function")
    fig.show()
    fig.write_image(produces)

kwargs = {"produces": BLD / "python" / "figures" / "mini_dist_params_comparison_exp.png"}
@pytask.mark.depends_on(
    {
    "data":BLD / "python" / "data" / "table_gmm.csv",
    }
)

@pytask.mark.task(kwargs=kwargs)
def task_plot_mini_dist_params_comparison_power(depends_on,produces):
    gmm = pd.read_csv(depends_on["data"])
    fig = mini_dist(df=gmm,a = "exp cost function")
    fig.show()
    fig.write_image(produces)
