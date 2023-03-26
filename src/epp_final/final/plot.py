import plotly.express as px
import plotly.graph_objects as go
from epp_final.config import treat
import numpy as np
from plotly.subplots import make_subplots


def distribution_graph (df,treat1,treat2,treat3,treat4):
    nwdf = df.loc[(df['treatment']== treat1) | (df['treatment']== treat2) | (df['treatment']== treat3)| (df['treatment']== treat4)]
    fig = px.ecdf(nwdf, x="buttonpresses",color="treatment", markers=True, marginal="histogram")
    return fig

def ci_graph (df):
    emp_mean = df.groupby("treatment").mean()
    emp_std = df.groupby("treatment").std()
    emp_count = df.groupby("treatment").count()
    CI_range = (emp_std['buttonpresses']*1.96)/np.sqrt(emp_count['buttonpresses']-1)
    fig = go.Figure([
        go.Scatter(
            name='Mean_Button_Press',
            x=treat,
            y=emp_mean['buttonpresses'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            
        ),
        go.Scatter(
            name='Upper Bound',
            x=treat,
            y=emp_mean['buttonpresses']+np.array(CI_range),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=treat,
            y=emp_mean['buttonpresses']-np.array(CI_range),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
        ])
    fig.update_layout(
        yaxis_title='Button_Press(Effort)',
        title='Effort correspond to each treatment',
        hovermode="x"
    )
    fig.update_layout(title = "Mean effort for each treatment with confidence interval")
    return fig

def method_comparison(r1,r2):
    method = ['curve_fit','least_squared','minimize_nd','authors']

    fig = make_subplots(rows = 2, cols= 2, shared_yaxes=False)
                        #subplot_titles=('Estimation of gamma', 'Estimation of Probability weighting'))

    trace1 = go.Bar(
        x=method,
        y=r1.iloc[0,1:5],
        name='my gamma estimation in power function',
        textposition='auto',
        text=r1.iloc[0,1:5]
    )

    trace2 = go.Bar(
        x=method,
        y=r2.iloc[0,1:5],
        name='my gamma estimation in exp function',
        textposition='auto',
        text=r2.iloc[0,1:5]
    )

    trace3 = go.Bar(
        x=method,
        y=r1.iloc[3,1:5],
        name='sum of squared error',
        textposition='auto',
        text=r1.iloc[3,1:5]
    )

    #fig.append_trace(trace3, 1,1)
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,2,1)

    # Change the bar mode
    fig.update_layout(barmode='group',
        title='Comparison of parameters and sse between my result and the result of authors')
    return fig


def params_comparison(t5,a):
    fig = make_subplots(rows = 3, cols= 2, shared_yaxes=False)#subplot_titles=('Estimation of gamma', 'Estimation of Probability weighting')
                        
    trace1 = go.Bar(
        x=['my gamma estimation','authors estimation'],
        y=t5.filter(regex= a).iloc[0,:],
        name='gamma estimation comparison',
        textposition='auto',
        text=t5.filter(regex= a).iloc[0,:]
    )

    trace2 = go.Bar(
        x=['my Social preferences estimation','authors estimation'],
        y=t5.filter(regex= a).iloc[3,:],
        name='Social preferences estimation comparison',
        textposition='auto',
        text=t5.filter(regex= a).iloc[3,:]
    )

    trace3 = go.Bar(
        x=['my coefficient a estimation','authors estimation'],
        y=t5.filter(regex= a).iloc[4,:],
        name='Warm glow coefficient a estimation comparision',
        textposition='auto',
        text=t5.filter(regex= a).iloc[4,:]
    )

    trace4 = go.Bar(
        x=['my Present bias β estimation','authors estimation'],
        y=t5.filter(regex= a).iloc[6,:],
        name='Present bias β estimation comparision',
        textposition='auto',
        text=t5.filter(regex= a).iloc[6,:]
    )

    trace5 = go.Bar(
        x=['discount factor δ estimation','authors estimation'],
        y=t5.filter(regex= a).iloc[7,:],
        name='Discount factor δ estimation comparision',
        textposition='auto',
        text=t5.filter(regex= a).iloc[7,:]
    )


    #fig.append_trace(trace3, 1,1)
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace3,2,2)
    fig.append_trace(trace5,3,1)



    # Change the bar mode,and add the title
    fig.update_layout(barmode='group',
        title='Comparison of'+ " "+ f'{a}'+" "+'cost function parameters between my result and the result of authors')
    return fig



def another_params_comparison(t6,type1,type2,type3):
    func=['with_curv=1', 'with_curv=0.88', 'with_optimal_curv']
    fig = make_subplots(rows = 1, cols= 2, shared_yaxes=False,
                    subplot_titles=('Estimation of gamma', 'Estimation of Probability weighting'))
    trace1 = go.Bar(
        x=func,
        y=t6.filter(regex= f'{type1}').iloc[0,:],
        name='my gamma estimation',
        textposition='auto',
        text=t6.filter(regex= f'{type1}').iloc[0,:]
    )

    trace2 = go.Bar(
        x=func,
        y=t6.filter(regex= f'{type2}').iloc[0,:],
        name='result of author',
        textposition='auto',
        text=t6.filter(regex= f'{type2}').iloc[0,:]
    )

    trace3 = go.Bar(
        x=func,
        y=t6.filter(regex= f'{type2}').iloc[3,:],
        name='result of author',
        textposition='auto',
        text=t6.filter(regex= f'{type2}').iloc[3,:]
    )

    trace4 = go.Bar(
        x=func,
        y=t6.filter(regex= f'{type1}').iloc[3,:],
        name='my probability weight estimation',
        textposition='auto',
        text=t6.filter(regex= f'{type1}').iloc[3,:]
    )
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,1)
    fig.append_trace(trace4,1,2)
    fig.append_trace(trace3,1,2)
    fig.update_layout(barmode='group',
    title="Comparison of "+" "+f'{type3}' +" "+ 'function parameters between my result and the result of authors')
    return fig


def mini_dist (df,a):
    fig = make_subplots(rows = 3, cols= 2, shared_yaxes=False)#subplot_titles=('Estimation of gamma', 'Estimation of Probability weighting')
                            
    trace1 = go.Bar(
            x=['my gamma estimation','authors estimation'],
            y=df.filter(regex= a).iloc[1,:],
            name='gamma estimation comparison',
            textposition='auto',
            text=df.filter(regex= a).iloc[1,:]
        )

    trace2 = go.Bar(
            x=['my Social preferences estimation','author estimation'],
            y=df.filter(regex= a).iloc[3,:],
            name='Social preferences estimation comparison',
            textposition='auto',
            text=df.filter(regex= a).iloc[3,:]
        )

    trace3 = go.Bar(
            x=['my coefficient a estimation','author estimation'],
            y=df.filter(regex= a).iloc[4,:],
            name='Warm glow coefficient a estimation comparision',
            textposition='auto',
            text=df.filter(regex= a).iloc[4,:]
        )

    trace4 = go.Bar(
            x=['my Present bias β estimation','author estimation'],
            y=df.filter(regex= a).iloc[6,:],
            name='Present bias β estimation comparision',
            textposition='auto',
            text=df.filter(regex= a).iloc[6,:]
        )

    trace5 = go.Bar(
            x=['discount factor δ estimation','author estimation'],
            y=df.filter(regex= a).iloc[7,:],
            name='Discount factor δ estimation comparision',
            textposition='auto',
            text=df.filter(regex= a).iloc[7,:]
        )


        #fig.append_trace(trace3, 1,1)
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace3,2,2)
    fig.append_trace(trace5,3,1)



        # Change the bar mode
    fig.update_layout(barmode='group',
            title='Comparison of minimum dist' + " "+ f'{a}'+" "+ 'function parameters between my result and the result of authors')
    return fig










