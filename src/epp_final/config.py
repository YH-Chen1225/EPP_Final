"""This module contains the general configuration of the project."""
from pathlib import Path
from decimal import Decimal


SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

GROUPS = ["marital_status", "qualification"]

genre = ['payoff_per_100','payoff_charity_per_100',
         'dummy_charity','delay_wks','delay_dummy','gift_dummy','prob','weight_dummy'] 

assum = {'payoff_per_100':{'1.1':0.01,'1.2':0.1,'1.3':0.0,'2':0.001,'1.4':0.04,'4.1':0.01,'4.2':0.01,'6.2':0.02,'6.1':1},
 'payoff_charity_per_100':{'3.1':0.01,'3.2':0.1},
 'dummy_charity':{'3.1':1,'3.2':1},
 'delay_wks':{'4.1':2,'4.2':4},
 'delay_dummy':{'4.1':1,'4.2':1},
 'prob':{'6.2':0.5,'6.1':0.01},
 'weight_dummy':{'6.1':1},
 'gift_dummy':{'10':1}
 }

# Set the initial values guess for the optimization procedure and scalers for k and s in the exp cost function case
st_values_exp = [0.015645717, 1.69443, 3.69198]#gamma_init_exp, k_init_exp, s_init_exp

st_values_power = [19.8117987, 1.66306e-10, 7.74996]#gamma_init_power, k_init_power, s_init_power

bp52_aut = [20.546,5.12e-70,3.17e-06]# The result of parameters of power cost function that author get.

stvale_spec = [0.003, 0.13, 1.16, 0.75, 5e-6] #alpha_init, a_init, beta_init, delta_init, gift_init

aut_power = [round(20.546,4),'{0:.2e}'.format(Decimal(5.12e-70)),'{0:.2e}'.format(Decimal(3.17e-06)),round(0.006,3),
             round(0.182,3),'{0:.2e}'.format(Decimal(2.04e-05)),round(1.36,2),round(0.75,2)]#Author's result for "table5 in the paper"

aut_exp = [round(0.0156,4),'{0:.2e}'.format(Decimal(1.71e-16)),'{0:.2e}'.format(Decimal(3.72e-06)),round(0.07,3),round(0.035,3),
           '{0:.2e}'.format(Decimal(3e-05)),round(0.79,2),round(0.86,2)]#Author's result for "table5 in the paper"
#Author's result for "table6" in the paper"
p_est_autp1 = [round(20.59,2),'{0:.2e}'.format(Decimal(3.77e-70)),'{0:.2e}'.format(Decimal(2.66e-06)),round(0.19,2),round(1,0)]#power_func,curve = 1
p_est_autp2 = [round(18.87,2),'{0:.2e}'.format(Decimal(3.92e-64)),'{0:.2e}'.format(Decimal(6.22e-06)),round(0.38,2),round(0.88,2)]#power_func,curve = 0.88
p_est_autp3 = [round(19.64,2),'{0:.2e}'.format(Decimal(1.02e-66)),'{0:.2e}'.format(Decimal(3.75e-06)),round(0.30,2),round(0.92,2)]#power_func,curve is optimal

e_est_autp1 = [round(0.0134,2),'{0:.2e}'.format(Decimal(2.42e-14)),'{0:.2e}'.format(Decimal(1.65e-5)),round(0.24,2),round(1,0)]#exp_func,curve = 1
e_est_autp2 = [round(0.0119,2),'{0:.2e}'.format(Decimal(7.5e-13)),'{0:.2e}'.format(Decimal(5.55e-5)),round(0.47,2),round(0.88,2)]#exp_func,curve = 0.88
e_est_autp3 = [round(0.0072,2),'{0:.2e}'.format(Decimal(5.46e-08)),'{0:.2e}'.format(Decimal(3.14e-3)),round(4.3,2),round(0.47,2)]#exp_func,curve is optimal

mini_dist_power_author = ['{0:.2e}'.format(Decimal(2.56E-112)),round(33.137,3),'{0:.2e}'.format(Decimal(7.13E-07)),
                          round(0.003,3), round(0.125,3), '{0:.2e}'.format(Decimal(3.26e-06)), round(1.17,2), round(0.75,2)]


mini_dist_exp_author = ['{0:.2e}'.format(Decimal(1.27E-16)),round(0.0158,3),'{0:.2e}'.format(Decimal(3.32E-06)),
                          round(0.003,3), round(0.143,3), '{0:.2e}'.format(Decimal(8.59e-06)), round(1.15,2), round(0.76,2)]

treat = ["1c PieceRate","10c PieceRate","No Payment","4c PieceRate","Gift_exchange","Very Low Pay","1c RedCross","10c RedCross","1c 2Wks","1c 4Wks","Gain 40c"
 ,"Loss 40c","Gain 80c","Prob.01 $1","Prob.5 2c","Social Comp","Ranking","Task Signif"]#All the treatment

__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]
