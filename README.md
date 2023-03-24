<p align="center">
  <img src="https://github.com/YH-Chen1225/EPP_Final/blob/master/src/Effective_Programming_Practice_for_Economist%C2%A0Final_Project.png" alt="Sublime's custom image"/>
</p>

<br> Replication of paper: What Motivates Effort? Evidence and Expert Forecasts
=========

![image](https://img.shields.io/badge/Language-Python-brightgreen)
![image](https://img.shields.io/badge/Version-3.11-yellowgreen)
![image](https://img.shields.io/badge/Kernel-epp__final-orange)

## Author
- Name: Yu-Hsin Chen  
- Matriculation Num: 3461265
- Email : s6ynche2@uni-bonn.de 

## Main contribution of this project
- Almost all the code have been re-write in a more efficient way.
- Several graphs have been added and some table have been modify to makes the project more comprehensive
- It can run in pytask successfully

## File Managemnet
- **EPP_P** is the original jupiter notebook code replicating part of Table 5(NLS part) and Table 6 in the paper 
- **EPP_Gmm** is the original jupiter notebook code replicating another part of the Table 5 in the paper(Minimum Distance Estimator on Average Effort part)
- **Plot** is the original jupiter notebook code that making graphs
- SRC/epp_final/**data management** include the data processing process
- SRC/epp_final/analysis/**NLS** are mainly converted from EPP_P for operating pytask
- SRC/epp_final/analysis/**mini_dist** are mainly converted from EPP_Gmm for operating pytask
- SRC/epp_final/**final** are mainly converted from plot for pytask reason


## Replicating the paper

### Background
This paper are discussing what motivates people to make effort. This paper is based on a online experiment, which asking participant to press
the button as much as they can in a limit of time. When timeout, the participant would be given reward according how many times they press the button and also according to the treatment they are allocated. In the experiment, participants would not know there are several treatments and it is also not allow to re-access or refresh the page. 

### Treatment
Follwing treatment are allocating to participants
- Benchmark Treatment
  - Normal Piece Rate: Participants may earn, **1 cent per 100 points** or **10 cent per 100 points**
  - Special Piece Rate: Participants may earn, **1 cent per 1000 points**(count as not enough pay) or **no reward**

- Social preferences
  - Charity: **Charity organization** may earn, **1 cent per 100 points** or **10 cent per 100 points**
  - Gift exchange: **40 cents** would be given anyway andscore would not affect the payment
 
- Discounting
  - Participants may earn 1 cent per 100 points after **two weeks from today** or **four weeks from today**

- 
  














## Credits
This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
=======
# EPP_Final
##I haven't finished yet
>>>>>>> 5f4bd893dc890bc685010ea53f79e5ead4b0e02f
