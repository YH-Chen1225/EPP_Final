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
- Several graphs have been added and more analysis result have been added and modify to makes the project more comprehensive
- It can run in pytask successfully

## File Management
- My original code
  - **EPP_P** is the original jupiter notebook code replicating part of Table 5(NLS part) and Table 6 in the paper 
  - **EPP_Gmm** is the original jupiter notebook code replicating another part(Minimum Distance Estimator on Average Effort part) of the Table 5 in the paper
  - **Plot** is the original jupiter notebook code that making graphs
- My pytask code
  - SRC/epp_final/**data management** include the data processing process
  - SRC/epp_final/analysis/**NLS** are mainly converted from EPP_P for operating pytask
  - SRC/epp_final/analysis/**mini_dist** are mainly converted from EPP_Gmm for operating pytask
  - SRC/epp_final/**final** are mainly converted from plot for pytask reason
- My result
  - BLD/latex/**epp_final.pdf**
  - BLD/python/**data**
  - BLD/python/**figures**

## How to operate this project
```
#First, pull the project to lacal repository via git bash
git pull origin master

#Second, open the anaconda powershell and cd to the corresponding path
cd (path)

#Third, activate the environment
conda activate epp_final

#Fourth, operating the project
pytask

#Fifth(optional), if there is any package that did not be install, please install according to the error message
pip install package_name
```

## Abstract
  This paper are discussing what motivates people efforts. There are many factors in our real life would makes people willing to makes efforts, including money, donation, ecouragement, comparing with other people and etc. There are totally 18 treatments discussed in the paper. This project would analyze which
  treatment motivates people the most, what is the distribution, mean and standard error of people's effort, what insightful information can we get from these distribution. Besides, both power cost function and exponential cost function would be applied to predict people's effort for each treatment. In the process, all the parameters in these two function would be estimated by several methods and assumptions. Finally, the estimation result would be compare with the one estimated by authors of paper graphically. 

## Reference
- Stefano DellaVigna and Devin Pope, 2018, "What Motivates Effort? Evidence and Expert Forecasts", Review of Economic Studies, 85(2): 1029–1069
- [Original code repository](https://github.com/MassimilianoPozzi/python_julia_structural_behavioral_economics)
- [Pytask Econ Project Template](https://github.com/OpenSourceEconomics/econ-project-templates)
