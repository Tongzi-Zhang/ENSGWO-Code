%*******************************************************************************%
%  Effective Nondominated Sorting Multiobjective Grey Wolf Optimizer (ENS-MOGWO)%
%                                                                               %
%                      Source codes dmo version 1.0.0                          %                   
%                            Date: 2021-04-01                                   %
%                                                                               %
%                     Developed in MATLAB R2020b(9.9)                           %
%%                  Author and programmer: Tong Zhang                   %
%                                                                               %
%                                                                               %
%%                e-Mail: liulangruoxiangxi@gmail.com                   %
%                            zhangtong@cau.edu.au                               %
%                                                                               %
% -------------------------------Reference ----------------------   -------------  %                                                               -
%  Main paper:                                                                  %
%% ------------------------------ Copyright ---------------------------     %
% Copyright (c) Zhang, T.. You are free to use the ENSGWO for research purposes. %
% All publications which use this code should acknowledge the use of ENS-MOGWO   %
%  and reference: Wating...                                                      %
%*******************************************************************************%

% Notes: You can run this algorthm with the help of this command in Linux
% matlab -nodesktop     -nosplash -r Main
    
    clc, clear
    clear global;           
    tic;

    %% Obtain the current folder and add the folder path to workspace
    pwd;   % current folder
    addpath(genpath(pwd));  

    %% Model initialization setting
    [options, wolves] = EnsGwoOpt();


    %% Run to ensgwo and return optimal grey wolves
    optGWO = EnsGwo(options,wolves);

        