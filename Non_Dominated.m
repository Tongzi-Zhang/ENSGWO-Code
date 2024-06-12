function [GreyWolves,front,Hypercubes,FrontNo] = Non_Dominated(varargin)
%******************************* Non_Dominated ****************************
%% Function: Non_Dominated()
%% Description:  
%   This fucntion can invoke non-dominated sort methods for populations. There are
%   two non-dominated sort methods: NSGA-II and ENS. Moreover, two
%   selection methods are also called in this funciton, they are crowded
%   distance and reference points methods. In addtion, if you use NSGA-II
%   method, ExtractGreyWolf function will be called when the iteration are
%   bigger than 0
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%    Non_Dominated(varargin)
%   [GreyWolves,front,Hypercubes,FrontNo] = Non_Dominated(GreyWolves,nonDM,constraintImp,diversityImp,gwCons,numGWO,Z,Zmin,0)
%   [GreyWolves,front,hy]                 = Non_Dominated(GreyWolves,nonDM,constraintImp,diversityImp,gwCons,adaptive);
%% Parameters: 
%   Inputs:
%       size(varargin) = 6, it will be called by crowed distance method
%       size(varargin) > 6, it will be called by referecne points method
%   Outputs
%       GreyWolves:         the populations of grey wolves
%       front:              the index of all rank levels
%       Hypercubes (hy):    be no longer used 
%       FrontNo:            rank level of all populations
%
%            Zhang, T.
%   Revision: 1.0.0  Date: 2021-04-01
%***************************************************************************
   
    % Obtain parameters 
    gWolf = varargin{1};        % grey wolves
    nonMethod = varargin{2};    % non-dominated sort method
    consImpMethod = varargin{3};% constraint violation method
    diversityImp = varargin{4}; % selection method
    gwCons = varargin{5};       % constrain values of grey wolves
    iteration   = varargin{6};  % number of current iterations
    gwoSize = length(gWolf);    % size of grey wolves 
    Hypercubes = false;         % original MOGWO method (not use)
    numRef = gwoSize;           % the number of reference points, if diversityImp is not refPoints, numRef = sizeGWO (default size) 
    
    %  N,Z,Zmin
    if nargin >6 
        numRef      = varargin{7};  %  the number of reference point
        refUniValue = varargin{8};  % the values of reference points
        minRef      = varargin{9};  % ideal point value
    end
    
  
    % nondominated sort method 
    switch nonMethod
        case 1
            % Evaluate the constraint values of the problem 
            if (~isempty(gWolf(1).constraints))
                for i_gwo = 1:gwoSize
                     gWolf(i_gwo) = EvalConstraints(gWolf(i_gwo));
                end
            end
           [GreyWolves, front, FrontNo, MaxFNo]= Non_NSGAII(gWolf,consImpMethod,gwCons);
        case 2
           [GreyWolves, front, MaxFNo, FrontNo] = Non_ENS(gWolf,numRef,gwCons); % consImpMethod is only PlatEMO method  因为并非所有种群都rank，所以记录所有的front在进行计算拥挤度距离是不合适的，因为好多种群个体rank=inf，其拥挤度距离
    end
%     GreyWolves = CrowdingDistance(GreyWolves,front);
    % nondominated selection method
    switch diversityImp
        case 'crowDistance'
            GreyWolves = CrowdingDistance(GreyWolves,front);
            % first iteration, it should not extract, the size of
            % population is same as numGwo or numRef, firt iteration we set 0
            if iteration >= 1
               GreyWolves = ExtractGreyWolf(GreyWolves);
            end
        %  non Method must be 2, otherwise an error will be occur in calcRefDistance function    
        case 'refPoints'
            if iteration >= 1  
                if nonMethod == 1
                     %  as the same as ExtractGreyWolf, front,FrontNo,and
                     %  MaxFNo are all double size when combine son and
                     %  father populations, so here must use ExtractGreyWolf
                    [GreyWolves, FrontNo,MaxFNo] = ExtractGreyWolf(GreyWolves,front); % front must be extract
                end
                [GreyWolves,FrontNo,front]      = RefPointSelection(GreyWolves,refUniValue,minRef,MaxFNo,FrontNo,numRef); % Extract the last rank populations and obtain them 
                
            else
                 return;
            end
        case 'Hypercubes'
            Hypercubes = true;
    end
    
end