function GreyWolves = EnsGwo(options, gWolf)
%******************************* EnsGwo ************************************
%% Function: EnsGwo()
%% Description: 
%   All processes of the algorithm are in this file. This function include 
%   parameter selection, grey wolves initialization, non-dominated sort and
%   grey wolf algorithm. Thereinto, there are two non-dominated sorting
%   methods(Non_NSGAII and Non_ENS) and two selection methods 
%   (crowding distance and reference points )in the process of
%   non-dominated.The function of NSGWONF is similar to the process of
%   crossover and mutation of GA. 
%% Notes: 
%    1. You can modify testPro,nonDM,constraintImp, diversityImp 
%    and adaptive in the process of parameter selection.
%    2. All objectives must be minimization. If a objective is maximization, the
%    objective should be multipled by -1.
% ----------------------------  Function  ---------------------------------
%% Syntax:  
%    GreyWolves = EnsGwo(options, gWolf)
%% Parameters:
%    Inputs:
%      options: The setting of the test problem under ENSGWO algorithm
%      gWolf:   The individual of grey wolf
%    Outputs:
%      GreyWolves : The populations of grey wolf having cost, constraints,
%      distance, rank, etc. 
%
%                             Zhang, T.
%               Revision: 1.0.0     Date: 2021-04-01
% 
%***************************************************************************
    

    %**********************************************************
    %% Parameter selection and acquisition
    %**********************************************************

    sizeGWO  = options.greySize;  % the number of grey wolves
    numIter = options.maxIter;  % the number of iterations
    testPro = options.proName{33}; % test problem: 6 zdt （zdt5 delete （binary））;     10 UF;    10 CF(M=3 => CF8 CF9);    9 DTLZ;  4 C_DTLZ; 1 DC1_DTLZ;   9WFG;  8IMOP; 2 LIRCMOP
    dimVar  = options.numVar;      % the number of varibles
    
    % You can use any combinations of these three variables.
    nonDM           = options.nondominated_method(2);  % selction of nondominated sorting method.1:NSGA-II method; 2: ENS method (reference points method can only select this method)
    constraintImp   = options.consImplement{2}; % constraint implementation mode 1.NPGM;2.PlatEMO
    diversityImp    = options.divImplement{2};  %  two methods: 1.crowding distance ; 2.reference points; if reference points is selected, nondominated sorting method must be ENS method
   
    adaptive    = options.adaptive{1}; % whether the adaptive method is applied or not, 2: nonuse, 1: use(Only reference point)
    metric      = options.metrics;  %  1=IGD, 2=HV, 3=Converage,4=GD,5=Spread,6=Spacing,7=PD,8=DeltaP;9=DM;10=CPF
    GreyWolves  = gWolf;    % this variable will save all populations of grey wolf
    
    %**********************************************************
    %% Obtain lower and upper boundaries of varibles
    %**********************************************************
    range   = VarBoundary(testPro,dimVar);
    LB      = range(:,1)';
    UB      = range(:,2)';
    
    %**********************************************************
    %% Selection methods
    %**********************************************************
    
    if strcmp(diversityImp,'refPoints')
        
         %**********************************************************
        %% Initialization(postion and ovjective values are obtained)
         %**********************************************************
        [GreyWolves, objFun,numGWO,Zmin,Z,interval] = options.initFun(GreyWolves,testPro,sizeGWO,dimVar,LB,UB,diversityImp); % initialization
        gwCons                                      =  vertcat(GreyWolves.constraints); % gwCons is used to represent constraint violations 
        

         %**********************************************************
        %% Non-dominated sort
         %**********************************************************
         [GreyWolves,front,hy,FrontNo]  = Non_Dominated(GreyWolves,nonDM,constraintImp,diversityImp,gwCons,0,numGWO,Z,Zmin); % non-dominated sorting and grey wolf selection
        
         %**********************************************************
        %% Non-dominated sort
         %**********************************************************
        GreyWolves = UpdateGWO(LB,UB,GreyWolves,numGWO,numIter,objFun,nonDM,constraintImp,diversityImp,testPro,dimVar,options,metric,adaptive,FrontNo,Zmin,Z,interval,front); % update grey wolves
%       GreyWolves = UpdateGWO1(LB,UB,GreyWolves,numGWO,numIter,objFun,nonDM,constraintImp,diversityImp,testPro,dimVar,options,metric,Zmin,Z,interval,FrontNo); % must be ENS+reference point method+no adaptive
    else
        
        %**********************************************************
       %% Grey wolf algorithm (elitist strategy)
        %**********************************************************
        [GreyWolves, objFun,numGWO] = options.initFun(GreyWolves,testPro,sizeGWO,dimVar,LB,UB,diversityImp); % initialization
        gwCons                      =  vertcat(GreyWolves.constraints); % gwCons is used to represent constraint violations 
        
        %**********************************************************
       %% Non-dominated sort
        %**********************************************************
        [GreyWolves,front,hy,FrontNo] = Non_Dominated(GreyWolves,nonDM,constraintImp,diversityImp,gwCons,0); % non-dominated sorting and grey wolf selection

        % **********************************************************
       %% Grey wolf algorithm  (elitist strategy)
        %**********************************************************
        GreyWolves = UpdateGWO(LB,UB,GreyWolves,numGWO,numIter,objFun,nonDM,constraintImp,diversityImp,testPro,dimVar,options,metric,adaptive,FrontNo); % update grey wolves

    end
end