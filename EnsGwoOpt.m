function [struct_defaultOpt, struct_greyWolf] = EnsGwoOpt()
%******************************* EnsGwoOpt ************************************
%% Function: EnsGwoOpt()
%% Description: 
%     Create default options of parameters and individual.
% ----------------------------  Function  ---------------------------------
%% Syntax:  
%    [struct_defaultOpt, struct_greyWolf] = EnsGwoOpt()
%% Structs:
%   struct_defaultOpt: 
%       This struct is used to set the defalut value of some
%       parameter. You can modify the values of parameters in this struct to
%       simulate your test problem.
%
%   struct_greyWolf: 
%        You can add attributes of individual and set them defalut value 
%          
%                             Zhang, T.
%               Revision: 1.0.0     Date: 2021-04-01
%*****************************************************************************


%% Parameter setting
struct_defaultOpt = struct(...
... % Parameters of the model
    'greySize',200, ...           % grey wolves size
   'maxIter', 1000, ...           % maximum iteration
    'numVar',22, ...            % number of variable  ZDT(1-3): 30,ZDT4:10,ZDT5:11(delete),ZDT6:10; UF: 30; CF:10;  DTLZ1:7;  DTLZ2-6:12;  DTLZ7:22; DTLZ8:30 DTZ9:20; 
...  %C1_DTLZ1: M+4=7; C2-DTLZ2:M+9=12; C1-DTLZ3:M+9=12; C3_DTLZ4 :7; DCI_DTLZ1:7;WFG（default：12）:10 or 30 or others, dim-a*(M-1) must be larger than 0, a is an integer; IMOP:10; LIRCMOP:30
... % Method selection
    'nondominated_method',[1 2],... % 1:NSGA-II method; 2: Effective nondominanted sorting method
    'consImplement',{{'NPGM','PlatEMO'}}, ... % Constraint implementation mode 1.NPGM;2.PlatEMO
    'divImplement',{{'crowDistance','refPoints','Hypercubes'}}, ...% Two methods: crowding distance and reference points 
... % Input and output function or files    
    'objFun', {{@ZDT1,@ZDT2,@ZDT3,@ZDT4,@ZDT5,@ZDT6,@UF1,@UF2,@UF3,@UF4,@UF5,@UF6,@UF7,@UF8,@UF9,@UF10,@CF1,@CF2,@CF3,@CF4,@CF5,@CF6,@CF7,@CF8,@CF9,@CF10,...
    @DTLZ1,@DTLZ2,@DTLZ3,@DTLZ4,@DTLZ5,@DTLZ6,@DTLZ7,@DTLZ8,@DTLZ9,@C1_DTLZ1,@C2_DTLZ2,@C1_DTLZ3,@C3_DTLZ4,@DC1_DTLZ1,@WFG1,@WFG2,@WFG3,@WFG4,@WFG5,@WFG6,@WFG7,@WFG8,@WFG9,...
    @IMOP1,@IMOP2,@IMOP3,@IMOP4,@IMOP5,@IMOP6,@IMOP7,@IMOP8,@LIRCMOP1,@LIRCMOP6}}, ... % the handle of test problem
    'proName', {{'ZDT1','ZDT2','ZDT3','ZDT4','ZDT5','ZDT6','UF1','UF2','UF3','UF4','UF5','UF6','UF7','UF8','UF9','UF10',...
    'CF1','CF2','CF3','CF4','CF5','CF6','CF7','CF8','CF9','CF10','DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7','DTLZ8','DTLZ9',...
    'C1_DTLZ1','C2_DTLZ2','C1_DTLZ3','C3_DTLZ4','DC1_DTLZ1'...
    'WFG1','WFG2','WFG3','WFG4','WFG5','WFG6','WFG7','WFG8','WFG9','IMOP1','IMOP2','IMOP3','IMOP4','IMOP5','IMOP6','IMOP7','IMOP8','LIRCMOP1','LIRCMOP6'}}, ...  % the name of test problem
    'initFun', @Initialization, ...         % population initialization function (use random number as default)
    'outFun', {{@outFun}}, ...   % output function  (inactive)
    'outFile', 'greyWolves.txt', ...% output file name (inactive)
    'useParallel', 'no', .... % parallel is or not (inactive)
    'currentGen', 1, ... % current generation 
    'poolSize',0, ...    % (inactive)
    'metrics',{{'IGD','HV','Coverage','GD','Spread','Spacing','PD','DeltaP','DM','CPF'}}, ...
    'adaptive',{{1,0}}...
); 


%% Population individual setting
struct_greyWolf = struct(...
    'position', [], ...  % postion of each grey wolf
    'cost', [] , ...      % objective values of each grey wolf
    'PF',[], ...        % save true pareto front
    'dominated', false, ... % the state of dominance of a grey wolf
    'rank',0,...         % nondominated rank of grey wolf
    'distance',0,...  % crowding distance or reference distance of grey wolf
    'constraints',0, ...    % constraints value
    'consViolValue',[], ...  % violation constraints abs value, the abs value is just for the sake of constriants sum.
    'numConsViol',0, ...   % the number of violatation constraints 
    'sumCons',0, ...    % the sum of absolute value of violatation constraints
    'GridSubIndex',[], ...
    'GridIndex',[] ...
);



