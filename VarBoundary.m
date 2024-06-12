function range = VarBoundary(name,dim)
%******************************* VarBoundary ************************************
%% Function: VarBoundary()
%% Description: 
%   Set the minimum and maximum values of test problems
%
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%   range = VarBoundary(name,dim)
%% Parameters:
%   Inputs:
%       name: the name of test problem
%       dim:  the number of variables
%   Output:
%      range: the minimum and maximum values of the test problem,
%      range(:,1): minimum, range(:,2):maximum
%                
%                    Zhang, T.
%        Revision: 1.0.0  Date: 2021-04-01
%************************************************************************
    
    range = ones(dim,2);   
    switch name
        case {'UF1','UF2','UF5','UF6','UF7','CF2'}
            range(1,1)      =  0;
            range(2:dim,1)  = -1;
        case 'UF3'
            range(:,1)      =  0;  
        case {'UF4','CF3','CF4','CF5','CF6','CF7'}
            range(1,1)      =  0;
            range(2:dim,1)  = -2;
            range(2:dim,2)  =  2; 
        case {'UF8','UF9','UF10','CF9','CF10'}
            range(1:2,1)    =  0;
            range(3:dim,1)  = -2;
            range(3:dim,2)  =  2;   
        case 'CF1'
            range(:,1)      =  0; 
        case {'CF8'}
            range(1:2,1)    =  0;
            range(3:dim,1)  = -4;
            range(3:dim,2)  =  4;  
        case {'ZDT1','ZDT2','ZDT3','ZDT6','DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7','DTLZ8','DTLZ9','C1_DTLZ1','C2_DTLZ2','C1_DTLZ3',...
                'C3_DTLZ4','IMOP1','IMOP2','IMOP3','IMOP4','IMOP5','IMOP6','IMOP7','IMOP8','DC1_DTLZ1','LIRCMOP1','LIRCMOP6'}
            range(:,1) = 0;
            range(:,2) = 1;
        case {'WFG1','WFG2','WFG3','WFG4','WFG5','WFG6','WFG7','WFG8','WFG9'}
            range(:,1) = 0;
            range(:,2) = (2 :2: 2*dim);
        case{'ZDT4'}
            range(1,1) = 0;
            range(2:dim,1) = -5;
            range(2:dim,2) = 5;
        case{'ZDT5'}
            range(:,1) = 0;
            range(1:2) =103741824;
            range(2:dim,2) = 32;
    end
end