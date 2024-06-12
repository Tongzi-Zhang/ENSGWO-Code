function [GreyWolves, front, MaxFNo, FrontNo] = Non_ENS(varargin)
%******************************* Non_ENS ************************************
%%  Function: Non_ENS()
%% Description: 
%   This function is used to implement an efficient approach to
%   non-dominated sorting for evolutionary multiobjective optimization
% 
%% Syntax:  
%    Non_ENS(varargin)
%   [GreyWolves, front, MaxFNo, FrontNo] = Non_ENS(gWolf,numRef,gwCons,diversityImp); 
%% Parameters:
%   Inputs:
%      gWolf:           grey wolves 
%      numRef:          the number of reference point
%      gwCon:           constraint values of grey wolves
%      diversityImp:    selection methods
%   Outputs:
%       GreyWolves: grey wolves including rank level
%       front:      the index of grey wolves; front(i).f: i is the rank level
%       FrontNo:    rank leve for each grey wolf (only be used in reference point method)
%       MaxFNo:     maximum rank level(only be used in reference point method)        
%% ------------------------------- Reference --------------------------------
% [1] X. Zhang, Y. Tian, R. Cheng, and Y. Jin, An efficient approach to
% nondominated sorting for evolutionary multiobjective optimization, IEEE
% Transactions on Evolutionary Computation, 2015, 19(2): 201-213.
% [2] X. Zhang, Y. Tian, R. Cheng, and Y. Jin, A decision variable
% clustering based evolutionary algorithm for large-scale many-objective
% optimization, IEEE Transactions on Evolutionary Computation, 2018, 22(1):
% 97-112.
%% Cited from PlatEMO All rights reserved.
%
%%              PlatEMO
%       Copyright (c) 2021 BIMK Group.
%
%% Modified by Zhang, T.
%                
%   Revision: 1.0.0  Date: 2021-04-01
%************************************************************************
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    
    % Obtain parameters
    if nargin == 2
        cost  = varargin{1};
        nSort = varargin{2};
        [numGWO , numObj] = size(cost);
    else
        GreyWolves   = varargin{1};
        numGWO       = length(GreyWolves);
        nSort        = varargin{2};% reference points count;It is used to reference point method
        gwConsViol   = -varargin{3};
       

        [~,numObj]  = size(GreyWolves(1).cost);
        cost        = vertcat(GreyWolves.cost);
        maxCost     = max(cost,[],1);
%         minCost     = min(cost,[],1);
        Infeasible         = any(gwConsViol>0, 2);
        cost(Infeasible,:) = repmat(maxCost,sum(Infeasible),1) + repmat(sum(max(0,gwConsViol(Infeasible,:)),2),1,numObj);

        % obtain distance with the reference points method  not be reasonalbe the solution selection distance = 0
        %  tournamentSelection method variant
%         if strcmp(diversityImp,'refPoints')
%             if any(Infeasible == 1)
%                 for i_Gwo = 1:numGWO 
%                     GreyWolves(i_Gwo).distance = 1/sum(cost(i_Gwo,:)); % bigger cost value can not be selected;  feasible solution superior than infeasible solution
%                     if  GreyWolves(i_Gwo).distance >= 1/sum(maxCost) % distance of all feasible solutions are bigger than 1/sum(maxCost)
%                         GreyWolves(i_Gwo).distance = 1/sum(minCost); % distance of all feasible solutions are set 1/sum(minCost), they have the equivalent level selection 
%                     end
%                 end
%             end
%         end
    end
    [costNew,~,Loc] = unique(cost,'rows');% Unique values in array and ranked with ascending order; default: first column
    % A = [9 2 9 5];Find the unique values of A and the index vectors ia and ic, such that C = A(ia) and A = C(ic).
    % C = [2     5     9];ia = [2 4 1]';ic=[3 1 3 2]'
    
    numUniGWO = length(costNew);%  first objective value among some individuals may be equal
   
    Table          = hist(Loc,1:max(Loc));% Histogram plot  not recommanded use histogram 
    
     [costNew,rank]  = sortrows(costNew);% first objective in ascending

    FrontNo        = inf(1,numUniGWO);                                   
    MaxFNo         = 0;
    while sum(Table(FrontNo<inf)) < min(nSort,length(Loc)) 
        MaxFNo = MaxFNo + 1;
        for i = 1 : numUniGWO
            if FrontNo(i) == inf
                Dominated = false;
                for j = i-1 : -1 : 1
                     if FrontNo(j) == MaxFNo
                        m = 2;
                        while m <= numObj && (costNew(i,m) >= costNew(j,m))
                            m = m + 1;
                        end
                        Dominated = m > numObj;
                        if Dominated || numObj == 2
                            break;
                        end
                    end
                end
                if ~Dominated
                    FrontNo(i) = MaxFNo;
                end
            end
        end
    end
    FrontNo(rank) = FrontNo;
    FrontNo = FrontNo(:,Loc);  % Use unique method to return the rank of all greywolves
    
    % not reference point method
    for i_gwo = 1:numGWO
        GreyWolves(i_gwo).rank = FrontNo(i_gwo);
    end
    
    frontStr = struct('f',[]);
    front = repmat(frontStr,MaxFNo,1);
    
    
    for i_front = 1:MaxFNo
       front(i_front).f = find(FrontNo == i_front);
    end

end