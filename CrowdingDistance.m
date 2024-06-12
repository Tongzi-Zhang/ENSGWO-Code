function gWolf = CrowdingDistance(gWolf, front)
%******************************* CrowdingDistance ****************************
%% Function: CrowdingDistance()
%% Description:  
%   This function is used to calculate crowed distance of grey wolf.
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%    gWolf = CrowdingDistance(gWolf, front)
%% Parameters: 
%   Inputs:
%       gWolf:      grey wolves
%       front :     the index of grey wolves
%      	
%    Outputs:
%       gWolf:      grey wolves owning crowded distances 
%
%% ----------------------------  Copyright  -------------------------------- 
%% Cited from NSPM  All rights reserved.
%%              NSPM
%           LSSSSWC, NWPU
%    Revision: 1.1  Date: 2011-07-25
%
%%          Modified by Zhang T.
%                
%   Revision: 1.0.0  Date: 2021-04-01
%*************************************************************************
 

    numObj = length( gWolf(1).cost );  % number of objectives
    
    colIdx = numObj+1;
    for fid = 1:length(front)
        idx = front(fid).f;
 
        frontPop = gWolf(idx);        % frontPop : individuals in front fid

        numInd = length(idx);       % nInd : number of individuals in current front

        cost = vertcat(frontPop.cost);
        costs = [cost, idx'];          % objective values are sorted with individual ID
        for m = 1:numObj
            costs = sortrows(costs, m); %Sort the rows of A based on the values in the second column. When the specified column has repeated elements, 
                                    %the corresponding rows maintain their original order.
            
            gWolf( costs(1, colIdx) ).distance = Inf;         % the first one
            gWolf( costs(numInd, colIdx) ).distance = Inf;    % the last one

            minobj = costs(1, m);         % the minimum of objective m
            maxobj = costs(numInd, m);    % the maximum of objective m

            for i = 2:(numInd-1)
                id = costs(i, colIdx);
                if (maxobj - minobj == 0)
                    gWolf(id).distance = inf;
                else
                    gWolf(id).distance = gWolf(id).distance + (costs(i+1, m) - costs(i-1, m)) / (maxobj - minobj); % normalization
                end
            end
        end
    end
end