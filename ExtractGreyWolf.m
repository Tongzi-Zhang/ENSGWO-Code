function [tempGwo,FrontNo,MaxFNo] = ExtractGreyWolf(varargin)
%******************************* CrowdingDistance ****************************
%% Function: ExtractGreyWolf()
%% Description:  
%   This function is used to obtain the same size of  grey wolves as the original population size .
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%    [tempGwo,FrontNo,MaxFNo] = ExtractGreyWolf(varargin)
%     GreyWolves = ExtractGreyWolf(GreyWolves); ; This is used to crowded distance method
%   [GreyWolves, FrontNo,MaxFNo] = ExtractGreyWolf(GreyWolves,front);  This is used to reference point method
%% Parameters: 
%   Inputs:
%       GreyWolves:      grey wolves
%       front :     the index of grey wolves
%      	
%    Outputs:
%       gWolf:      grey wolves owning crowded distances 
%       FrontNo:    rank leve for each grey wolf (only be used in reference point method)
%       MaxFNo:     maximum rank level(only be used in reference point method)
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
    comGwo = varargin{1};
    gwoSize = length(comGwo) / 2;
    tempGwo = comGwo(1:gwoSize);    %just for initializing
    tempFront = [];
    rankVector = vertcat(comGwo.rank);

    n = 0;          % individuals number of next population
    rank = 1;       % current rank number
    idx = find(rankVector == rank);
    numInd = length(idx);       % number of individuals in current front
    while( n + numInd <= gwoSize )
        tempGwo( n+1 : n+numInd ) = comGwo( idx );

        n = n + numInd;
        rank = rank + 1;

        idx = find(rankVector == rank);
        numInd = length(idx);
    end

    % If the number of individuals in the next front plus the number of individuals 
    % in the current front is greater than the population size, then select the
    % best individuals by corwding distance(NSGA-II) or preference distance(R-NSGA-II).
    if( n < gwoSize ) 
        if nargin >1
            front = varargin{2};
            for i_rank =1 : rank
                tempFront = [tempFront  front(i_rank).f];
            end
            tempGwo = comGwo(tempFront);
        else
            distance = vertcat(comGwo(idx).distance);
            distance = [distance, idx];
            distance = flipud( sortrows( distance, 1) );      % Sort the individuals in descending order of crowding distance in the front.
            idxSelect  = distance( 1:gwoSize-n, 2);           % Select the (popsize-n) individuals with largest crowding distance.
            tempGwo(n+1 : gwoSize) = comGwo(idxSelect);
        end
    end
    % they will be applied by reference poiont method
    FrontNo = vertcat(tempGwo.rank);
    MaxFNo = max(FrontNo);
end
