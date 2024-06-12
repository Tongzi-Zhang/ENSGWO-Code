function [GreyWolves, front,FrontNo,MaxFNo]= Non_NSGAII(gWolf,consImp,gwCons)
%******************************* Non_NSGAII ****************************
%% Function: Non_NSGAII()
%% Description:  
%   The number of individuals dominate this individual and a set of individuals that this individual dominate
%   are obtained in this function. Moreover, the rank level of individual
%   is aslo divided.
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%    [GreyWolves, front,FrontNo,MaxFNo]= Non_NSGAII(gWolf,consImp,gwCons)
%% Parameters: 
%   Inputs:
%       gWolf:      grey wolves
%       consImp :   constaints method
%       gwCons:     constrain values of all grey wolves
%    Outputs:
%       GreyWolves: grey wolves including rank level
%       front:      the index of grey wolves; front(i).f: i is the rank level
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
 


    % Obtain parameters
    GreyWolves = gWolf;

    %*************************************************************************
   %% 1. Initialize variables
    %   nonDpack.np£ºnumber of individuals which dominate this individual
    %   nonDpack.sp: a set of individuals that this individual dominate
    %*************************************************************************
    
    greySize = length(GreyWolves);    % The number of grey wolves
    nonDPack = repmat(struct('np',0, 'sp', []),[1,greySize]);

    %*************************************************************************
    %% 2. Obtain dominance relationships between two individuals
    %*************************************************************************
    
    % Calculate the domination matrix for improving the efficiency.
    cost    =  vertcat(GreyWolves.cost); % 
    numConVoil = vertcat(GreyWolves.numConsViol);
    
    sumCons = vertcat(GreyWolves.sumCons);
    % gwCons =  vertcat(GreyWolves.constraints); % gwCons is used to represent constraint violations 
    domMat  = CalcDominationMatrix(numConVoil,sumCons,cost,consImp,gwCons); % domination matrix for efficiency 

    % Compute np and sp of each grey wolf 
    for p_gwo = 1:greySize-1
        for q_gwo = p_gwo+1:greySize
            if(domMat(p_gwo, q_gwo) == 1)          % p dominate q
                nonDPack(q_gwo).np = nonDPack(q_gwo).np + 1;
                nonDPack(p_gwo).sp = [nonDPack(p_gwo).sp , q_gwo];
            elseif(domMat(p_gwo, q_gwo) == -1)     % q dominate p
                nonDPack(p_gwo).np = nonDPack(p_gwo).np + 1;
                nonDPack(q_gwo).sp = [nonDPack(q_gwo).sp , p_gwo];
            end
        end
    end

    %*************************************************************************
    %% 3. Divide rank levels of populations
    %*************************************************************************
    % The first front(rank = 1)
    front(1).f = [];    % There are only one field 'f' in structure 'front'.
                        % This is intentional because the number of individuals
                        % in the front is difference.
    for i_gwo = 1:greySize
        if( nonDPack(i_gwo).np == 0 )
            GreyWolves(i_gwo).rank = 1;
            front(1).f = [front(1).f, i_gwo];  
        end
    end
    
    % Calculate pareto rank of each individuals, viz., pop(:).rank 
    fid = 1;        %pareto front ID
    while( ~isempty(front(fid).f) )
        Q = [];
        for p_gwo = front(fid).f
            if ~isempty(nonDPack(p_gwo).sp)
                for q_gwo = nonDPack(p_gwo).sp
                    nonDPack(q_gwo).np = nonDPack(q_gwo).np -1;
                    if( nonDPack(q_gwo).np == 0 )
                        GreyWolves(q_gwo).rank = fid+1;
                        Q = [Q, q_gwo];
                    end
                end
            end
        end
        fid = fid + 1;
        front(fid).f = Q;% 
    end
    front(fid) = [];    % delete the last empty front set 
    
    % These two variables are used to reference point method
    FrontNo = vertcat(GreyWolves.rank); 
    MaxFNo = max(FrontNo);
end