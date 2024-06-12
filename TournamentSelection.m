function index = TournamentSelection(K,N,varargin)
%******************************* Non_Dominated ****************************
%% Function: Non_Dominated()
%% Description:  
%
%   P = TournamentSelection(K,N,fitness1,fitness2,...) returns the indices
%   of N solutions by K-tournament selection based on their fitness values.
%   In each selection, the candidate having the MINIMUM fitness1 value will
%   be selected; if more than one candidates have the same minimum value of
%   fitness1, then compare their fitness2 values, and so on.
%% Parameters:
%   Inputs:
%          K:           the number of tournament selection 
%          N:           the number of grey wolvesss
%          varargin:    other variables
%   Outputs:
%          index:       the index of grey wolf (you can obtain more than one individual )
%   Example:
%       P = TournamentSelection(2,100,FrontNo)
%% ----------------------------  Copyright  -------------------------------- 
%% Cited from PlatEMO All rights reserved.
%    
%       Copyright (c) 2021 BIMK Group.
%% Modified by Zhang T.
%                
%   Revision: 1.0.0  Date: 2021-04-01
%************************************************************************

    varargin = cellfun(@(S) reshape(S,[],1),varargin,'UniformOutput',false);% return by the form of cell type
    % For example
    %   A = cellfun(@(x) x(1:3),C,'UniformOutput',false)
    %   A = 1x5 cell
    %   {'Mon'}    {'Tue'}    {'Wed'}    {'Thu'}    {'Fri'}
    % reshape restruct array 
    [~,rank] = sortrows([varargin{:}]);
    [~,rank] = sort(rank);
    Parents  = randi(length(varargin{1}),K,N);
    [~,best] = min(rank(Parents),[],1);
    index    = Parents(best+(0:N-1)*K);% the index of seleciton individual or population
end