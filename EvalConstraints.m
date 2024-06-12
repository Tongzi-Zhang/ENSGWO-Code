function indiGWolf = EvalConstraints(indiGWolf)
%********************* EvalConstraints ************************************
%% Function:  EvalConstraints()
%% Description: 
%   This function is used to obtain informations about violation of
%   constraints.
%% Parameters:
%    Inputs: 
%       indiGWolf: individual of grey wolf
%    Outputs:
%       indiGwolf: the individual owning the number of cosntraint violations
%        and the sum of abs( constraint violations)
%
%                Zhang, T.
%   Revision: 1.0.0  Date: 2021-04-01
%*************************************************************************
   
    lenCons = length(indiGWolf.constraints);
    for i_lenC =1: lenCons
        if (indiGWolf.constraints(i_lenC)<0) % Note:  constraints of present problem are all greater than or equal to 0
            indiGWolf.consViolValue(i_lenC) = abs(indiGWolf.constraints(i_lenC));% consViolValue represent  constraint violation 
        end
    end
    
    %% Save the  constraint violations
    if (~isempty(indiGWolf.consViolValue))
        index = find(indiGWolf.consViolValue); % Find indices and values of nonzero elements
        if( ~isempty(index) )
            indiGWolf.numConsViol = length(index); % the number of constrians
            indiGWolf.sumCons = sum(abs(indiGWolf.constraints));    % sum of abs constrains
        else
            indiGWolf.numConsViol = 0; % the number of constrians
            indiGWolf.sumCons = 0;    % sum of abs constrains
        end
    end
end