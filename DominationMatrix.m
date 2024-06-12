function domMat = DominationMatrix(numCons, sumCons, cost, consImp,gwCons)
% Function: domMat = calcDominationMatrix(nViol, violSum, obj)
% Description: Calculate the domination maxtrix which specified the domination
%   releation between two individual using constrained-domination.
%
% Return: 
%   domMat(N,N) : domination matrix
%       domMat(p,q)=1  : p dominates q
%       domMat(p,q)=-1 : q dominates p
%       domMat(p,q)=0  : non dominate
%
%  
%    Copyright 2011 by LSSSSWC
%    Revision: 1.0  Data: 2011-07-13
%*************************************************************************

    % Obtain parameters
    numGwo  = size(cost, 1); %
    numObj  = size(cost, 2);% 
    domMat  = zeros(numGwo, numGwo); 
    gwConsViol = -gwCons;
    
    % constrian implementation method selection
    if strcmp(consImp,'NPGM')
        domMat = calcDMNPGM(numCons, sumCons, numGwo, numObj,cost,domMat);
    elseif strcmp(consImp,'PlatEMO')
        domMat = calcDMPlatEMO(numGwo,cost,domMat,gwConsViol);
    end
end



% NPGM 
function domMat = calcDMNPGM(numConVoil, sumCons, numGwo, numObj, cost, domMat)

    for p = 1:numGwo-1
        for q = p+1:numGwo
            %*************************************************************************
            % 1. p and q are both feasible  (weak domination)
            %*************************************************************************
            if(numConVoil(p) == 0 && numConVoil(q)==0) 
                pdomq = false;
                qdomp = false;
                for i = 1:numObj
                    if( cost(p, i) < cost(q, i) )         % objective function is minimization!
                        pdomq = true; % 
                    elseif(cost(p, i) > cost(q, i))
                        qdomp = true;
                    end
                end
                if( pdomq && ~qdomp)
                    domMat(p, q) = 1; %
                elseif(~pdomq && qdomp )
                    domMat(p, q) = -1; %
                end
            %*************************************************************************
            % 2. p is feasible, and q is infeasible
            %*************************************************************************
            elseif(numConVoil(p) == 0 && numConVoil(q)~=0)
                domMat(p, q) = 1;
            %*************************************************************************
            % 3. q is feasible, and p is infeasible
            %*************************************************************************
            elseif(numConVoil(p) ~= 0 && numConVoil(q)==0)
                domMat(p, q) = -1;
            %*************************************************************************
            % 4. p and q are both infeasible (the less constraint voilation)
            %*************************************************************************
            else
                if(sumCons(p) < sumCons(q))
                    domMat(p, q) = 1;
                elseif(sumCons(p) > sumCons(q))
                    domMat(p, q) = -1;
                end
            end
        end
    end
end


% PlatEMO
function domMat = calcDMPlatEMO(varargin)
    
    % Obtain parameters
    numGwo  = varargin{1};
    cost    = varargin{2};
    domMat   = varargin{3};
    [~,numObj]  = size(cost);
    if nargin == 4
        gwCV = varargin{4};
    end
    
    Infeasible            = any(gwCV>0,2);
    cost(Infeasible,:) = repmat(max(cost,[],1),sum(Infeasible),1) + repmat(sum(max(0,gwCV(Infeasible,:)),2),1,numObj);
    
    % only comparing p and q based on costSum
    for p_Gwo = 1:numGwo-1
        for q_Gwo = p_Gwo+1:numGwo
            if ( any(cost(p_Gwo,:) < cost(q_Gwo,:)) && all(cost(p_Gwo,:) <= cost(q_Gwo,:)))
                domMat(p_Gwo,q_Gwo) = 1;
            elseif (any(cost(q_Gwo,:) < cost(p_Gwo,:)) && all(cost(q_Gwo,:) <= cost(p_Gwo,:)))
                domMat(p_Gwo,q_Gwo) = -1;
            end
        end
    end
    
end









