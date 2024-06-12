function positionC = SelectLeaderNew(front)
%******************************* SelectLeader ************************************
%%  Function: SelectLeaderNew()
%% Description: 
%   This function is used to select α,β,δ
% 
%% Syntax:  
%    positionC = SelectLeaderNew(front)
%% Parameters:
%   Inputs:
%      front:       grey wolves of the first rank level (Archive)
%   Outputs:
%      positionC:   index of the selected grey wolf       
%
%
%%              Zhang T.
%                
%       Revision: 1.0.0  Date: 2021-04-01
%*************************************************************************
    
      
      distance = vertcat(front.distance);%Obtain the crowded distance of each population of Ffront
%     [C indexc] = sort(distance);% C is in sorted order;ia and ic are index of C and distance
%     infC =find(C == inf);
%     C(infC) = [];
%     C(infC) = mean(C);
%     proDistance = C/sum(C);%Note: the first and the last population are assigned to inf
%     %RouletteWheel Method
     [~, numObj] = size(front(1).cost);
    % only one solution
    if length(front) == 1
        positionC = 1; 
    else
%        indexInf = find(distance == inf);    % reshape restruct array 
% 
%        if ~isempty(indexInf)
%             positionC = indexInf(randi(length(indexInf)));
%        else
%              proDistance = distance/sum(distance);
%              r = rand;
%              c = cumsum(proDistance);
%              positionC = find(r<c,1,'first'); % location in C
%        end
%        if nargin >= 2
%               lengthDis = length(distance);
%               positionC = randi(lengthDis);
% %            else
% %                proDistance = distance/sum(distance);
% %                r = rand;
% %                c = cumsum(proDistance);
% %                positionC = find(r<c,1,'first'); % location in C
% %            end
%        end
 

%            end
%             positionC = indexInf(randi(length(indexInf)));
        % if the values of distance have not inf, then probability picks a value   
%         
%       if all(distance == 0)
%           front  = FirstFrontCD(front);
%           distance = vertcat(front.distance);%Obtain the crowded distance of each population of Ffront
%           r = rand();
          
          Div = DivFunc(front);
          
          gwConsViol  = -vertcat(front.constraints);
           
          cost = vertcat(front.cost);
          maxCost     = max(cost,[],1);
          Infeasible         = any(gwConsViol>0, 2);
          cost(Infeasible,:) = repmat(maxCost,sum(Infeasible),1) + repmat(sum(max(0,gwConsViol(Infeasible,:)),2),1,numObj);
          
          if min(Div) < max(Div)/2 % the smaller, the better,convergence 
%                indexInf = find(distance == inf);    % reshape restruct array 
%                positionC = indexInf(randi(length(indexInf)));
%                  [~,positionC ]= max(distance);
                positionC = TournamentSelection(2,1,-Div);
          else%if min(Div)> 2* max(Div)/3
%               [~,rank] = sortrows([distance]);
%               [~,rank] = sort(rank);
%               Parents  = randi(length(distance),2,1);
%               [~,best] = max(rank(Parents),[],1);
%               positionC    = Parents(best);% the index of seleciton individual or population
%                 positionC = randi(length(distance));
%               positionC = TournamentSelection(2,1,sum(cost,2));
%           else
% %               positionC =  randi(length(Div));
           
              positionC = TournamentSelection(2,1,sum(max(0,gwConsViol),2));
             
%      
          end
     

    end
end