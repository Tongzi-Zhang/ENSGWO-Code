function positionC = SelectLeader(front)
%******************************* SelectLeader ************************************
%%  Function: SelectLeader()
%% Description: 
%   This function is used to select α,β,δ
% 
%% Syntax:  
%    positionC = SelectLeader(front)
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
    
    % only one solution
    if length(front) == 1
        positionC = 1; 
    else
        indexInf = find(distance == inf);
        % if the values of distance have inf, then randomly select any one 
        if (~isempty(indexInf))
           if length(indexInf) == length(distance)
               positionC = indexInf(randi(length(indexInf)));
           else
               distance(indexInf) = [];
               distance(indexInf) = 2*max(distance);
               proDistance = distance/sum(distance);
               r = rand;
               c = cumsum(proDistance);
               positionC = find(r<c,1,'first'); % location in C
           end
%             positionC = indexInf(randi(length(indexInf)));
        % if the values of distance have not inf, then probability picks a value   
        else
            if all(distance == 0)
%                 positionC = randi(length(distance)); % reference point method, not use distance
                FConViols = -vertcat(front.constraints);
                positionC = TournamentSelection(2,1,sum(max(0,FConViols),2));
            else
                
                proDistance = distance/sum(distance);
                r = rand;
                c = cumsum(proDistance);
                positionC = find(r<c,1,'first'); % location in C
            end
        end
    end
end