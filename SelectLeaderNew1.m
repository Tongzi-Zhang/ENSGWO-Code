function positionC = SelectLeaderNew1(varargin)
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
      selectNum = 0;
      front = varargin{1};
      if nargin>1
         selectNum = varargin{2};
      end
     [~, numObj] = size(front(1).cost);
      
 %     distance = vertcat(front.distance);%Obtain the crowded distance of each population of Ffront
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
%        indexInf = find(distance == inf);    % reshape restruct array 
% 
%        if ~isempty(indexInf)
%             positionC = indexInf(randi(length(indexInf)));
%        else
% %             [C, ~] = sort(distance);% C is in sorted order;ia and ic are index of C and distance
% %             infC =find(C == inf);
% %             C(infC) = [];
% %             C(infC) = max(C);
%             proDistance = distance/sum(distance);
%             r = rand;
%             c = cumsum(proDistance);
%             positionC = find(r<c,1,'first'); % location in C
%        end
%            positionC =  TournamentSelection(2,1,-distance); 
      
% %       if all(distance == 0)
%           front  = FirstFrontCD(front);
          distance = vertcat(front.distance);%Obtain the crowded distance of each population of Ffront
%           r = rand();
          indexInf = find(distance == inf);    % reshape restruct array 
          Div = DivFunc(front);
         
% %           
% %           
% %           % 
          gwConsViol  = -vertcat(front.constraints);
           
          cost = vertcat(front.cost);
          maxCost     = max(cost,[],1);
          Infeasible         = any(gwConsViol>0, 2);
          cost(Infeasible,:) = repmat(maxCost,sum(Infeasible),1) + repmat(sum(max(0,gwConsViol(Infeasible,:)),2),1,numObj);
          tempFront = find(Div > (sum(Div)/length(front)));
%           tempFront = find(Div > ((min(Div)+max(Div))/2));
           if  length(tempFront)<1/2*length(front)
%           if min(Div) <  max(Div)/2
%                 if selectNum ==1 %|| selectNum ==2
    % %                   if  ~isempty(indexInf)
    % %                        positionC = indexInf(randi(length(indexInf)));
    % %                   else
%                            [~, positionC] = max(Div);
    % %                   end
    % % % % %                      positionC =  TournamentSelection(2,1,-distance); 
%                 elseif selectNum ==2
                       positionC =  TournamentSelection(2,1,-Div);
    % %                   positionC =  TournamentSelection(2,1,sum(cost,2),-Div); % Better  distribution
%                 else
%                       positionC =  TournamentSelection(2,1,sum(max(0,gwConsViol),2)); % Better  distribution              
    % % % % %                     positionC =  TournamentSelection(2,1,sum(cost,2));  % Two best way
    % %                    positionC =  randi(length(front));
%                 end
% %               end
% %               [~, positionC] = max(Div);
%           else%if (min(Div) >= 1* max(Div)/2) && (min(Div) < 2* max(Div)/3)
%                positionC =  TournamentSelection(2,1,sum(cost,2));
% %                positionC =  TournamentSelection(2,1,sum(max(0,gwConsViol),2));
% % %           else
% % % % %               [~, positionC] =  min(sum(max(0,gwConsViol),2));
% % % %                positionC =  TournamentSelection(2,1,sum(max(0,gwConsViol),2));
%           end
%           if  min(Div) < max(Div)/5 % the smaller, the better,convergence 
%                if selectNum ==1
%                      positionC =  TournamentSelection(2,1,-distance); 
%                else
%                  [~, positionC] = max(distance);
% %            elseif  (max(Div)/5 <= min(Div)) &&  (min(Div)<max(Div)/4)
% %                     positionC =  TournamentSelection(2,1,-distance); 
%                end  
%           elseif min(Div)> max(Div)/2
%                 positionC =  TournamentSelection(2,1,sum(cost,2));
%             if selectNum == 2
           elseif length(tempFront)>2/3*length(front)
               positionC =  TournamentSelection(2,1,sum(max(0,gwConsViol),2));
           else
                r = rand();
                if r >=0.5
                    positionC =  TournamentSelection(2,1,sum(max(0,gwConsViol),2));
                else
                    positionC =  TournamentSelection(2,1,-Div);
                end
           end
      end 
       
end