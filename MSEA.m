function MatingPool = MSEA (FrontNo, GWolf, numGWO,distance)
    Div = DivFunc(GWolf);
    [~,numObj] = size(GWolf(1).cost);
    gwConsViol  = -vertcat(GWolf.constraints);
%     cost = vertcat(GWolf.cost);
%     maxCost     = max(cost,[],1);
%     Infeasible         = any(gwConsViol>0, 2);
%     cost(Infeasible,:) = repmat(maxCost,sum(Infeasible),1) + repmat(sum(max(0,gwConsViol(Infeasible,:)),2),1,numObj);
%    
     [~,indexDiv] = sort(Div);
     rankDiv = indexDiv(numGWO-2:numGWO);
%      [~,indexDis] = sort(distance);
%     
    if  max(FrontNo)>1
%        indexInf = find(distance == inf);  
%        if  ~isempty(indexInf)
%             MatingPool(1) = indexInf(randi(length(indexInf)));
%         else
%            [~,  MatingPool(1)] = max(distance);
%        end   
%        LInf = length(indexInf);
%        if  LInf ~= numGWO && LInf >1
%             rankDis = indexDis(numGWO-LInf:numGWO-LInf+1);
%        else
%             rankDis = TournamentSelection(2,2,-distance);
%        end
%        MatingPool(2:3) = rankDis;
% %         
%        MatingPool(4:numGWO) =  TournamentSelection(2,numGWO-3,sum(max(0,gwConsViol),2)); % More real
%          MatingPool =  TournamentSelection(2,numGWO,FrontNo,-Div);
        MatingPool =  TournamentSelection(2,numGWO,sum(max(0,gwConsViol),2)); % More real
    elseif min(Div)< max(Div)/2
%        rankDisc = indexDis(numGWO-2:numGWO);
      [~, MatingPool(1) ]= max(distance);   
       MatingPool(2:numGWO) =  TournamentSelection(2,numGWO-1,-distance); 

%        indexInf = find(distance == inf);  
%        if  ~isempty(indexInf)
%             MatingPool(1) = indexInf(randi(length(indexInf)));
%             MatingPool(2) = indexInf(randi(length(indexInf)));
%             MatingPool(3) = indexInf(randi(length(indexInf)));
%         else
%             MatingPool(1:3) = indexDis(numGWO-2:numGWO);
%        end   
%            MatingPool(4:numGWO) =  TournamentSelection(2,numGWO-3,-distance); 
%             MatingPool =  TournamentSelection(2,numGWO,-Div); 
%        rankDis = indexDis(numGWO-2:numGWO-1);
%       
%        MatingPool(2:3) = rankDis;
% %         
%        MatingPool(4:numGWO) =  TournamentSelection(2,numGWO-3,-distance); 
    else
        r = unifrnd(0,1,1);
        
        if r > 0.5
            MatingPool =  TournamentSelection(2,numGWO,-Div); 
        else
            MatingPool =  TournamentSelection(2,numGWO,sum(max(0,gwConsViol),2),-Div);
        end
  
    end
end