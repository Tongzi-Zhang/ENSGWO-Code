function Div = DivFunc(front) 
  
  frontPop = vertcat(front.cost);
  fmax   = max(vertcat(front.cost),[],1);
  fmin   = min(vertcat(front.cost),[],1);
  frontPop = (frontPop-repmat(fmin,size(frontPop,1),1))./repmat(fmax-fmin,size(frontPop,1),1);

   % Calculate the distance between each two solutions
  popDis = pdist2(frontPop,frontPop);
  popDis(logical(eye(length(popDis)))) = inf;
  sDis = sort(popDis,2);
  Div  = sDis(:,1) + 0.01*sDis(:,2); % 
  
end