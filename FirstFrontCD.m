function gWolf  = FirstFrontCD(gWolf)
    
    numObj = length( gWolf(1).cost );  % number of objectives
    colIdx = numObj+1;

    numInd = length(gWolf);       % nInd : number of individuals in current front
    idx   =  [1:numInd];
    cost = vertcat(gWolf.cost);
    costs = [cost, idx'];          % objective values are sorted with individual ID
    for m = 1:numObj
        costs = sortrows(costs, m); %Sort the rows of A based on the values in the second column. When the specified column has repeated elements, 
                                %the corresponding rows maintain their original order.

        gWolf( costs(1, colIdx) ).distance = inf;         % the first one
        gWolf( costs(numInd, colIdx) ).distance = inf;    % the last one

        minobj = costs(1, m);         % the minimum of objective m
        maxobj = costs(numInd, m);    % the maximum of objective m

        for i = 2:(numInd-1)
            id = costs(i, colIdx);
            if (maxobj - minobj == 0)
                gWolf(id).distance = inf;
            else
                gWolf(id).distance = gWolf(id).distance + (costs(i+1, m) - costs(i-1, m)) / (maxobj - minobj); % normalization
            end
        end
    end


end