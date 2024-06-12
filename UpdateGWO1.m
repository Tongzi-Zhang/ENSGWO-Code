
function GreyWolves = UpdateGWO1(LB,UB,gWolf,numGWO,maxIter,objFun,nonDM,consImp,divImp,testPro,dimVar,options,metric,Zmin,Z,interval,FrontNo)
%% Add MSEA
   
    %% Algorithm Variables
    GreyWolves = gWolf;
    [~,numObj] = size(GreyWolves(1).cost);
%     [~,numVar] = size (GreyWolves(1).position)
    [~,GW] = EnsGwoOpt(); % initialize a grey wolf struct
    greyWolfNew= repmat(GW,numGWO,1); % Size: the number of grey wolves
    funStr    = func2str(objFun);
    
    % Whale_pos_ad = zeros(SearchAgents_no,K);
    %% Optimization Circle
    Iteration = 1;
    % evalParallel(options,numGWO);
    objPF = PF(testPro); 
    pf = objPF();
%     Ffront = GreyWolves((find(vertcat(GreyWolves.rank) == 1))); % first front
%     GW = GreyWolves;
    while Iteration <= maxIter % for each generation   
        a=2-Iteration*((2)/maxIter ); % a decreases linearly fron 2 to 0 in Eq. (2.3)
        options.currentGen = Iteration;
        % evalParallel(options,numGWO);
        fprintf('\n\n************************************************************\n');
        fprintf('*      Current generation %d / %d\n', options.currentGen, options.maxIter);
        fprintf('************************************************************\n');       
     
%         PopCons = -vertcat(GreyWolves.constraints);
%         MatingPool = TournamentSelection(2,numGWO,sum(max(0,PopCons),2));
        GW  = GreyWolves;
%         minFront = min(vertcat(GW.rank));
        Ffront = GreyWolves((vertcat(GreyWolves.rank) == 1)); % first front 
        
%         greyWolfOld = GreyWolves;
      % Ffront = GreyWolves((find(vertcat(GreyWolves.rank) == 1))); % first front      
        
%          Fcost = vertcat(Ffront.cost);
%         [~,ia,ic] = unique(Fcost,'rows');
%         Ffront = Ffront(ia);
%         [LFfront,~] = size(Ffront);%Length of the first front
        
        PopObj = vertcat(GreyWolves.cost);      
        gwConsViol  = -vertcat(GreyWolves.constraints);
        fmax   = max(vertcat(Ffront.cost),[],1);
        fmin   = min(vertcat(Ffront.cost),[],1);
        PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);
     
        %  feasible solutions are superior to infeasible solutions (UpdateFront)
  
%         cost = vertcat(GreyWolves.cost); 
%         maxCost     = max(cost,[],1);
%         Infeasible         = any(gwConsViol>0, 2);
%         PopCons =  repmat(sum(max(0,gwConsViol(Infeasible,:)),2),1,numObj);
%         cost(Infeasible,:) = repmat(maxCost,sum(Infeasible),1) + (PopCons-repmat(fmin,size(PopCons,1),1))./repmat(fmax-fmin,size(PopCons,1),1);
%         

 
        
        % Calculate the distance between each two solutions
        Distance = pdist2(PopObj,PopObj);
        Distance(logical(eye(length(Distance)))) = inf;

        % Local search
        for i_Gwo = 1 : numGWO
         % Determining the stage
           clear residuePop;
           clear residue2Pop;
           sDis = sort(Distance,2);
           Div  = sDis(:,1) + 0.01*sDis(:,2);
           if max(FrontNo) > 1 % not converaged
               stage = 1;
           elseif min(Div) < max(Div)/2 % spread not too minimization
               stage = 2;
           else
               stage = 3;
           end

           % Generate an offspring
            switch stage
              case 1
                 MatingPool(1) = TournamentSelection(2,1,FrontNo,sum(PopObj,2));
                 residuePop = repmat(sum(PopObj,2),1,1);
                 residuePop(MatingPool(1),:) = [];
                 residueFrontNo = FrontNo;
                 residueFrontNo(MatingPool(1)) = [];
                 MatingPool(2) = TournamentSelection(2,1,residueFrontNo,sum(residuePop,2));
                 residue2Pop = repmat(residuePop(:,:),1,1);
                 residue2Pop(MatingPool(2),:) = [];
                 residue2FrontNo = residueFrontNo;
                 residue2FrontNo(MatingPool(2)) = [];
                 MatingPool(3) = TournamentSelection(2,1,residue2FrontNo,sum(residue2Pop,2));
              case 2
                 [~,MatingPool(1)] = max(Div);
                 residueDiv = Div;
                 residueDiv(MatingPool(1)) = [];
                 MatingPool(2) = TournamentSelection(2,1,-residueDiv);      
                 residue2Div = residueDiv;
                 residue2Div(MatingPool(2)) =[];
                 MatingPool(3) = TournamentSelection(2,1,-residue2Div); 
                 
%                MatingPool(2:3)     = tournamentSelection(2,2,-Div);
                otherwise
                    MatingPool(1) = TournamentSelection(2,1,FrontNo,sum(PopObj,2));
                    MatingPool(2) = TournamentSelection(2,1,-Div);
                    residuePop = repmat(sum(PopObj,2),1,1);
                    residueDiv = Div;
                    residuePop(MatingPool(1),:) = [];
                    residueDiv(MatingPool(1)) = [];
                    residueFrontNo = FrontNo;
                    residueFrontNo(MatingPool(1)) = [];
                    r = rand();
                    if r > 0.5
                        MatingPool(3) = TournamentSelection(2,1,residueFrontNo,sum(residuePop,2)); 
                    elseif r == 0.5 && mod(i_Gwo,2) == 0
                        MatingPool(3) = TournamentSelection(2,1,-residueDiv);
                    elseif r == 0.5 && mod(i_Gwo,2) ~=0
                        MatingPool(3) = TournamentSelection(2,1,residueFrontNo,sum(residuePop,2)); 
                    else 
                        MatingPool(3) = TournamentSelection(2,1,-residueDiv);
                    end
            end
                    
            Alpha(1).position = GreyWolves(MatingPool(1)).position;
            Beta(1).position  = GreyWolves(MatingPool(2)).position;
            Delta(1).position = GreyWolves(MatingPool(3)).position;
           
            % Modify r1,r2,A1 and C1 dimension to 1*dimVar
%             r1 = rand(); % r1 is a random number in [0,1]
%             r2 = rand(); % r2 is a random number in [0,1]   
            r1 = rand(1,dimVar); % r1 is a random number in [0,1]
            r2 = rand(1,dimVar); % r2 is a random number in [0,1]       
            A1 = 2.*a.*r1-a;  % Eq. (2.3) in the paper
            C1 = 2.*r2;      % Eq. (2.4) in the paper 
            X1 = Alpha(1).position-A1.*abs(C1.*Alpha(1).position-GW(i_Gwo).position);

%             r1 = rand(); % r1 is a random number in [0,1]
%             r2 = rand(); % r2 is a random number in [0,1]   
            r1 = rand(1,dimVar); % r1 is a random number in [0,1]
            r2 = rand(1,dimVar); % r2 is a random number in [0,1]       
            A2 = 2.*a.*r1-a;  % Eq. (2.3) in the paper
            C2 = 2.*r2;      % Eq. (2.4) in the paper 
            X2 = Beta(1).position-A2.*abs(C2.*Beta(1).position-GW(i_Gwo).position);


%             r1 = rand(); % r1 is a random number in [0,1]
%             r2 = rand(); % r2 is a random number in [0,1]   
            r1 = rand(1,dimVar); % r1 is a random number in [0,1]
            r2 = rand(1,dimVar); % r2 is a random number in [0,1]       
            A3 = 2.*a.*r1-a;  % Eq. (2.3) in the paper
            C3 = 2.*r2;      % Eq. (2.4) in the paper 
            X3 = Delta(1).position-A3.*abs(C3.*Delta(1).position-GW(i_Gwo).position);

            greyWolfNew(i_Gwo).position=(X1+X2+X3)./3;
            greyWolfNew(i_Gwo).position = min(max(greyWolfNew(i_Gwo).position,LB),UB); 
            
            switch funStr(1)
                    case 'C'
                        [greyWolfNew(i_Gwo).cost,greyWolfNew(i_Gwo).constraints]= objFun(greyWolfNew(i_Gwo).position');
                        % Store by column way for later use
                        greyWolfNew(i_Gwo).cost = greyWolfNew(i_Gwo).cost';
                        %greyWolfNew(i_Gwo).constraints = abs(greyWolfNew(i_Gwo).constraints'); %why abs?
                        greyWolfNew(i_Gwo).constraints = greyWolfNew(i_Gwo).constraints'; 
                    otherwise
                        greyWolfNew(i_Gwo).cost = objFun(greyWolfNew(i_Gwo).position');
                         % Store by column way for later use
                        greyWolfNew(i_Gwo).cost = greyWolfNew(i_Gwo).cost';
           end 
                    
           OffObj    = (greyWolfNew(i_Gwo).cost-fmin)./(fmax-fmin);
%            Infeasible         = any(-greyWolfNew(i_Gwo).constraints > 0, 2);
%            if Infeasible == 1
%                PopCons = repmat(sum(max(0,-greyWolfNew(i_Gwo).constraints),2),1,numObj);
%                costOff = max([maxCost;greyWolfNew(i_Gwo).cost]) + PopCons;
%            else 
%                costOff = OffObj;
%            end
               
           % Non-dominated sorting
           NewFront = UpdateFront([PopObj;OffObj],FrontNo);
           if NewFront(end) > 1
               continue;
           end

           % Calculate the distances
           OffDis = pdist2(OffObj,PopObj);

            % Determining the stage
           if max(NewFront) > 1
               stage = 1;
           elseif min(Div) < max(Div)/2
               stage = 2;
           else
               stage = 3;
           end

                    % Update the population
                    replace = false;
                    switch stage
                        case 1
                            Worse = find(NewFront==max(NewFront));
                            [~,q] = max(sum(PopObj(Worse,:),2));
                            q     = Worse(q);
                            OffDis(q) = inf;
                            replace   = true;
                        case 2
                            [~,q]     = min(Div);
                            OffDis(q) = inf;
                            sODis     = sort(OffDis);
                            ODiv      = sODis(1) + 0.01*sODis(2);
                            if ODiv >= Div(q)
                                replace = true;
                            end
                        otherwise
                            [~,q]     = min(OffDis);
                            OffDis(q) = inf;
                            sODis     = sort(OffDis);
                            ODiv      = sODis(1) + 0.01*sODis(2);
                            if sum(OffObj) <= sum(PopObj(q,:)) && ODiv >= Div(q)
                                replace = true;
                            end
                    end
                    if replace
                        % Update the front numbers
                        FrontNo = UpdateFront([PopObj;OffObj],NewFront,q);
                        FrontNo = [FrontNo(1:q-1),FrontNo(end),FrontNo(q:end-1)];
                        % Update the population
                        GreyWolves(q) = greyWolfNew(i_Gwo);
                        PopObj(q,:)   = OffObj;
                        GreyWolves(q).rank = FrontNo(q);
                        % Update the distances
                        Distance(q,:) = OffDis;
                        Distance(:,q) = OffDis';
                    end 
        end
              Iteration = Iteration+1;
              Archive = GreyWolves(vertcat(GreyWolves.rank) == 1);
              try 
                 l = length(Archive);
              catch
                  if l == 0
                      warning('Archives are not obtained!');
                  end
              end
              
%               if(Iteration == maxIter)
%                   % only obtain feasible non-dominated solutions
%                    archiveConsViol         = -vertcat(Archive.constraints);
%                    archiveInfeasible         = any(archiveConsViol>0, 2);
%                    Archive(archiveInfeasible) = [];
%               end

        % plot
%         archiveCost = vertcat(Archive.cost);
         costOld = vertcat(GreyWolves.cost);
% %          DrawOld(numObj,costOld,Archive,testPro,Iteration-1,maxIter,pf,metric,divImp);  
         DrawOldTogether(numObj,costOld,Archive,testPro,Iteration-1,maxIter,pf,metric,divImp); 
                             %%% 

              
%             costOld = vertcat(GreyWolves.cost);
%             costOpt = vertcat(Archive.cost); 
%             Draw(numObj,costOld,costOpt,testPro,Iteration,maxIter,pf,metric,divImp);

    end
    
    
end