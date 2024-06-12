function GreyWolves = UpdateGWO(LB,UB,gWolf,numGWO,maxIter,objFun,nonDM,consImp,divImp,testPro,dimVar,options,metric,adaptive,FrontNo,Zmin,Z,interval,front)
%******************************* NSGWONF ************************************
%%  Function: NSGWONF()
%% Description: 
%   This function is used to implement MOGWO.% 
%% Syntax:  
%    reference point method:  GreyWolves = NSGWONF(LB,UB,GreyWolves,numGWO,numIter,objFun,nonDM,constraintImp,diversityImp,testPro,dimVar,options,metric,adaptive,Zmin,Z,interval);
%    crowded distance method: GreyWolves = NSGWONF(LB,UB,GreyWolves,numGWO,numIter,objFun,nonDM,constraintImp,diversityImp,testPro,dimVar,options,metric,adaptive);
%% Parameters:
%   Inputs:
%      LB:      the minimum of variables
%      UB:      the maximum of variables
%      numGWO:  the number of grey wolves
%      maxIter: the maximum of iterations
%      objFun:  the handle of test problem 
%      nonDM:   non-dominated sort method 
%      consImp: constraint methods
%      divImp:  selection methos
%      testPro: the name of test problem
%      dimVar:  the number of variables
%      options: defalut setting of the algorithm
%      metric:  metrics of algorithm
%      adaptive: whether adaptive method or not
%      Zmin:     ideal point
%      Z         reference points
%      interval: the difference between minimum and maximum reference
%      points
%   Outputs:
%      GreyWolves: updated grey wolves
% ---------------------------- Reference  ---------------------------------
%    S. Mirjalili, S. Saremi, S. M. Mirjalili, L. Coelho,          
%    Multi-objective grey wolf optimizer: A novel algorithm for     
%    multi-criterion optimization, Expert Systems with Applications,
%    in press, DOI: http://dx.doi.org/10.1016/j.eswa.2015.10.039   
% ----------------------------  Copyright  -------------------------------- 
%% Cited from MOGWO  All rights reserved.
%        Multi-Objective Grey Wolf Optimizer (MOGWO)                      
%           Source codes demo version 1.0                                    
%                                                                   
%           Developed in MATLAB R2011b(7.13)                                
%                                                                 
%           Author and programmer: Seyedali Mirjalili                        
%                                                                   
%                e-Mail: ali.mirjalili@gmail.com                           
%                 seyedali.mirjalili@griffithuni.edu.au            
%                                                                   
%       Homepage: http://www.alimirjalili.com 
%
%%          Modified by Zhang T.
%                
%   Revision: 1.0.0  Date: 2021-04-01
%*************************************************************************

   
    %% Algorithm Variables
    GreyWolves = gWolf;
    [~,numObj] = size(GreyWolves(1).cost);
    [~,GW] = EnsGwoOpt(); % initialize a grey wolf struct
    greyWolfNew= repmat(GW,numGWO,1); % Size: the number of grey wolves
    funStr     = func2str(objFun);
    clear GW;
    if isempty(Zmin)
        Zmin = ones(1,size(Z,2));
    end    
  
    %% Optimization Circle
    Iteration = 1;
    % evalParallel(options,numGWO);
    objPF = PF(testPro); 
    pf = objPF();
    numRef = numGWO;
%     PopCons = -vertcat(GreyWolves.constraints);
%     MatingPool = TournamentSelection(2,numGWO,sum(max(0,PopCons),2));
%     GWolf  = GreyWolves;
    while Iteration <= maxIter % for each generation   
        a=2-Iteration*((2)/maxIter ); % a decreases linearly fron 2 to 0 in Eq. (2.3)
        options.currentGen = Iteration;
        % evalParallel(options,numGWO);
        fprintf('\n\n************************************************************\n');
        fprintf('*      Current generation %d / %d\n', options.currentGen, options.maxIter);
        fprintf('************************************************************\n');
        
        
        GreyWolves = CrowdingDistance(GreyWolves,front);
       
        distance = vertcat(GreyWolves.distance);
        PopCons = -vertcat(GreyWolves.constraints);
%         cost = vertcat(GreyWolves.cost);
%         Div = DivFunc(GreyWolves);
%         maxCost     = max(cost,[],1);
%         Infeasible         = any(PopCons>0, 2);
%         cost(Infeasible,:) = repmat(maxCost,sum(Infeasible),1) + repmat(sum(max(0,PopCons(Infeasible,:)),2),1,numObj);
%         MatingPool = TournamentSelection(2,numGWO,FrontNo,-Div);
         MatingPool = TournamentSelection(2,numGWO,sum(max(0,PopCons),2));
%           GWolf  = GreyWolves(MatingPool);
          
%           MatingPool = MSEA(FrontNo,GreyWolves,numGWO,distance);
          GWolf  = GreyWolves(MatingPool);
           
        %%
        Ffront = GWolf((find(vertcat(GWolf.rank) == 1))); % first front
        if isempty(Ffront)
             Ffront = GreyWolves((find(vertcat(GreyWolves.rank) == 1)));
        end

        Fcost = vertcat(Ffront.cost);
        [~,ia,~] = unique(Fcost,'rows');
        Ffront = Ffront(ia);
        [LFfront,~] = size(Ffront);%Length of the first front
        
        for i_Gwo = 1:numGWO   %  (Moth for each individual)  
            
% %             clear residue2
% %             clear residue3

            indexAlpha = SelectLeaderNew1(Ffront,1);
            indexBeta  = SelectLeaderNew1(Ffront,2);
            indexDelta = SelectLeaderNew1(Ffront);
% 
%              Alpha(1) = SelectLeaderID(Ffront,beta);
%              Beta(1)  = SelectLeaderID(Ffront,beta);
%              Delta(1) = SelectLeaderID(Ffront,beta);

            Alpha(1).position = Ffront(indexAlpha).position;
            Beta(1).position  = Ffront(indexBeta).position;
            Delta(1).position = Ffront(indexDelta).position;  
%             
            if LFfront >1 
          
                for newi = 1:LFfront
                   if Alpha(1).position == Ffront(newi).position                    
                        residue2 = Ffront;
                        residue2(newi)=[];
                        break;
                    end
                end
%                 Beta(1) = SelectLeaderID(residue2,beta);
                  positionBeta =  SelectLeaderNew1(residue2,2);
                  Beta(1).position = residue2(positionBeta).position;
            end
                
           % This scenario is the same if the second least crowded distance
           % has one solution, so the delta leader should be chosen from the
           % third least crowded distance.
                
            if LFfront > 2
            
                 for newi = 1:size(residue2)
                    if Beta(1).position == residue2(newi).position
                       residue3 = residue2; 
                       residue3(newi) =[];
                       break;
                    end
                 end
%                 Delta(1)  = SelectLeaderID(residue3,beta);
                 positionDelta =  SelectLeaderNew1(residue3);
                 Delta(1).position = residue3(positionDelta).position;
            end


            r1 = rand(1,dimVar); % r1 is a random number in [0,1]
            r2 = rand(1,dimVar); % r2 is a random number in [0,1]       
            A1 = 2.*a.*r1-a;  % Eq. (2.3) in the paper
            C1 = 2.*r2;      % Eq. (2.4) in the paper 
            X1 = Alpha(1).position-A1.*abs(C1.*Alpha(1).position-GWolf(i_Gwo).position);

            r1 = rand(1,dimVar); % r1 is a random number in [0,1]
            r2 = rand(1,dimVar); % r2 is a random number in [0,1]       
            A2 = 2.*a.*r1-a;  % Eq. (2.3) in the paper
            C2 = 2.*r2;      % Eq. (2.4) in the paper 
            X2 = Beta(1).position-A2.*abs(C2.*Beta(1).position-GWolf(i_Gwo).position);


            r1 = rand(1,dimVar); % r1 is a random number in [0,1]
            r2 = rand(1,dimVar); % r2 is a random number in [0,1]       
            A3 = 2.*a.*r1-a;  % Eq. (2.3) in the paper
            C3 = 2.*r2;      % Eq. (2.4) in the paper 
            X3 = Delta(1).position-A3.*abs(C3.*Delta(1).position-GWolf(i_Gwo).position);
             
%             greyWolfNew(i_Gwo).position=(X1+X2+X3)./3;
%             greyWolfNew(i_Gwo).position=(X1./2+X2./3+X3./6); 
               greyWolfNew(i_Gwo).position=(X1.*0.98+X2.*0.01+X3.*0.01);
%             unusalUB = find(greyWolfNew(i_Gwo).position > UB);
%             unusalLB = find(greyWolfNew(i_Gwo).position < LB);
%             unusalIndex = [unusalUB unusalLB];
            greyWolfNew(i_Gwo).position = min(max(greyWolfNew(i_Gwo).position,LB),UB); 
%             greyWolfNew(i_Gwo).position(unusalIndex) = rand(1,length(unusalIndex)).*(UB(unusalIndex)-LB(unusalIndex))+LB(unusalIndex);
           
            switch funStr(1)
                case 'C'
                    [greyWolfNew(i_Gwo).cost,greyWolfNew(i_Gwo).constraints]= objFun(greyWolfNew(i_Gwo).position');
                    % Store by column way for later use
                    greyWolfNew(i_Gwo).cost = greyWolfNew(i_Gwo).cost';
                    greyWolfNew(i_Gwo).constraints = greyWolfNew(i_Gwo).constraints'; 
                otherwise
                    greyWolfNew(i_Gwo).cost = objFun(greyWolfNew(i_Gwo).position');
                     % Store by column way for later use
                    greyWolfNew(i_Gwo).cost = greyWolfNew(i_Gwo).cost';
            end 
            
        end
        
        greyWolfOld = GreyWolves;
        fatherAndSon = [greyWolfOld;greyWolfNew];

        for i_fSon =1:length(fatherAndSon)
            fatherAndSon(i_fSon).distance = 0;
            fatherAndSon(i_fSon).rank = 0;  
        end
       
        
        if strcmp(divImp,'crowDistance') || strcmp(divImp, 'Hypercubes')
            gwCons = vertcat(fatherAndSon.constraints);
            GreyWolves = Non_Dominated(fatherAndSon,nonDM,consImp,divImp,gwCons,Iteration);
        elseif strcmp(divImp,'refPoints')
%             Zmin = min([Zmin;vertcat(fatherAndSon.cost);],[],1); %       2021220 add  delete
            gWFCons = -vertcat(greyWolfNew.constraints);
            Zmin = min([Zmin;vertcat(greyWolfNew(all(gWFCons<=0,2)).cost)],[],1);
            gwCons = vertcat(fatherAndSon.constraints);
            [GreyWolves,front,~,FrontNo] = Non_Dominated(fatherAndSon,nonDM,consImp,divImp,gwCons,Iteration,numRef,Z,Zmin);
            if adaptive  == 1
               Z  = Adaptive(vertcat(GreyWolves.cost),Z,numGWO,interval,Zmin);
            end
        end
        
        Iteration = Iteration+1;
        Archive = GreyWolves(vertcat(GreyWolves.rank) == 1);
%         Ffront = Archive;

        % plot
%         archiveCost = vertcat(Archive.cost);
         costOld = vertcat(GreyWolves.cost);
% %          DrawOld(numObj,costOld,Archive,testPro,Iteration-1,maxIter,pf,metric,divImp);  
         DrawOldTogether(numObj,costOld,Archive,testPro,Iteration-1,maxIter,pf,metric,divImp);    
         


%            GWolf = GreyWolves;
%         minFront = min(vertcat(GWolf.rank));
%         Ffront = GreyWolves((find(vertcat(GreyWolves.rank) == 1))); % first front
        
%         costOld = vertcat(GreyWolves.cost);
%         Draw(numObj,costOld,Archive,testPro,Iteration,maxIter,pf,metric,divImp);
      
    end

end
