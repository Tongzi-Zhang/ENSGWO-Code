function GreyWolves = UpdateGWOTest(LB,UB,gWolf,numGWO,maxIter,objFun,nonDM,consImp,divImp,testPro,dimVar,options,metric,adaptive,Zmin,Z,interval)
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
    

    
    % Whale_pos_ad = zeros(SearchAgents_no,K);
    %% Optimization Circle
    Iteration = 1;
    % evalParallel(options,numGWO);
    objPF = PF(testPro); 
    pf = objPF();
    PopCons = -vertcat(GreyWolves.constraints);
    MatingPool = TournamentSelection(2,numGWO,sum(max(0,PopCons),2));
    GWolf  = GreyWolves(MatingPool);
%     GWolf = GreyWolves;
    while Iteration <= maxIter % for each generation   
        a=2-Iteration*((2)/maxIter ); % a decreases linearly fron 2 to 0 in Eq. (2.3)
        options.currentGen = Iteration;
        % evalParallel(options,numGWO);
        fprintf('\n\n************************************************************\n');
        fprintf('*      Current generation %d / %d\n', options.currentGen, options.maxIter);
        fprintf('************************************************************\n');       

        %%%
        Ffront = GreyWolves((find(vertcat(GreyWolves.rank) == 1))); % first front
        Fcost = vertcat(Ffront.cost);
        [~,ia,ic] = unique(Fcost,'rows');
        Ffront = Ffront(ia);
        [LFfront,~] = size(Ffront);%Length of the first front
        
        for i_Gwo = 1:numGWO   %  (Moth for each individual)  
            
            clear residue2
            clear residue3
            %%%
            FConViols = -vertcat(Ffront.constraints);
%             indexAlpha = SelectLeader(Ffront);
%             indexBeta  = SelectLeader(Ffront);
%             indexDelta = SelectLeader(Ffront);
            indexAlpha = TournamentSelection(2,1,sum(max(0,FConViols),2));
            indexBeta = TournamentSelection(2,1,sum(max(0,FConViols),2));
            indexDelta = TournamentSelection(2,1,sum(max(0,FConViols),2));
            
            Alpha(1).position = Ffront(indexAlpha).position;
            Beta(1).position  = Ffront(indexBeta).position;
            Delta(1).position = Ffront(indexDelta).position;  
            
            if LFfront >1 
                counter = 0;
                for newi = 1:LFfront
                   if sum(Alpha(1).position ~= Ffront(newi).position) ~=0
                       counter = counter +1;
                       residue2(counter,1) = Ffront(newi);
                    end
                end
                residue2CV = -vertcat(residue2.constraints);
                positionBeta = TournamentSelection(2,1,sum(max(0,residue2CV),2));
                Beta(1).position = residue2(positionBeta).position;
             end
            
           % This scenario is the same if the second least crowded distance
           % has one solution, so the delta leader should be chosen from the
           % third least crowded distance.
                
            if LFfront > 2
                 counter=0;
                 for newi = 1:size(residue2)
                    if sum(Beta(1).position ~= residue2(newi).position)~=0
                        counter = counter+1;
                        residue3(counter,1) = residue2(newi);
                    end
                 end
                residue3CV = -vertcat(residue3.constraints);
                positionDelta = TournamentSelection(2,1,sum(max(0,residue3CV),2));
%                  Delta(1).position = residue3(positionDelta).position;
                 Alpha(1).position = residue3(positionDelta).position;
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

            greyWolfNew(i_Gwo).position=(X1+X2+X3)/3; 
%             greyWolfNew(i_Gwo).position=(X1/2+X2/3+X3/6);
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
            GreyWolves = Non_Dominated(fatherAndSon,nonDM,consImp,divImp,gwCons,adaptive,Iteration);
        elseif strcmp(divImp,'refPoints')
%             Zmin = min([Zmin;vertcat(fatherAndSon.cost);],[],1); %       2021220 add  delete
            gWFCons = -vertcat(greyWolfNew.constraints);
            Zmin = min([Zmin;vertcat(greyWolfNew(all(gWFCons<=0,2)).cost)],[],1);
            gwCons = vertcat(fatherAndSon.constraints);
            GreyWolves = Non_Dominated(fatherAndSon,nonDM,consImp,divImp,gwCons,adaptive,Iteration,numGWO,Z,Zmin,interval);
        end
        
        Iteration = Iteration+1;
        Archive = GreyWolves(find(vertcat(GreyWolves.rank) == 1));
%         Ffront = Archive;

        % plot
%         archiveCost = vertcat(Archive.cost);
         costOld = vertcat(GreyWolves.cost);
% %          DrawOld(numObj,costOld,Archive,testPro,Iteration-1,maxIter,pf,metric,divImp);  
         DrawOldTogether(numObj,costOld,Archive,testPro,Iteration-1,maxIter,pf,metric,divImp);    
         
                  %%% 
        PopCons = -vertcat(GreyWolves.constraints);
        MatingPool = TournamentSelection(2,numGWO,sum(max(0,PopCons),2));
        GWolf  = GreyWolves(MatingPool);
%         minFront = min(vertcat(GWolf.rank));
%         Ffront = GreyWolves((find(vertcat(GreyWolves.rank) == 1))); % first front
        
%         costOld = vertcat(GreyWolves.cost);
%         Draw(numObj,costOld,Archive,testPro,Iteration,maxIter,pf,metric,divImp);
      
    end

end
