function [GreyWolves,objFun,numGWO,Zmin,Z,interval] = Initialization(gWolf,tPro,nGwo,nVar,lb,ub,diversityImp)
%******************************* Initialization ************************************
%% Function: NSGWONF()
%% Description: 
%   Initialize populations and obtain positions,costs, and cosntraints
%   of grey wolves.
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%   [GreyWolves,objFun,numGWO,Zmin,Z,interval] = Initialization(gWolf,tPro,nGwo,nVar,lb,ub,diversityImp)
%% Parameters:
%   Inputs:
%      gWolf:           individual of grey wolf
%      tPro:            the name of test problem
%      nGwo:            the number of grey wolf
%      nVar:            the number of variable
%      lb:              the minimum of variables
%      ub:              the maximum of variables
%      diversityImp:    selection methods
%   Outputs:
%      crowdDistance: [GreyWolves, objFun,numGwo] 
%      refPoints:     [GreyWolves,objFun,numGWO,Zmin,Z,interval]
%      GreyWolves:     populations of grey wolf
%      objFun:         the handle of test problem
%      numGwo:         the number of grey wolf
%      Zmin:           the ideal point
%      Z:              reference points
%      interval:       maximum and minimum difference after sorting the reference points 
%
%                             Zhang T.
%               Revision: 1.0.0     Date: 2021-04-01
%   
%*****************************************************************************
    
    % Obtain parameters
    numGWO = nGwo; % the number of grey wolf
    dimVar = nVar; % the number of variables
    LB=lb;         % minimum values of variables
    UB=ub;         % maximum values of variables
   
    % Obtain test problem and the number of obj ective functions
    [objFun,numObj] = TestProblem(tPro);
    funStr          = func2str(objFun);
    Zmin            = zeros(1,numObj);
    
    % Construct uniform reference points
    if strcmp(diversityImp,'refPoints')
        %%       [W,N] = UniformPoint(286,10,'ILD')
         %       [W,N] = UniformPoint(102,10,'MUD')
         %       [W,N] = UniformPoint(1000,3,'grid')
         %       [W,N] = UniformPoint(103,10,'Latin')
        [Z, N]      = UniformPoint(numGWO,numObj); 
        numGWO      = N;
        Z           = sortrows(Z);
        interval    = Z(1,end) - Z(2,end);
    end
     
    % Initialize grey wolves
    GreyWolves =  repmat(gWolf,[numGWO,1]);
    for i_Gwo = 1 : numGWO
        
    % Randomly get varible values of grey wolves
%         for j_dimVar =1 : dimVar
%         % returns an random numbers generated from the continuous uniform distributions with 
%         % lower and upper endpoints specified by LB and UB, respectively
%           GreyWolves(i_Gwo).position(j_dimVar) = unifrnd(LB(j_dimVar),UB(j_dimVar),1);  % have a littel difference with the following way
% %             GreyWolves(i_Gwo).position(j_dimVar) = LB(j_dimVar)+(UB(j_dimVar)-LB(j_dimVar))*rand(1);
%         end % end for j_dimVar
        GreyWolves(i_Gwo).position = unifrnd(LB(1:dimVar),UB(1:dimVar),1,dimVar);
%        GreyWolves(i_Gwo).position = LB(1:dimVar)+(UB(1:dimVar)-LB(1:dimVar)).*rand(1,dimVar);
        % Whether the problem has constraints or not, if yes, tPro(1) = C
        switch funStr(1)
            case 'C' 
                [GreyWolves(i_Gwo).cost, GreyWolves(i_Gwo).constraints] = objFun(GreyWolves(i_Gwo).position');
                % Store by column way for later use
                GreyWolves(i_Gwo).cost                                  = GreyWolves(i_Gwo).cost';
                GreyWolves(i_Gwo).constraints                           = GreyWolves(i_Gwo).constraints';
            otherwise
                GreyWolves(i_Gwo).cost = objFun(GreyWolves(i_Gwo).position');
                 % Store by column way for later use
                GreyWolves(i_Gwo).cost = GreyWolves(i_Gwo).cost';
        end 
       
    end % end for i_Gwo
 
   PopCons = -vertcat(GreyWolves.constraints);
   %  Create the ideal point which will be used to the normalization of objective values
   if strcmp(diversityImp,'refPoints')
        Pop_objs    = vertcat(GreyWolves.cost);  
        Zmin        = min(Pop_objs(all(PopCons<=0,2),:),[],1); % Zmin: ideal point
   end
%         %%% 

%     MatingPool = TournamentSelection(2,numGWO,sum(max(0,PopCons),2));
%     GreyWolves  = GreyWolves(MatingPool);

end
