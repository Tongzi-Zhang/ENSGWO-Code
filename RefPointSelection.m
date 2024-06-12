function [GreyWolves,FrontNo,front] = RefPointSelection(GreyWolves,refUniValue,minRef,MaxFNo,FrontNo,numRef)
%******************************* RefPointSelection ************************************
%%  Function: RefPointSelection()
%% Description: 
%   This function is used to extract grey wolves as the same size of
%   original populations
% 
%% Syntax:  
%    GreyWolves= RefPointSelection(GreyWolves,refUniValue,minRef,MaxFNo,FrontNo,numRef)
%% Parameters:
%   Inputs:
%      GreyWolves:       grey wolves 
%      refUniValue:      reference points 
%      minRef:           ideal point
%      MaxFNo:           the maximum rank level
%      FrontNo:          the rank levels of each grey wolf
%      numRef:           the number of reference point
%   Outputs:
%      GreyWolves:   	 selected grey wolves      
%
%------------------------------- Reference --------------------------------
%. Kalyanmoy Deb.An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
% Nondominated Sorting Approach,Part I: Solving Problems With Box
% Constraints.IEEE Transactions on Evolutionary Computation,2014,18(4):
% 577-601.
%% Cited from PlatEMO All rights reserved.
%
%%              PlatEMO
%       Copyright (c) 2021 BIMK Group.
%
%% Modified by Zhang, T.
%                
%   Revision: 1.0.0  Date: 2021-04-01
%************************************************************************   
    % Obtain parameters
    Next = FrontNo < MaxFNo;
    
    %% Select the solutions in the last front 按照while的条件，返回的FrontNo==MaxFNo有需要去掉的个体或者正好
       %%% 05/12/2021
    if isempty(minRef)
        minRef = ones(1,size(refUniValue,2));
    end
    Last   = find(FrontNo == MaxFNo);%最后一层索引位置
    Choose = LastSelection(vertcat(GreyWolves(Next).cost),vertcat(GreyWolves(Last).cost),numRef-sum(Next),refUniValue,minRef);%获取最后一层选择情况
    Next(Last(Choose)) = true;%样本中最后一层选择位置确定
    % Population for next generation
    GreyWolves = GreyWolves(Next,:);%确定最后选择的种群个体
    FrontNo = vertcat(GreyWolves.rank);
    
    frontStr = struct('f',[]);
    front = repmat(frontStr,1,1);
                
    for i_front = 1:MaxFNo
        front(i_front).f = find(FrontNo' == i_front);
    end
   
end

%这一步是NSGA-III Algorithm 1第12步之后的操作;Zmin获取是Algorith 2第2步实现
function Choose = LastSelection(PopObj1,PopObj2,K,Z,Zmin)
% Select part of the solutions in the last front
    % Algorithm 2 第3步 Translate objectives：fj'=fj-zmin
    PopObj = [PopObj1;PopObj2] - repmat(Zmin,size(PopObj1,1)+size(PopObj2,1),1);
    [N,M]  = size(PopObj);%当前offspring的种群数目
    N1     = size(PopObj1,1);%offspring 1，2，...,l-1前沿的种群数目
    N2     = size(PopObj2,1);%offspring l前沿的种群数目
    NZ     = size(Z,1);% 参考点的数目或者说种群的数据，因为文献中建议最好种群数目与参考点的数目相同

    %% Normalization
    % Detect the extreme points， Algorithm 2第4步
    Extreme = zeros(1,M);
    w       = zeros(M)+1e-6+eye(M);
    for i = 1 : M
        [~,Extreme(i)] = min(max(PopObj./repmat(w(i,:),N,1),[],2));% Extreme(i)表示种群索引;max各个目标最大，然后获取各个种群中最小的那个
    end
    % Calculate the intercepts of the hyperplane constructed by the extreme
    % points and the axes；Algorithm 2 第6-7步
%     Hyperplane = PopObj(Extreme,:)\ones(M,1);%A的逆矩阵是 A1，则 B/A = B*A1，A\B = A1*B ,且Hyperplane是列向量,系数
    Hyperplane = pinv(PopObj(Extreme,:)) * ones(M,1); %防止奇异矩阵出现
    a = 1./Hyperplane;%获取每个方向的截距,其实这个截距也是减去理想点之后的，这里我称之为“理想截距”
    if any(isnan(a))%若a的元素为NaN（非数值），在对应位置上返回逻辑1（真），否则返回逻辑0（假）
        a = max(PopObj,[],1)';
    end
    % Normalization Algorithm 2第 end Normalization
    %a = a-Zmin';
    PopObj = PopObj./repmat(a',N,1);%如果a、b是矩阵，a./b就是a、b中对应的每个元素相除，得到一个新的矩阵；
 
    %% Associate each solution with one reference point
    % Calculate the distance of each solution to each reference vector
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');%cosine’ 针对向量,夹角余弦距离Cosine distance(‘cosine’)余弦相似度(余弦角)= 1-余弦距离 衡量两个个体之间的差异
    %pdist2 pairwise distance between two sets of observations
    %杰卡德距离Jaccard distance(‘jaccard’)Jaccard距离常用来处理仅包含非对称的二元(0-1)属性的对象。很显然，Jaccard距离不关心0-0匹配[1]。
    %夹角余弦距离Cosine distance(‘cosine’)与Jaccard距离相比，Cosine距离不仅忽略0-0匹配，而且能够处理非二元向量，即考虑到变量值的大小。
    %对这两者，距离与相似度和为一。
    %https://www.cnblogs.com/chaosimple/archive/2013/06/28/3160839.html

    Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);% Popobj距离*sinα，需要注意，每个参考点（种群）均与每一个种群个体（参考点）构成一个夹角，即两两关系
    %Distance = pdist2(PopObj,Z,'euclidean');%上面注释的两个语句也可以哦
    % Associate each solution with its nearest reference point
    [d,pi] = min(Distance',[],1);%距离最小，则参考点经过的line将与种群个体具有最小的距离，也就是说参考点接近种群个体，每一列是一个种群个体，离种群最近的参考点，pi是参考点索引

    %% Calculate the number of associated solutions except for the last front of each reference point
    rho = hist(pi(1:N1),1:NZ);% The number of different reference points that selected by grey wolves.

    %% Environmental selection
    Choose  = false(1,N2);
    Zchoose = true(1,NZ);
    % Select K solutions one by one
    while sum(Choose) < K
        % Select the least crowded reference point
        Temp = find(Zchoose);%索引
        Jmin = find(rho(Temp)==min(rho(Temp)));%所有参考点中与之关联个数最少的那个参考点，一般就是0个，目的是多样性
        j    = Temp(Jmin(randi(length(Jmin))));%随机选择一个个数最小的参考点
        I    = find(Choose==0 & pi(N1+1:end)==j);% 索引位置是根据现有元素来确定的，可能有多个
        % Then select one solution associated with this reference point
        if ~isempty(I)
            if rho(j) == 0
                [~,s] = min(d(N1+I));%有多个选择最小的那个
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;%所在的位置
            rho(j) = rho(j) + 1;%个数增加1，也就是说下次再选择它的概率将非常小了
        else
            Zchoose(j) = false;
        end
    end
end