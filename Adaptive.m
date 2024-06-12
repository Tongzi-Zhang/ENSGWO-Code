function Z = Adaptive(PopObj,Z,N,interval,Zmin)
% Addition and deletion of reference points

%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    M = size(PopObj,2);
    
    %% Addition of reference points
    rho   = Associate(PopObj,Z,Zmin);
    old_Z = [];
    while any(rho>=2) && ~isequal(old_Z,Z)
        old_Z = Z;
        for i = find(rho>=2)
            p = repmat(Z(i,:),M,1) - interval/M;
            p(logical(eye(M))) = p(logical(eye(M))) + interval;
            Z = [Z;p];
        end
        Z(any(Z<0,2),:) = [];
        [~,index]       = unique(round(Z*1e4)/1e4,'rows','stable');
        Z               = Z(index,:);
        rho = Associate(PopObj,Z,Zmin);
    end
    
    %% Deletion of reference points
    Z(intersect(N+1:size(Z,1),find(~rho)),:) = [];
end

function rho = Associate(PopObj,Z,Zmin)
% Associate each solution with one reference point

    %% Calculate the distance of each solution to each reference vector
    PopObj = PopObj - repmat(Zmin,size(PopObj,1),1);
    [N,M]  = size(PopObj);%当前offspring的种群数目
  
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
    
    NormP    = sqrt(sum(PopObj.^2,2));
    Cosine   = 1 - pdist2(PopObj,Z,'cosine');
    Distance = repmat(NormP,1,size(Z,1)).*sqrt(1-Cosine.^2);
    
    %% Associate each solution with its nearest reference point
    [~,pi] = min(Distance',[],1);
    
    %% Calculate the number of associated solutions of each reference point
    rho = hist(pi,1:size(Z,1));
end