function Score = IGD(PopObj,PF)
% <metric> <min>
% Inverted generational distance

%------------------------------- Reference --------------------------------
% C. A. Coello Coello and N. C. Cortes, Solving multiobjective optimization
% problems using an artificial immune system, Genetic Programming and
% Evolvable Machines, 2005, 6(2): 163-190.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
 
    Distance = min(pdist2(PF,PopObj),[],2);  % PF: m*n  PopObj:l*n  pdist2:m*l; min: calculate the minimum Euclidean-distance between individual in PF and PopObj set
    Score    = mean(Distance);
end

% pdist2 
% For example
% a=[2 3;4 5]
% a =
%      2     3
%      4     5
% b=[23 4; 4 5; 6 7; 8 9;]
% b =
%     23     4
%      4     5
%      6     7
%      8     9
% pdist2(b,a)
% ans =
%    21.0238   19.0263
%     2.8284         0           sqrt((23-2)^2+(4-3)^2)
%     5.6569    2.8284
%     8.4853    5.6569
% min(pdist2(b,a),[],2)
% ans =
%    19.0263
%          0
%     2.8284
%     5.6569
% a=[a; 6 7]
% a =
%      2     3
%      4     5
%      6     7
% pdist2(b,a)
% ans =
%    21.0238   19.0263   17.2627
%     2.8284         0    2.8284
%     5.6569    2.8284         0
%     8.4853    5.6569    2.8284