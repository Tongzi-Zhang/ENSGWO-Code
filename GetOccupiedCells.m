%___________________________________________________________________%
%  Multi-Objective Grey Wolf Optimizer (MOGWO)                      %
%  Source codes demo version 1.0                                    %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper:                                                     %
%                                                                   %
%    S. Mirjalili, S. Saremi, S. M. Mirjalili, L. Coelho,           %
%    Multi-objective grey wolf optimizer: A novel algorithm for     %
%    multi-criterion optimization, Expert Systems with Applications,%
%    in press, DOI: http://dx.doi.org/10.1016/j.eswa.2015.10.039    %       %
%                                                                   %
%___________________________________________________________________%

% I acknowledge that this version of MOGWO has been written using
% a large portion of the following code:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MATLAB Code for                                                  %
%                                                                   %
%  Multi-Objective Particle Swarm Optimization (MOPSO)              %
%  Version 1.0 - Feb. 2011                                          %
%                                                                   %
%  According to:                                                    %
%  Carlos A. Coello Coello et al.,                                  %
%  "Handling Multiple Objectives with Particle Swarm Optimization," %
%  IEEE Transactions on Evolutionary Computation, Vol. 8, No. 3,    %
%  pp. 256-279, June 2004.                                          %
%                                                                   %
%  Developed Using MATLAB R2009b (Version 7.9)                      %
%                                                                   %
%  Programmed By: S. Mostapha Kalami Heris                          %
%                                                                   %
%         e-Mail: sm.kalami@gmail.com                               %
%                 kalami@ee.kntu.ac.ir                              %
%                                                                   %
%       Homepage: http://www.kalami.ir                              %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archive为非支配解  返回排序后的非重复索引位置的非支支配解索引 以及各个非支配解索引位置解的个数数组
function [occ_cell_index occ_cell_member_count Archive]=GetOccupiedCells(Archive)
    %获取索引位置
  
    alpha=0.1;  % Grid Inflation Parameter 作用？
    nGrid=10;   % Number of Grids per each Dimension 每个维度的网格数
    Archive_costs = vertcat(Archive.cost);
    % 获取目标的数值网格分布
    G=CreateHypercubes(Archive_costs,nGrid,alpha);
    % 获取每一个非支配解在空间网格中索引的位置GridIndex 以及各个维度维度的位置GridSubIndex
    for i=1:numel(Archive)
        [Archive(i).GridIndex Archive(i).GridSubIndex]=GetGridIndex(Archive(i),G);
    end
    
    GridIndices=[Archive.GridIndex];
    
    occ_cell_index=unique(GridIndices); %将GridIndices重复的值删除，并按照一定从小到大的顺序排列
    
    occ_cell_member_count=zeros(size(occ_cell_index));

    m=numel(occ_cell_index); %获取非重复非支配解的个数
    for k=1:m
        occ_cell_member_count(k)=sum(GridIndices==occ_cell_index(k)); 
        % sum函数布尔1中若相等，则在GridIndices中相对应元素位置布尔1,若存在相同，则对于布尔求和
    end
    
end