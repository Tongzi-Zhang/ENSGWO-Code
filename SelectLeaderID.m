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
% rep 为非支配解集合  
function rep_h=SelectLeaderID(rep,beta)
    if nargin<2
        beta=1;
    end
    % 获取排序的网格索引  返回排序后的非重复支配解索引 以及各个支配解索引位置解的个数数组
    [occ_cell_index occ_cell_member_count rep]=GetOccupiedCells(rep);

    p=occ_cell_member_count.^(-beta); %相同支配解的个数越多，概率越小  least被选择的概率越大
    p=p/sum(p); % 概率对应各非支配解索引
    % 轮盘赌方法目的返回一个随机的支配解位置（在所有排序非重复支配解中的位置） 
    selected_cell_index=occ_cell_index(RouletteWheelSelection(p));% 进而返回非支配支配解的索引
    % 原网格支配解集各解的索引
    GridIndices=[rep.GridIndex];
    % 所选择的非支配解在非支配解解集网格索引数组中的位置=RouletteWheelSelection
    selected_cell_members=find(GridIndices==selected_cell_index); % 获取非支配解，若多个相同解则为数组，这个位置是在GridIndices中的
    
    n=numel(selected_cell_members); % 获取随机选取非支配解的个数
    
    selected_memebr_index=randi([1 n]); %伪随机整数，区间在[1 n]
    
    h=selected_cell_members(selected_memebr_index); % 随机获取与随机解的相同的任一非支配解所在的位置
    
    rep_h=rep(h); %获取该随机位置上 与随机非支配解相同的个体
end