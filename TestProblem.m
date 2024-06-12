function [fobj, numObj]= TestProblem(name)
%******************************* TestProblem ************************************
%% Function: TestProblem()
%% Description: 
%   To calculate objectives and constraints of the test problem
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%   [fobj, numObj]= TestProblem(name)
%% Parameters:
%   Inputs:
%      name:    the name of test problem
%      
%   Outputs:
%      fobj:    the handle of test problem
%      numObj:  the number of objective function
%% Notes:
%   If you add test problem in the algorithm, please follow these two points:  
%   1. Don't forget y=y' and c=c' in the end of functions
%   2. c should be larger than 0
%
% ---------------------------- Reference  ---------------------------------
% 1. Himanshu Jain and Kalyanmoy Deb.An Evolutionary Many-Objective 
% Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach,
% Part II: Handling Constraints and Extending to an Adaptive Approach.IEEE
% Transactions on Evolutionary Computation,2014,18(4):602-622
% 2. Qingfu Zhang, Aimin Zhou, Shizheng Zhaoy, Ponnuthurai Nagaratnam Suganthany, Wudong Liu and
% Santosh Tiwariz.Multiobjective optimization Test Instances for the CEC 2009 Special Session and
% Competition. Technical Report CES-487,2009
% 3. Kalyanmoy Deb,Lothar Thiele, Marco Laumanns and Eckart Zitzler.Scalable Test Problems 
% for Evolutionary Multi-Objective Optimization.TIK-Technical Report, 2001, 112
% 4. S. Huband, P. Hingston, L. Barone, and L. While, A review of
% multiobjective test problems and a scalable test problem toolkit, IEEE
% Transactions on Evolutionary Computation, 2006, 10(5): 477-506. 
%
%% ----------------------------  Copyright  -------------------------------- 
%% Cited from NSPM and PlatEMO All rights reserved.
%%              NSPM
%           LSSSSWC, NWPU
%    Revision: 1.1  Date: 2011-07-25
%
%%              PlatEMO
%       Copyright (c) 2021 BIMK Group.
%% Modified by Zhang, T.
%                
%   Revision: 1.0.0  Date: 2021-04-01
%************************************************************************


    switch name
        case 'UF1'
            fobj = @UF1;
            numObj = 2;
        case 'UF2'
            fobj = @UF2; 
            numObj = 2;
        case 'UF3'
            fobj = @UF3;
            numObj = 2;
        case 'UF4'
            fobj = @UF4;
            numObj = 2;
        case 'UF5'
            fobj = @UF5;
            numObj = 2;
        case 'UF6'
            fobj = @UF6;
            numObj = 2;
        case 'UF7'
            fobj = @UF7;
            numObj = 2;
        case 'UF8'
            fobj = @UF8;
            numObj = 3;
        case 'UF9'
            fobj = @UF9;
            numObj = 3;
        case 'UF10'
            fobj = @UF10;
            numObj = 3;
        case 'CF1'
            fobj = @CF1;
            numObj = 2;
        case 'CF2'
            fobj = @CF2;
            numObj = 2;
        case 'CF3'
            fobj = @CF3;
            numObj = 2;
        case 'CF4'
            fobj = @CF4;
            numObj = 2;
        case 'CF5'
            fobj = @CF5;
            numObj = 2;
        case 'CF6'
            fobj = @CF6;
            numObj = 2;
        case 'CF7'
            fobj = @CF7;
            numObj = 2;
        case 'CF8'
            fobj = @CF8;
            numObj = 3;
        case 'CF9'
            fobj = @CF9;
            numObj = 3;
        case 'CF10'
            fobj = @CF10;
            numObj = 3;      
        case 'ZDT1'
            fobj = @ZDT1;
            numObj = 2; 
        case 'ZDT2'
            fobj = @ZDT2;
            numObj = 2;        
        case 'ZDT3'
            fobj = @ZDT3;
            numObj = 2;         
        case 'ZDT4'
            fobj = @ZDT4;
            numObj = 2;
        case 'ZDT5'
            fobj = @ZDT5;
            numObj = 2;
        case 'ZDT6'
            fobj = @ZDT6;
            numObj = 2;      
        case 'C1_DTLZ1'
            fobj = @C1_DTLZ1;
            numObj = 3;
        case 'C2_DTLZ2'
            fobj = @C2_DTLZ2;
            numObj = 3;
        case 'C1_DTLZ3'
            fobj = @C1_DTLZ3;
            numObj = 3;
        case  'C3_DTLZ4'
            fobj = @C3_DTLZ4;
            numObj = 3;
        case 'WFG1'
            fobj = @WFG1;
            numObj = 3;
        case 'WFG2'
            fobj = @WFG2;
            numObj = 3;
        case 'WFG3'
            fobj = @WFG3;
            numObj = 3;  
        case 'WFG4'
            fobj = @WFG4;
            numObj = 3;
        case 'WFG5'
            fobj = @WFG5;
            numObj = 3;
        case 'WFG6'
            fobj = @WFG6;
            numObj = 3;
        case 'WFG7'
            fobj = @WFG7;
            numObj = 3;
        case 'WFG8'
            fobj = @WFG8;
            numObj = 3; 
        case 'WFG9'
            fobj = @WFG9;
            numObj = 3;  
        case 'DTLZ1'
            fobj = @DTLZ1;
            numObj = 3;
        case 'DTLZ2'
            fobj = @DTLZ2;
            numObj = 3;
        case 'DTLZ3'
            fobj = @DTLZ3;
            numObj = 3;
        case 'DTLZ4'
            fobj = @DTLZ4;
            numObj = 3;
        case 'DTLZ5'
            fobj = @DTLZ5;
            numObj = 3;
        case 'DTLZ6'
            fobj = @DTLZ6;
            numObj = 3;
        case 'DTLZ7'
            fobj = @DTLZ7;
            numObj = 3;
        case 'DTLZ8'
            fobj = @CDTLZ8;
            numObj = 3;
        case 'DTLZ9'
            fobj = @CDTLZ9;
            numObj = 2;
        case 'IMOP1'
            fobj = @IMOP1;
            numObj = 2;
         case 'IMOP2'
            fobj = @IMOP2;
            numObj = 2;        
        case 'IMOP3'
            fobj = @IMOP3;
            numObj = 2;        
        case 'IMOP4'
            fobj = @IMOP4;
            numObj = 3;
        case 'IMOP5'
            fobj = @IMOP5;
            numObj = 3; 
        case 'IMOP6'
            fobj = @IMOP6;
            numObj = 3; 
        case 'IMOP7'
            fobj = @IMOP7;
            numObj = 3; 
        case 'IMOP8'
            fobj = @IMOP8;
            numObj = 3;
        case 'DC1_DTLZ1'
            fobj = @DC1_DTLZ1;
            numObj = 3;
        case 'LIRCMOP1'
            fobj = @CLIRCMOP1;
            numObj =2;           
        case 'LIRCMOP6'
            fobj = @CLIRCMOP6;
            numObj =2;
    end
end

%% Note: 
%   x,c, and y are columnwise, the imput x must be inside the search space and it could be a matrix
function [y, c] = CLIRCMOP1(x)
    x = x';
    x_odd       = x(:,3:2:end);
    x_even      = x(:,2:2:end);
    g_1         = sum((x_odd - sin(0.5 * pi * x(:,1))).^2,2);
    g_2         = sum((x_even - cos(0.5 * pi * x(:,1))).^2,2);
    y(:,1) = x(:,1) + g_1;
    y(:,2) = 1 - x(:,1) .^ 2 + g_2;
    x_odd       = x(:,3:2:end);
    x_even      = x(:,2:2:end);
    g_1         = sum((x_odd - sin(0.5 * pi * x(:,1))).^2,2);
    g_2         = sum((x_even - cos(0.5 * pi * x(:,1))).^2,2);
    c(:,1) = (0.51 - g_1).*(g_1 - 0.5);
    c(:,2) = (0.51 - g_2).*(g_2 - 0.5);
    y = y';
    c = c';
 end


function [y, c] = CLIRCMOP6(x)
    x = x';
    variable_length = size(x,2);
    popsize         = size(x,1);
    sum1            = zeros(popsize,1);
    sum2            = zeros(popsize,1);
    for j = 2 : variable_length
        if mod(j,2) == 1
            sum1 = sum1+(x(:,j)-sin((0.5*j/variable_length*pi)*x(:,1))).^2;
        else
            sum2 = sum2+(x(:,j)-cos((0.5*j/variable_length*pi)*x(:,1))).^2;
        end
    end
    gx          = 0.7057;
    y(:,1) = x(:,1)+10*sum1+gx;
    y(:,2) = 1-x(:,1).^2+10.*sum2+gx;
    c = -Constraint(y);
    y = y';
    c = c';
end



function [y,c] = CDC1_DTLZ1(x)
    x = x';
    M = 3;
    D = M+4;
    g      = 100*(D-M+1+sum((x(:,M:end)-0.5).^2-cos(20.*pi.*(x(:,M:end)-0.5)),2));
    y = 0.5*repmat(1+g,1,M).*fliplr(cumprod([ones(size(x,1),1),x(:,1:M-1)],2)).*[ones(size(x,1),1),1-x(:,M-1:-1:1)];
    c =cos(3*pi*x(:,1)) - 0.5 ;
    y = y';
    c = c';
end



function y = IMOP1(x)
    a1 = 0.05;  % Parameter a1
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:K),2).^a1;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    y(:,1) = g + cos(y1*pi/2).^8;
    y(:,2) = g + sin(y1*pi/2).^8;
    y = y';
end

function y = IMOP2(x)
    a1 = 0.05;  % Parameter a1
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:K),2).^a1;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    y(:,1) = g + cos(y1*pi/2).^0.5;
    y(:,2) = g + sin(y1*pi/2).^0.5;
    y = y';
end
function y = IMOP3(x)
    a1 = 0.05;  % Parameter a1
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:K),2).^a1;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    y(:,1) = g + (1+cos(y1*pi*10)/5-y1);
    y(:,2) = g + y1;
    y = y';
end
function y = IMOP4(x)
    a1 = 0.05;  % Parameter a1
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:K),2).^a1;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    y(:,1) = (1+g).*(y1);
    y(:,2) = (1+g).*(y1+sin(10*pi*y1)/10);
    y(:,3) = (1+g).*(1-y1);
    y = y';
end
function y = IMOP5(x)
    a1 = 0.05;  % Parameter a1
    a2 = 10;    % Parameter a2
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:2:K),2).^a1;
    y2 = mean(x(:,2:2:K),2).^a2;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    y(:,1) = 0.4*cos(pi*ceil(y1*8)/4) + 0.1*y2.*cos(16*pi*y1);
    y(:,2) = 0.4*sin(pi*ceil(y1*8)/4) + 0.1*y2.*sin(16*pi*y1);
    y(:,3) = 0.5 - sum(y(:,1:2),2);
    y = y + repmat(g,1,3);
    y = y';
end
function y = IMOP6(x)
    a1 = 0.05;  % Parameter a1
    a2 = 10;    % Parameter a2
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:2:K),2).^a1;
    y2 = mean(x(:,2:2:K),2).^a2;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    r  = max(0,min(sin(3*pi*y1).^2,sin(3*pi*y2).^2)-0.05);
    y(:,1) = (1+g).*y1 + ceil(r);
    y(:,2) = (1+g).*y2 + ceil(r);
    y(:,3) = (0.5+g).*(2-y1-y2) + ceil(r);
    y = y';
end
function y = IMOP7(x)
    a1 = 0.05;  % Parameter a1
    a2 = 10;    % Parameter a2    
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:2:K),2).^a1;
    y2 = mean(x(:,2:2:K),2).^a2;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    y(:,1) = (1+g).*cos(y1*pi/2).*cos(y2*pi/2);
    y(:,2) = (1+g).*cos(y1*pi/2).*sin(y2*pi/2);
    y(:,3) = (1+g).*sin(y1*pi/2);
    r = min(min(abs(y(:,1)-y(:,2)),abs(y(:,2)-y(:,3))),abs(y(:,3)-y(:,1)));
    y = y + repmat(10*max(0,r-0.1),1,3);
    y = y';
end
function y = IMOP8(x)
    a1 = 0.05;  % Parameter a1
    a2 = 10;    % Parameter a2
    K  = 5;     % Parameter K
    x = x';
    y1 = mean(x(:,1:2:K),2).^a1;
    y2 = mean(x(:,2:2:K),2).^a2;
    g  = sum((x(:,K+1:end)-0.5).^2,2);
    y(:,1) = y1;
    y(:,2) = y2;
    y(:,3) = (1+g).*(3-sum(y(:,1:2)./(1+repmat(g,1,2)).*(1+sin(19*pi.*y(:,1:2))),2));
    y = y';
end


function y = DTLZ1(x)
   M = 3;
   x = x';
   D = M+4;
%    g      = 100*(D-M+1+sum((x(:,M:end)-0.5).^2-cos(20.*pi.*(x(:,M:end)-0.5)),2));
   g      = (D-M+1+sum((x(:,M:end)-0.5).^2-cos(20.*pi.*(x(:,M:end)-0.5)),2));
   y = 0.5*repmat(1+g,1,M).*fliplr(cumprod([ones(size(x,1),1),x(:,1:M-1)],2)).*[ones(size(x,1),1),1-x(:,M-1:-1:1)];
   y = y';
end

function y = DTLZ2(x)
   M = 3;
   x = x';
   g      = sum((x(:,M:end)-0.5).^2,2);
   y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];
   y = y';
end

function y = DTLZ3(x)
   M = 3;
   x = x';
   D = M +9;
%    g      = 100*(D-M+1+sum((x(:,M:end)-0.5).^2-cos(20.*pi.*(x(:,M:end)-0.5)),2));
   g      = (D-M+1+sum((x(:,M:end)-0.5).^2-cos(20.*pi.*(x(:,M:end)-0.5)),2));
   y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(x,1),1),cos(x(:,1:M-1)*pi/2)],2)).*[ones(size(x,1),1),sin(x(:,M-1:-1:1)*pi/2)];
   y = y';
end
function y = DTLZ4(x)
   M = 3;
   x = x';
   x(:,1:M-1) = x(:,1:M-1).^100;
   g      = sum((x(:,M:end)-0.5).^2,2);
   y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];
   y = y';
end
function y = DTLZ5(x)
   M = 3;
   x = x';
   g    = sum((x(:,M:end)-0.5).^2,2);
   Temp = repmat(g,1,M-2);
   x(:,2:M-1) = (1+2*Temp.*x(:,2:M-1))./(2+2*Temp);
   y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];
   y = y';
end
function y = DTLZ6(x)
   M = 3;
   x = x';
   g    = sum(x(:,M:end).^0.1,2);
   Temp = repmat(g,1,M-2);
   x(:,2:M-1) = (1+2*Temp.*x(:,2:M-1))./(2+2*Temp);
   y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(x(:,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(x(:,M-1:-1:1)*pi/2)];
   y = y';
end

function y = DTLZ7(x)
   M = 3;
   x = x';
   y = zeros(size(x,1),M);
   g      = 1+9*mean(x(:,M:end),2);
   y(:,1:M-1) = x(:,1:M-1);
   y(:,M)     = (1+g).*(M-sum(y(:,1:M-1)./(1+repmat(g,1,M-1)).*(1+sin(3*pi.*y(:,1:M-1))),2));
   y = y';
end
function [y ,c] = CDTLZ8(x)
    M = 3;
    x = x';
    D = 10 * M;
    y = zeros(size(x,1),M);
    for m = 1 : M
       y(:,m) = mean(x(:,(m-1)*D/M+1:m*D/M),2);
    end

    c = zeros(size(y,1),M);
    c(:,1:M-1) = repmat(y(:,M),1,M-1)+4*y(:,1:M-1)-1;
    if M == 2
        c(:,M) = 0;
    else
        minValue = sort(y(:,1:M-1),2);
        c(:,M) = 2*y(:,M) + sum(minValue(:,1:2),2)-1;
    end
    y = y';
    c = c';
end
function [y ,c ] = CDTLZ9(x)
    M = 2;
    D = 10 * M;
    x = x';
    x = x.^0.1;
    y = zeros(size(x,1),M);
    for m = 1 : M
      y(:,m) = sum(x(:,(m-1)*D/M+1:m*D/M),2);
    end
    c =  repmat(y(:,M).^2,1,M-1)+ y(:,1:M-1).^2 -1;
    y = y';
    c = c';
end


function y = WFG1(x)
    [dim, num] = size(x);
    x = x';
    M = 3;
    K = M-1; % a mulitple of M-1
    L = dim - K;
    dim = 1; % is a distance scaling constant
    S = 2 : 2 : 2*M;
    A = ones(1,M-1);

    z01 = x./repmat(2:2:size(x,2)*2,num,1);

    t1 = zeros(num,K+L);
    t1(:,1:K)     = z01(:,1:K);
    t1(:,K+1:end) = s_linear(z01(:,K+1:end),0.35);

    t2 = zeros(num,K+L);
    t2(:,1:K)     = t1(:,1:K);
    t2(:,K+1:end) = b_flat(t1(:,K+1:end),0.8,0.75,0.85);

    t3 = zeros(num,K+L);
    t3 = b_poly(t2,0.02);

    t4 = zeros(num,M);
    for i = 1 : M-1
       t4(:,i) = r_sum(t3(:,(i-1)*K/(M-1)+1:i*K/(M-1)),2*((i-1)*K/(M-1)+1):2:2*i*K/(M-1));
     end
    t4(:,M) = r_sum(t3(:,K+1:K+L),2*(K+1):2:2*(K+L));

    x = zeros(num,M);
    for i = 1 : M-1
        x(:,i) = max(t4(:,M),A(i)).*(t4(:,i)-0.5)+0.5;
    end
     x(:,M) = t4(:,M);

     h      = convex(x);
     h(:,M) = mixed(x);
     y = repmat(dim*x(1,M),1,M) + repmat(S,num,1).*h;
     y = y';
    
end

function y = WFG2(x)
    [dim, num] = size(x);
    x = x';
    M = 3;
    K = M-1;
    L = dim - K;% DISTANCE-RELATED PARAMETERS 
    dim = 1;
    S = 2 : 2 : 2*M;
    A = ones(1,M-1);

    z01 = x./repmat(2:2:size(x,2)*2,num,1);
            
    t1 = zeros(num,K+L);
    t1(:,1:K)     = z01(:,1:K);
    t1(:,K+1:end) = s_linear(z01(:,K+1:end),0.35);

    t2 = zeros(num,K+L/2);
    t2(:,1:K) = t1(:,1:K);
            % Same as <t2(:,i)=r_nonsep(t1(:,K+2*(i-K)-1:K+2*(i-K)),2)>
    t2(:,K+1:K+L/2) = (t1(:,K+1:2:end) + t1(:,K+2:2:end) + 2*abs(t1(:,K+1:2:end)-t1(:,K+2:2:end)))/3;
            % ---------------------------------------------------------
            
    t3 = zeros(num,M);
    for i = 1 : M-1
        t3(:,i) = r_sum(t2(:,(i-1)*K/(M-1)+1:i*K/(M-1)),ones(1,K/(M-1)));
    end
    t3(:,M) = r_sum(t2(:,K+1:K+L/2),ones(1,L/2));

    x = zeros(num,M);
    for i = 1 : M-1
       x(:,i) = max(t3(:,M),A(:,i)).*(t3(:,i)-0.5)+0.5;
    end
    x(:,M) = t3(:,M);

    h      = convex(x);
    h(:,M) = disc(x);
    y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;
    y = y';

end


function y = WFG3(x)
    [dim, num] = size(x);
    x = x';
    M = 3;
    K = M-1;
    L = dim - K;
    dim = 1;
    S = 2 : 2 : 2*M;
    A = [1,zeros(1,M-2)];

    z01 = x./repmat(2:2:size(x,2)*2,num,1);

    t1 = zeros(num,K+L);
    t1(:,1:K)     = z01(:,1:K);
    t1(:,K+1:end) = s_linear(z01(:,K+1:end),0.35);

    t2 = zeros(num,K+L/2);
    t2(:,1:K) = t1(:,1:K);
    % Same as <t2(:,i)=r_nonsep(t1(:,K+2*(i-K)-1:K+2*(i-K)),2)>
    t2(:,K+1:K+L/2) = (t1(:,K+1:2:end) + t1(:,K+2:2:end) + 2*abs(t1(:,K+1:2:end)-t1(:,K+2:2:end)))/3;
    % ---------------------------------------------------------
            
    t3 = zeros(num,M);
    for i = 1 : M-1
       t3(:,i) = r_sum(t2(:,(i-1)*K/(M-1)+1:i*K/(M-1)),ones(1,K/(M-1)));
    end
    t3(:,M) = r_sum(t2(:,K+1:K+L/2),ones(1,L/2));

    x = zeros(num,M);
    for i = 1 : M-1
        x(:,i) = max(t3(:,M),A(:,i)).*(t3(:,i)-0.5)+0.5;
    end
    x(:,M) = t3(:,M);

    h      = linear(x);
    y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;
    y = y';
end


function y = WFG4(x)
        [dim, num] = size(x);
        x = x';
        M = 3;
        K = M-1;
            L = dim - K;
            dim = 1;
            S = 2 : 2 : 2*M;
            A = ones(1,M-1);

            z01 = x./repmat(2:2:size(x,2)*2,num,1);

            t1 = zeros(num,K+L);
            t1 = s_multi(z01,30,10,0.35);

            t2 = zeros(num,M);
            for i = 1 : M-1
                t2(:,i) = r_sum(t1(:,(i-1)*K/(M-1)+1:i*K/(M-1)),ones(1,K/(M-1)));
            end
            t2(:,M) = r_sum(t1(:,K+1:K+L),ones(1,L));

            x = zeros(num,M);
            for i = 1 : M-1
                x(:,i) = max(t2(:,M),A(:,i)).*(t2(:,i)-0.5)+0.5;
            end
            x(:,M) = t2(:,M);

            h = concave(x);
            y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;
            y = y';
end


function  y = WFG5(x)
        [dim, num] = size(x);
        x = x';
        M = 3;
        K = M-1;
        L = dim - K;
            dim = 1;
            S = 2 : 2 : 2*M;
            A = ones(1,M-1);

            z01 = x./repmat(2:2:size(x,2)*2,num,1);
            
            t1 = zeros(num,K+L);
            t1 = s_decept(z01,0.35,0.001,0.05);

            t2 = zeros(num,M);
            for i = 1 : M-1
                t2(:,i) = r_sum(t1(:,(i-1)*K/(M-1)+1:i*K/(M-1)),ones(1,K/(M-1)));
            end
            t2(:,M) = r_sum(t1(:,K+1:K+L),ones(1,L));

            x = zeros(num,M);
            for i = 1 : M-1
                x(:,i) = max(t2(:,M),A(:,i)).*(t2(:,i)-0.5)+0.5;
            end
            x(:,M) = t2(:,M);

            h = concave(x);
            y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;       
            y = y';
end


function  y = WFG6(x)
        [dim, num] = size(x);
        x = x';
        M = 3;
        K = M-1;
        L = dim - K;
        dim = 1;
        S = 2 : 2 : 2*M;
        A = ones(1,M-1);

        z01 = x./repmat(2:2:size(x,2)*2,num,1);
            
        t1 = zeros(num,K+L);
        t1(:,1:K)     = z01(:,1:K);
        t1(:,K+1:end) = s_linear(z01(:,K+1:end),0.35);

        t2 = zeros(num,M);
        for i = 1 : M-1
           t2(:,i) = r_nonsep(t1(:,(i-1)*K/(M-1)+1:i*K/(M-1)),K/(M-1));
        end
        % Same as <t2(:,M)=r_nonsep(t1(:,K+1:end),L)>
        SUM = zeros(num,1);
        for i = K+1 : K+L-1
          for j = i+1 : K+L
             SUM = SUM + abs(t1(:,i)-t1(:,j));
          end
        end
        t2(:,M) = (sum(t1(:,K+1:end),2)+SUM*2)/ceil(L/2)/(1+2*L-2*ceil(L/2));
            % -------------------------------------------

        x = zeros(num,M);
        for i = 1 : M-1
            x(:,i) = max(t2(:,M),A(:,i)).*(t2(:,i)-0.5)+0.5;
        end
        x(:,M) = t2(:,M);

        h = concave(x);
        y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;
        y = y';
        
end


function  y = WFG7(x)
        [dim, num] = size(x);
        x = x';
        M = 3;
        K = M-1;
        L = dim - K;
        dim = 1;
        S = 2 : 2 : 2*M;
        A = ones(1,M-1);

        z01 = x./repmat(2:2:size(x,2)*2,num,1);
            
        t1 = zeros(num,K+L);
        % Same as <t1(:,i)=b_param(z01(:,i),r_sum(z01(:,i+1:end),ones(1,K+L-i)),0.98/49.98,0.02,50)>
        Y = (fliplr(cumsum(fliplr(z01),2))-z01)./repmat(K+L-1:-1:0,num,1);
        t1(:,1:K) = z01(:,1:K).^(0.02+(50-0.02)*(0.98/49.98-(1-2*Y(:,1:K)).*abs(floor(0.5-Y(:,1:K))+0.98/49.98)));
        % ------------------------------------------------------------------------------------------
        t1(:,K+1:end) = z01(:,K+1:end);

        t2 = zeros(num,K+L);
        t2(:,1:K)     = t1(:,1:K);
        t2(:,K+1:end) = s_linear(t1(:,K+1:end),0.35);

        t3 = zeros(num,M);
        for i = 1 : M-1
           t3(:,i) = r_sum(t2(:,(i-1)*K/(M-1)+1:i*K/(M-1)),ones(1,K/(M-1)));
        end
        t3(:,M) = r_sum(t2(:,K+1:K+L),ones(1,L));

        x = zeros(num,M);
        for i = 1 : M-1
          x(:,i) = max(t3(:,M),A(:,i)).*(t3(:,i)-0.5)+0.5;
        end
        x(:,M) = t3(:,M);

        h = concave(x);
        y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;
        y = y';
end


function  y = WFG8(x)
    [dim, num] = size(x);
    x = x';
    M = 3;
    K = M-1;
    L = dim - K;
    dim = 1;
    S = 2 : 2 : 2*M;
    A = ones(1,M-1);

    z01 = x./repmat(2:2:size(x,2)*2,num,1);
            
    t1 = zeros(num,K+L);
    t1(:,1:K) = z01(:,1:K);
    % Same as <t1(:,i)=b_param(z01(:,i),r_sum(z01(:,1:i-1),ones(1,i-1)),0.98/49.98,0.02,50)>
    Y = (cumsum(z01,2)-z01)./repmat(0:K+L-1,num,1);
    t1(:,K+1:K+L) = z01(:,K+1:K+L).^(0.02+(50-0.02)*(0.98/49.98-(1-2*Y(:,K+1:K+L)).*abs(floor(0.5-Y(:,K+1:K+L))+0.98/49.98))); 
    % --------------------------------------------------------------------------------------

    t2 = zeros(num,K+L);
    t2(:,1:K)     = t1(:,1:K);
    t2(:,K+1:end) = s_linear(t1(:,K+1:end),0.35);

    t3 = zeros(num,M);
    for i = 1 : M-1
        t3(:,i) = r_sum(t2(:,(i-1)*K/(M-1)+1:i*K/(M-1)),ones(1,K/(M-1)));
    end
    t3(:,M) = r_sum(t2(:,K+1:K+L),ones(1,L));

    x = zeros(num,M);
    for i = 1 : M-1
       x(:,i) = max(t3(:,M),A(:,i)).*(t3(:,i)-0.5)+0.5;
    end
    x(:,M) = t3(:,M);

    h = concave(x);
    y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;
    y = y';

end

function  y = WFG9(x)
    [dim, num] = size(x);
    x = x';
    M = 3;
    K = M-1;
    L = dim - K;
    dim = 1;
    S = 2 : 2 : 2*M;
    A = ones(1,M-1);

    z01 = x./repmat(2:2:size(x,2)*2,num,1);
            
    t1 = zeros(num,K+L);
    % Same as <t1(:,i)=b_param(z01(:,i),r_sum(z01(:,i+1:end),ones(1,K+L-i)),0.98/49.98,0.02,50)>
    Y = (fliplr(cumsum(fliplr(z01),2))-z01)./repmat(K+L-1:-1:0,num,1);
    t1(:,1:K+L-1) = z01(:,1:K+L-1).^(0.02+(50-0.02)*(0.98/49.98-(1-2*Y(:,1:K+L-1)).*abs(floor(0.5-Y(:,1:K+L-1))+0.98/49.98)));
    % ------------------------------------------------------------------------------------------
    t1(:,end)     = z01(:,end);

    t2 = zeros(num,K+L);
    t2(:,1:K)     = s_decept(t1(:,1:K),0.35,0.001,0.05);
    t2(:,K+1:end) = s_multi(t1(:,K+1:end),30,95,0.35);

    t3 = zeros(num,M);
    for i = 1 : M-1
        t3(:,i) = r_nonsep(t2(:,(i-1)*K/(M-1)+1:i*K/(M-1)),K/(M-1));
    end
    % Same as <t3(:,M)=r_nonsep(t2(:,K+1:end),L)>
    SUM = zeros(num,1);
    for i = K+1 : K+L-1
        for j = i+1 : K+L
           SUM = SUM + abs(t2(:,i)-t2(:,j));
        end
    end
    t3(:,M) = (sum(t2(:,K+1:end),2)+SUM*2)/ceil(L/2)/(1+2*L-2*ceil(L/2));
     % -------------------------------------------

    x = zeros(num,M);
    for i = 1 : M-1
         x(:,i) = max(t3(:,M),A(:,i)).*(t3(:,i)-0.5)+0.5;
    end
    x(:,M) = t3(:,M);

    h = concave(x);
    y = repmat(dim*x(:,M),1,M) + repmat(S,num,1).*h;
    y = y';  
end

function [y, c] = C1_DTLZ1(x)% have some problem
    [dim, num] = size(x);
    x = x';
    M = 3;
    g      = 10*(dim-M+1+sum((x(1,M:end)-0.5).^2-cos(20.*pi.*(x(1,M:end)-0.5)),2));
    y = 0.5*repmat(1+g,1,M).*fliplr(cumprod([ones(size(x,1),1),x(1,1:M-1)],2)).*[ones(size(x,1),1),1-x(1,M-1:-1:1)];
    c = 1-y(:,end)/0.6 - sum(y(:,1:end-1)/0.5,2);
    y = y';
    c = c';
end

function [y, c] = C2_DTLZ2(x)
    [dim, num] = size(x);
    x = x';
    M = 3;
    r = 0.4;
    g = sum((x(1,M:dim)-0.5).^2,2);
    y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(x(1,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(x(1,M-1:-1:1)*pi/2)];
    c = min(min((y-1).^2+repmat(sum(y.^2,2),1,M)-y.^2-r^2,[],2),sum((y-1/sqrt(M)).^2,2)-r^2);
    y = y';
    c = c';
end

function [y, c] = C1_DTLZ3(x)
    [dim, num] = size(x);
    M = 3;
    x = x';
    g      = 10 *(dim-M+1+sum((x(1,M:end)-0.5).^2-cos(20.*pi.*(x(1,M:end)-0.5)),2));
    y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(x,1),1),cos(x(1,1:M-1)*pi/2)],2)).*[ones(size(x,1),1),sin(x(1,M-1:-1:1)*pi/2)];
    if M == 2
      r = 6;
    elseif M <= 3
      r = 9;
    elseif M <= 8
      r = 12.5;
    else
      r = 15;
    end
    c = (sum(y.^2,2)-16).*(sum(y.^2,2)-r.^2);
    y = y';
    c = c';
end
 
function [y, c] = C3_DTLZ4(x)
   [dim, num] = size(x);
   M = 3;
   x = x';
   x(1,1:M-1) = x(1,1:M-1).^100;
   g      = sum((x(1,M:end)-0.5).^2,2);
   y = repmat(1+g,1,M).*fliplr(cumprod([ones(size(g,1),1),cos(x(1,1:M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(x(1,M-1:-1:1)*pi/2)];
   c =  y.^2/4 + (repmat(sum(y.^2,2),1,M)-y.^2)-1;
   y = y';
   c =c';
end

%% Zitzler1 function (ZDT1)
function y = ZDT1 (x)
    % Number of objective is 2.
    % Number of variables is 30. Range x [0,1]
    x = x';
    y(:,1) = x(:,1);
    g = 1 + 9*mean(x(:,2:end),2);
    h = 1 - (y(:,1)./g).^0.5;
    y(:,2) = g.*h;
    y= y';
end


%% Zitzler1 function (ZDT2)
function y = ZDT2 (x)
    % Number of objective is 2.
    % Number of variables is 30. Range x [0,1]
    x = x';
    y(:,1) = x(:,1);
    g = 1 + 9*mean(x(:,2:end),2);
    h = 1 - (y(:,1)./g).^2;
    y(:,2) = g.*h;
    y = y';
end

%% Zitzler1 function (ZDT3) 

function y = ZDT3 (x)
    % Number of objective is 2.
    % Number of variables is 30. Range x [0,1]
    x = x';
    y(:,1) = x(:,1);
    g = 1 + 9*mean(x(:,2:end),2);
    h = 1 - (y(:,1)./g).^0.5 - y(:,1)./g.*sin(10*pi*y(:,1));
    y(:,2) = g.*h;
    y = y';
end

function y = ZDT4 (x)
    % Number of objective is 2.
    % Number of variables is 30. Range x [0,1]
    x = x';
    y(:,1) = x(:,1);
    g = 1 + 10*(size(x,2)-1) + sum(x(:,2:end).^2-10*cos(4*pi*x(:,2:end)),2);
    h = 1 - (y(:,1)./g).^0.5;
    y(:,2) =  g.*h;
    y = y';
end

function y = ZDT5 (x)
    % Number of objective is 2.
    % Number of variables is 30. Range x [0,1]
    x = x';
    u      = zeros(size(x,1),1+(size(x,2)-30)/5);
    u(:,1) = sum(x(:,1),2);
    for i = 2 : size(u,2)
        u(:,i) = sum(x(:,(i-2)*5+31:(i-2)*5+35),2);
    end
    v           = zeros(size(u));
    v(u<5)      = 2 + u(u<5);
    v(u==5)     = 1;
    y(:,1) = 1 + u(:,1);
    g           = sum(v(:,2:end),2);
    h           = 1./y(:,1);
    y(:,2) = g.*h;
    y = y';
end

function  y = ZDT6 (x)
    x = x';
    y(:,1) = 1 - exp(-4*x(:,1)).*sin(6*pi*x(:,1)).^6;
    g = 1 + 9*mean(x(:,2:end),2).^0.25;
    h = 1 - (y(:,1)./g).^2;
    y(:,2) = g.*h;
    y = y';
end


%% UF1

function [y, P] = UF1(x)
    [dim, num]  = size(x);
    tmp         = zeros(dim,num);
    tmp(2:dim,:)= (x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]))).^2;
    tmp1        = sum(tmp(3:2:dim,:));  % odd index
    tmp2        = sum(tmp(2:2:dim,:));  % even index
    y(1,:)      = x(1,:)             + 2.0*tmp1/size(3:2:dim,2);
    y(2,:)      = 1.0 - sqrt(x(1,:)) + 2.0*tmp2/size(2:2:dim,2);

    clear tmp;
end



%% UF2
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y, P] = UF2(x)
    [dim, num]  = size(x);
    X1          = repmat(x(1,:),[dim-1,1]);
    A           = 6*pi*X1 + pi/dim*repmat((2:dim)',[1,num]);
    tmp         = zeros(dim,num);    
    tmp(2:dim,:)= (x(2:dim,:) - 0.3*X1.*(X1.*cos(4.0*A)+2.0).*cos(A)).^2;
    tmp1        = sum(tmp(3:2:dim,:));  % odd index
    tmp(2:dim,:)= (x(2:dim,:) - 0.3*X1.*(X1.*cos(4.0*A)+2.0).*sin(A)).^2;
    tmp2        = sum(tmp(2:2:dim,:));  % even index
    y(1,:)      = x(1,:)             + 2.0*tmp1/size(3:2:dim,2); 
    y(2,:)      = 1.0 - sqrt(x(1,:)) + 2.0*tmp2/size(2:2:dim,2); 
    
    clear X1 A tmp;
end

%% UF3
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF3(x)
    [dim, num]   = size(x);
    Y            = zeros(dim,num);
    Y(2:dim,:)   = x(2:dim,:) - repmat(x(1,:),[dim-1,1]).^(0.5+1.5*(repmat((2:dim)',[1,num])-2.0)/(dim-2.0));
    tmp1         = zeros(dim,num);
    tmp1(2:dim,:)= Y(2:dim,:).^2;
    tmp2         = zeros(dim,num);
    tmp2(2:dim,:)= cos(20.0*pi*Y(2:dim,:)./sqrt(repmat((2:dim)',[1,num])));
    tmp11        = 4.0*sum(tmp1(3:2:dim,:)) - 2.0*prod(tmp2(3:2:dim,:)) + 2.0;  % odd index
    tmp21        = 4.0*sum(tmp1(2:2:dim,:)) - 2.0*prod(tmp2(2:2:dim,:)) + 2.0;  % even index
    y(1,:)       = x(1,:)             + 2.0*tmp11/size(3:2:dim,2);
    y(2,:)       = 1.0 - sqrt(x(1,:)) + 2.0*tmp21/size(2:2:dim,2);
    clear Y tmp1 tmp2;
end

%% UF4
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF4(x)
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(2:dim,:)  = x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    H           = zeros(dim,num);
    H(2:dim,:)  = abs(Y(2:dim,:))./(1.0+exp(2.0*abs(Y(2:dim,:))));
    tmp1        = sum(H(3:2:dim,:));  % odd index
    tmp2        = sum(H(2:2:dim,:));  % even index
    y(1,:)      = x(1,:)          + 2.0*tmp1/size(3:2:dim,2);
    y(2,:)      = 1.0 - x(1,:).^2 + 2.0*tmp2/size(2:2:dim,2);
    clear Y H;
end

%% UF5
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF5(x)
    N           = 10.0;
    E           = 0.1;
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(2:dim,:)  = x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    H           = zeros(dim,num);
    H(2:dim,:)  = 2.0*Y(2:dim,:).^2 - cos(4.0*pi*Y(2:dim,:)) + 1.0;
    tmp1        = sum(H(3:2:dim,:));  % odd index
    tmp2        = sum(H(2:2:dim,:));  % even index
    tmp         = (0.5/N+E)*abs(sin(2.0*N*pi*x(1,:)));
    y(1,:)      = x(1,:)      + tmp + 2.0*tmp1/size(3:2:dim,2);
    y(2,:)      = 1.0 - x(1,:)+ tmp + 2.0*tmp2/size(2:2:dim,2);
    clear Y H;
end

%% UF6
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF6(x)
    N            = 2.0;
    E            = 0.1;
    [dim, num]   = size(x);
    Y            = zeros(dim,num);
    Y(2:dim,:)  = x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    tmp1         = zeros(dim,num);
    tmp1(2:dim,:)= Y(2:dim,:).^2;
    tmp2         = zeros(dim,num);
    tmp2(2:dim,:)= cos(20.0*pi*Y(2:dim,:)./sqrt(repmat((2:dim)',[1,num])));
    tmp11        = 4.0*sum(tmp1(3:2:dim,:)) - 2.0*prod(tmp2(3:2:dim,:)) + 2.0;  % odd index
    tmp21        = 4.0*sum(tmp1(2:2:dim,:)) - 2.0*prod(tmp2(2:2:dim,:)) + 2.0;  % even index
    tmp          = max(0,(1.0/N+2.0*E)*sin(2.0*N*pi*x(1,:)));
    y(1,:)       = x(1,:)       + tmp + 2.0*tmp11/size(3:2:dim,2);
    y(2,:)       = 1.0 - x(1,:) + tmp + 2.0*tmp21/size(2:2:dim,2);
    clear Y tmp1 tmp2;
end

%% UF7
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF7(x)
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(2:dim,:)  = (x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]))).^2;
    tmp1        = sum(Y(3:2:dim,:));  % odd index
    tmp2        = sum(Y(2:2:dim,:));  % even index
    tmp         = (x(1,:)).^0.2;
    y(1,:)      = tmp       + 2.0*tmp1/size(3:2:dim,2);
    y(2,:)      = 1.0 - tmp + 2.0*tmp2/size(2:2:dim,2);
    clear Y;
end

%% UF8
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF8(x)
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(3:dim,:)  = (x(3:dim,:) - 2.0*repmat(x(2,:),[dim-2,1]).*sin(2.0*pi*repmat(x(1,:),[dim-2,1]) + pi/dim*repmat((3:dim)',[1,num]))).^2;
    tmp1        = sum(Y(4:3:dim,:));  % j-1 = 3*k
    tmp2        = sum(Y(5:3:dim,:));  % j-2 = 3*k
    tmp3        = sum(Y(3:3:dim,:));  % j-0 = 3*k
    y(1,:)      = cos(0.5*pi*x(1,:)).*cos(0.5*pi*x(2,:)) + 2.0*tmp1/size(4:3:dim,2);
    y(2,:)      = cos(0.5*pi*x(1,:)).*sin(0.5*pi*x(2,:)) + 2.0*tmp2/size(5:3:dim,2);
    y(3,:)      = sin(0.5*pi*x(1,:))                     + 2.0*tmp3/size(3:3:dim,2);
    clear Y;
end

%% UF9
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF9(x)
    E           = 0.1;
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(3:dim,:)  = (x(3:dim,:) - 2.0*repmat(x(2,:),[dim-2,1]).*sin(2.0*pi*repmat(x(1,:),[dim-2,1]) + pi/dim*repmat((3:dim)',[1,num]))).^2;
    tmp1        = sum(Y(4:3:dim,:));  % j-1 = 3*k
    tmp2        = sum(Y(5:3:dim,:));  % j-2 = 3*k
    tmp3        = sum(Y(3:3:dim,:));  % j-0 = 3*k
    tmp         = max(0,(1.0+E)*(1-4.0*(2.0*x(1,:)-1).^2));
    y(1,:)      = 0.5*(tmp+2*x(1,:)).*x(2,:)     + 2.0*tmp1/size(4:3:dim,2);
    y(2,:)      = 0.5*(tmp-2*x(1,:)+2.0).*x(2,:) + 2.0*tmp2/size(5:3:dim,2);
    y(3,:)      = 1-x(2,:)                       + 2.0*tmp3/size(3:3:dim,2);
    clear Y;
end

%% UF10
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function y = UF10(x)
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(3:dim,:)  = x(3:dim,:) - 2.0*repmat(x(2,:),[dim-2,1]).*sin(2.0*pi*repmat(x(1,:),[dim-2,1]) + pi/dim*repmat((3:dim)',[1,num]));
    H           = zeros(dim,num);
    H(3:dim,:)  = 4.0*Y(3:dim,:).^2 - cos(8.0*pi*Y(3:dim,:)) + 1.0;
    tmp1        = sum(H(4:3:dim,:));  % j-1 = 3*k
    tmp2        = sum(H(5:3:dim,:));  % j-2 = 3*k
    tmp3        = sum(H(3:3:dim,:));  % j-0 = 3*k
    y(1,:)      = cos(0.5*pi*x(1,:)).*cos(0.5*pi*x(2,:)) + 2.0*tmp1/size(4:3:dim,2);
    y(2,:)      = cos(0.5*pi*x(1,:)).*sin(0.5*pi*x(2,:)) + 2.0*tmp2/size(5:3:dim,2);
    y(3,:)      = sin(0.5*pi*x(1,:))                     + 2.0*tmp3/size(3:3:dim,2);
    clear Y H;
end


%% CF1
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF1(x)
    a            = 1.0;
    N            = 10.0;
    [dim, num]   = size(x);
    Y            = zeros(dim,num);
    Y(2:dim,:)   = (x(2:dim,:) - repmat(x(1,:),[dim-1,1]).^(0.5+1.5*(repmat((2:dim)',[1,num])-2.0)/(dim-2.0))).^2;
    tmp1         = sum(Y(3:2:dim,:));% odd index
    tmp2         = sum(Y(2:2:dim,:));% even index 
    y(1,:)       = x(1,:)       + 2.0*tmp1/size(3:2:dim,2);
    y(2,:)       = 1.0 - x(1,:) + 2.0*tmp2/size(2:2:dim,2);
    c(1,:)       = y(1,:) + y(2,:) - a*abs(sin(N*pi*(y(1,:)-y(2,:)+1.0))) - 1.0; % >=0
    clear Y;
end

%% CF2
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF2(x)
    a           = 1.0;
    N           = 2.0;
    [dim, num]  = size(x);
    tmp         = zeros(dim,num);
    tmp(2:dim,:)= (x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]))).^2;
    tmp1        = sum(tmp(3:2:dim,:));  % odd index
    tmp(2:dim,:)= (x(2:dim,:) - cos(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]))).^2;
    tmp2        = sum(tmp(2:2:dim,:));  % even index
    y(1,:)      = x(1,:)             + 2.0*tmp1/size(3:2:dim,2);
    y(2,:)      = 1.0 - sqrt(x(1,:)) + 2.0*tmp2/size(2:2:dim,2);
    t           = y(2,:) + sqrt(y(1,:)) - a*sin(N*pi*(sqrt(y(1,:))-y(2,:)+1.0)) - 1.0; 
%     c(1,:)      = sign(t).*abs(t)./(1.0+exp(4.0*abs(t)));
    c(1,:)      = t./(1.0+exp(4.0*abs(t)));
    clear tmp;
end

%% CF3
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF3(x)
    a            = 1.0;
    N            = 2.0;
    [dim, num]   = size(x);
    Y            = zeros(dim,num);
    Y(2:dim,:)   = x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    tmp1         = zeros(dim,num);
    tmp1(2:dim,:)= Y(2:dim,:).^2;
    tmp2         = zeros(dim,num);
    tmp2(2:dim,:)= cos(20.0*pi*Y(2:dim,:)./sqrt(repmat((2:dim)',[1,num])));
    tmp11        = 4.0*sum(tmp1(3:2:dim,:)) - 2.0*prod(tmp2(3:2:dim,:)) + 2.0;  % odd index
    tmp21        = 4.0*sum(tmp1(2:2:dim,:)) - 2.0*prod(tmp2(2:2:dim,:)) + 2.0;  % even index
    y(1,:)       = x(1,:)          + 2.0*tmp11/size(3:2:dim,2);
    y(2,:)       = 1.0 - x(1,:).^2 + 2.0*tmp21/size(2:2:dim,2);
    c(1,:)       = y(2,:) + y(1,:).^2 - a*sin(N*pi*(y(1,:).^2-y(2,:)+1.0)) - 1.0;   
    clear Y tmp1 tmp2;
end

%% CF4
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF4(x)
    [dim, num]  = size(x);
    tmp         = zeros(dim,num);
    tmp(2:dim,:)= x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    tmp1        = sum(tmp(3:2:dim,:).^2);  % odd index
    tmp2        = sum(tmp(4:2:dim,:).^2);  % even index
    index1      = tmp(2,:) < (1.5-0.75*sqrt(2.0));
    index2      = tmp(2,:)>= (1.5-0.75*sqrt(2.0));
    tmp(2,index1) = abs(tmp(2,index1));
    tmp(2,index2) = 0.125 + (tmp(2,index2)-1.0).^2;
    y(1,:)      = x(1,:)                  + tmp1;
    y(2,:)      = 1.0 - x(1,:) + tmp(2,:) + tmp2;
    t           = x(2,:) - sin(6.0*pi*x(1,:)+2.0*pi/dim) - 0.5*x(1,:) + 0.25;
    c(1,:)      = sign(t).*abs(t)./(1.0+exp(4.0*abs(t)));
    clear tmp index1 index2;
end

%% CF5
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF5(x)
    [dim, num]  = size(x);
    tmp         = zeros(dim,num);
    tmp(2:dim,:)= x(2:dim,:) - 0.8*repmat(x(1,:),[dim-1,1]).*cos(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    tmp1        = sum(2.0*tmp(3:2:dim,:).^2-cos(4.0*pi*tmp(3:2:dim,:))+1.0);  % odd index
    tmp(2:dim,:)= x(2:dim,:) - 0.8*repmat(x(1,:),[dim-1,1]).*sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));    
    tmp2        = sum(2.0*tmp(4:2:dim,:).^2-cos(4.0*pi*tmp(4:2:dim,:))+1.0);  % even index
    index1      = tmp(2,:) < (1.5-0.75*sqrt(2.0));
    index2      = tmp(2,:)>= (1.5-0.75*sqrt(2.0));
    tmp(2,index1) = abs(tmp(2,index1));
    tmp(2,index2) = 0.125 + (tmp(2,index2)-1.0).^2;
    y(1,:)      = x(1,:)                  + tmp1;
    y(2,:)      = 1.0 - x(1,:) + tmp(2,:) + tmp2;
    c(1,:)      = x(2,:) - 0.8*x(1,:).*sin(6.0*pi*x(1,:)+2.0*pi/dim) - 0.5*x(1,:) + 0.25;
    clear tmp;
end

%% CF6
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF6(x)
    [dim, num]  = size(x);
    tmp         = zeros(dim,num);
    tmp(2:dim,:)= x(2:dim,:) - 0.8*repmat(x(1,:),[dim-1,1]).*cos(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    tmp1        = sum(tmp(3:2:dim,:).^2);  % odd index
    tmp(2:dim,:)= x(2:dim,:) - 0.8*repmat(x(1,:),[dim-1,1]).*sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));    
    tmp2        = sum(tmp(2:2:dim,:).^2);  % even index
    y(1,:)      = x(1,:)            + tmp1;
    y(2,:)      = (1.0 - x(1,:)).^2 + tmp2;
    tmp         = 0.5*(1-x(1,:))-(1-x(1,:)).^2;
    c(1,:)      = x(2,:) - 0.8*x(1,:).*sin(6.0*pi*x(1,:)+2*pi/dim) - sign(tmp).*sqrt(abs(tmp));
    tmp         = 0.25*sqrt(1-x(1,:))-0.5*(1-x(1,:));
    c(2,:)      = x(4,:) - 0.8*x(1,:).*sin(6.0*pi*x(1,:)+4*pi/dim) - sign(tmp).*sqrt(abs(tmp));    
    clear tmp;
end

%% CF7
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF7(x)
    [dim, num]  = size(x);
    tmp         = zeros(dim,num);
    tmp(2:dim,:)= x(2:dim,:) - cos(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    tmp1        = sum(2.0*tmp(3:2:dim,:).^2-cos(4.0*pi*tmp(3:2:dim,:))+1.0);  % odd index
    tmp(2:dim,:)= x(2:dim,:) - sin(6.0*pi*repmat(x(1,:),[dim-1,1]) + pi/dim*repmat((2:dim)',[1,num]));
    tmp2        = sum(2.0*tmp(6:2:dim,:).^2-cos(4.0*pi*tmp(6:2:dim,:))+1.0);  % even index
    tmp(2,:)    = tmp(2,:).^2;
    tmp(4,:)    = tmp(4,:).^2;
    y(1,:)      = x(1,:)                                  + tmp1;
    y(2,:)      = (1.0 - x(1,:)).^2 + tmp(2,:) + tmp(4,:) + tmp2;
    tmp         = 0.5*(1-x(1,:))-(1-x(1,:)).^2;
    c(1,:)      = x(2,:) - sin(6.0*pi*x(1,:)+2*pi/dim) - sign(tmp).*sqrt(abs(tmp));
    tmp         = 0.25*sqrt(1-x(1,:))-0.5*(1-x(1,:));
    c(2,:)      = x(4,:) - sin(6.0*pi*x(1,:)+4*pi/dim) - sign(tmp).*sqrt(abs(tmp));    
    clear tmp;
end

%% CF8
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF8(x)
    N           = 2.0;
    a           = 4.0;
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(3:dim,:)  = (x(3:dim,:) - 2.0*repmat(x(2,:),[dim-2,1]).*sin(2.0*pi*repmat(x(1,:),[dim-2,1]) + pi/dim*repmat((3:dim)',[1,num]))).^2;
    tmp1        = sum(Y(4:3:dim,:));  % j-1 = 3*k
    tmp2        = sum(Y(5:3:dim,:));  % j-2 = 3*k
    tmp3        = sum(Y(3:3:dim,:));  % j-0 = 3*k
    y(1,:)      = cos(0.5*pi*x(1,:)).*cos(0.5*pi*x(2,:)) + 2.0*tmp1/size(4:3:dim,2);
    y(2,:)      = cos(0.5*pi*x(1,:)).*sin(0.5*pi*x(2,:)) + 2.0*tmp2/size(5:3:dim,2);
    y(3,:)      = sin(0.5*pi*x(1,:))                     + 2.0*tmp3/size(3:3:dim,2);
    c(1,:)      = (y(1,:).^2+y(2,:).^2)./(1.0-y(3,:).^2) - a*abs(sin(N*pi*((y(1,:).^2-y(2,:).^2)./(1.0-y(3,:).^2)+1.0))) - 1.0;
    clear Y;
end

%% CF9
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF9(x)
    N           = 2.0;
    a           = 3.0;
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(3:dim,:)  = (x(3:dim,:) - 2.0*repmat(x(2,:),[dim-2,1]).*sin(2.0*pi*repmat(x(1,:),[dim-2,1]) + pi/dim*repmat((3:dim)',[1,num]))).^2;
    tmp1        = sum(Y(4:3:dim,:));  % j-1 = 3*k
    tmp2        = sum(Y(5:3:dim,:));  % j-2 = 3*k
    tmp3        = sum(Y(3:3:dim,:));  % j-0 = 3*k
    y(1,:)      = cos(0.5*pi*x(1,:)).*cos(0.5*pi*x(2,:)) + 2.0*tmp1/size(4:3:dim,2);
    y(2,:)      = cos(0.5*pi*x(1,:)).*sin(0.5*pi*x(2,:)) + 2.0*tmp2/size(5:3:dim,2);
    y(3,:)      = sin(0.5*pi*x(1,:))                     + 2.0*tmp3/size(3:3:dim,2);
    c(1,:)      = (y(1,:).^2+y(2,:).^2)./(1.0-y(3,:).^2) - a*sin(N*pi*((y(1,:).^2-y(2,:).^2)./(1.0-y(3,:).^2)+1.0)) - 1.0;
    clear Y;
end

%% CF10
% x and y are columnwise, the imput x must be inside the search space and
% it could be a matrix
function [y,c] = CF10(x)
    a           = 1.0;
    N           = 2.0;
    [dim, num]  = size(x);
    Y           = zeros(dim,num);
    Y(3:dim,:)  = x(3:dim,:) - 2.0*repmat(x(2,:),[dim-2,1]).*sin(2.0*pi*repmat(x(1,:),[dim-2,1]) + pi/dim*repmat((3:dim)',[1,num]));
    H           = zeros(dim,num);
    H(3:dim,:)  = 4.0*Y(3:dim,:).^2 - cos(8.0*pi*Y(3:dim,:)) + 1.0;
    tmp1        = sum(H(4:3:dim,:));  % j-1 = 3*k
    tmp2        = sum(H(5:3:dim,:));  % j-2 = 3*k
    tmp3        = sum(H(3:3:dim,:));  % j-0 = 3*k
    y(1,:)      = cos(0.5*pi*x(1,:)).*cos(0.5*pi*x(2,:)) + 2.0*tmp1/size(4:3:dim,2);
    y(2,:)      = cos(0.5*pi*x(1,:)).*sin(0.5*pi*x(2,:)) + 2.0*tmp2/size(5:3:dim,2);
    y(3,:)      = sin(0.5*pi*x(1,:))                     + 2.0*tmp3/size(3:3:dim,2);
    c(1,:)      = (y(1,:).^2+y(2,:).^2)./(1.0-y(3,:).^2) - a*sin(N*pi*((y(1,:).^2-y(2,:).^2)./(1.0-y(3,:).^2)+1.0)) - 1.0;
    clear Y H;
end





%*****************************************
function Output = s_linear(y,A)
    Output = abs(y-A)./abs(floor(A-y)+A);
end
function Output = r_nonsep(y,A)
    Output = zeros(size(y,1),1);
    for j = 1 : size(y,2)
        Temp = zeros(size(y,1),1);
        for k = 0 : A-2
            Temp = Temp+abs(y(:,j)-y(:,1+mod(j+k,size(y,2))));
        end
        Output = Output+y(:,j)+Temp;
    end
    Output = Output./(size(y,2)/A)/ceil(A/2)/(1+2*A-2*ceil(A/2));
end
function Output = b_flat(y,A,B,C)
    Output = A+min(0,floor(y-B))*A.*(B-y)/B-min(0,floor(C-y))*(1-A).*(y-C)/(1-C);
    Output = round(Output*1e4)/1e4;
end

function Output = b_poly(y,a)
    Output = y.^a;
end

function Output = r_sum(y,w)
    Output = sum(y.*repmat(w,size(y,1),1),2)./sum(w);
end

function Output = convex(x)
    Output = fliplr(cumprod([ones(size(x,1),1),1-cos(x(:,1:end-1)*pi/2)],2)).*[ones(size(x,1),1),1-sin(x(:,end-1:-1:1)*pi/2)];
end
function Output = mixed(x)
    Output = 1-x(:,1)-cos(10*pi*x(:,1)+pi/2)/10/pi;
end

function Output = disc(x)
    Output = 1-x(:,1).*(cos(5*pi*x(:,1))).^2;
end
function Output = linear(x)
    Output = fliplr(cumprod([ones(size(x,1),1),x(:,1:end-1)],2)).*[ones(size(x,1),1),1-x(:,end-1:-1:1)];
end
function Output = concave(x)
    Output = fliplr(cumprod([ones(size(x,1),1),sin(x(:,1:end-1)*pi/2)],2)).*[ones(size(x,1),1),cos(x(:,end-1:-1:1)*pi/2)];
end

function Output = s_multi(y,A,B,C)
    Output = (1+cos((4*A+2)*pi*(0.5-abs(y-C)/2./(floor(C-y)+C)))+4*B*(abs(y-C)/2./(floor(C-y)+C)).^2)/(B+2);
end
function Output = s_decept(y,A,B,C)
    Output = 1+(abs(y-A)-B).*(floor(y-A+B)*(1-C+(A-B)/B)/(A-B)+floor(A+B-y)*(1-C+(1-A-B)/B)/(1-A-B)+1/B);
end

function PopCon = Constraint(PopObj)
    p     = [1.8,2.8];
    q     = [1.8,2.8];
    a     = [2,2];
    b     = [8,8];
    r     = 0.1;
    theta = -0.25 * pi;
    for k = 1 : 2
        PopCon(:,k) = r - ((PopObj(:,1)-p(k))*cos(theta)-(PopObj(:,2)-q(k))*sin(theta)).^2/(a(k)^2) -...
                      ((PopObj(:,1)-p(k))*sin(theta)+(PopObj(:,2)-q(k))*cos(theta)).^2/(b(k)^2);
    end
end
