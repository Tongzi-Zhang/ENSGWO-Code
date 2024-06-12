function [fobj, numObj]= PF(name)
%******************************* PF ************************************
%% Function: PF()
%% Description: 
%   To construct the true pareto front of the test problem
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%   [fobj, numObj]= PF(name)
%% Parameters:
%   Inputs:
%      name:    the name of test problem
%      
%   Outputs:
%      fobj:    the handle of test problem
%      numObj:  the number of objective function (not use)
% ----------------------------  Copyright  -------------------------------- 
%% Cited from PlatEMO All rights reserved.
%   
%   Copyright (c) 2021 BIMK Group.
%
%% Modified by Zhang, T.
%                
%   Revision: 1.0.0  Date: 2021-04-01
%************************************************************************

    switch name
        case 'UF1'
            fobj = @PF_UF1;
            numObj = 2;
        case 'UF2'
            fobj = @PF_UF2; 
            numObj = 2;
        case 'UF3'
            fobj = @PF_UF3;
            numObj = 2;
        case 'UF4'
            fobj = @PF_UF4;
            numObj = 2;
        case 'UF5'
            fobj = @PF_UF5;
            numObj = 2;
        case 'UF6'
            fobj = @PF_UF6;
            numObj = 2;
        case 'UF7'
            fobj = @PF_UF7;
            numObj = 2;
        case 'UF8'
            fobj = @PF_UF8;
            numObj = 3;
        case 'UF9'
            fobj = @PF_UF9;
            numObj = 3;
        case 'UF10'
            fobj = @PF_UF10;
            numObj = 3;
        case 'CF1'
            fobj = @PF_CF1;
            numObj = 2;
        case 'CF2'
            fobj = @PF_CF2;
            numObj = 2;
        case 'CF3'
            fobj = @PF_CF3;
            numObj = 2;
        case 'CF4'
            fobj = @PF_CF4;
            numObj = 2;
        case 'CF5'
            fobj = @PF_CF5;
            numObj = 2;
        case 'CF6'
            fobj = @PF_CF6;
            numObj = 2;
        case 'CF7'
            fobj = @PF_CF7;
            numObj = 2;
        case 'CF8'
            fobj = @PF_CF8;
            numObj = 3;
        case 'CF9'
            fobj = @PF_CF9;
            numObj = 3;
        case 'CF10'
            fobj = @PF_CF10;
            numObj = 3;
 
        case 'ZDT1'
            fobj = @PF_ZDT1;
            numObj = 2;
        case 'ZDT2'
            fobj = @PF_ZDT2;
            numObj = 2;
       case 'ZDT3'
            fobj = @PF_ZDT3;
            numObj = 2;
       case 'ZDT4'
            fobj = @PF_ZDT4;
            numObj = 2;
        case 'ZDT5'
            fobj = @PF_ZDT5;
            numObj = 2;
       case 'ZDT6'
            fobj = @PF_ZDT6;
            numObj = 2; 
        case 'DTLZ1' 
            fobj = @PF_DTLZ1;
            numObj = 3;
        case 'DTLZ2' 
            fobj = @PF_DTLZ2;
            numObj = 3;
        case 'DTLZ3' 
            fobj = @PF_DTLZ3;
            numObj = 3;           
        case 'DTLZ4' 
            fobj = @PF_DTLZ4;
            numObj = 3;           
        case 'DTLZ5' 
            fobj = @PF_DTLZ5;
            numObj = 3;            
        case 'DTLZ6' 
            fobj = @PF_DTLZ6;
            numObj = 3;            
        case 'DTLZ7' 
            fobj = @PF_DTLZ7;
            numObj = 3;           
        case 'DTLZ8' 
            fobj = @PF_DTLZ8;
            numObj = 3;            
        case 'DTLZ9' 
            fobj = @PF_DTLZ9;
            numObj = 2;            
        case 'C1_DTLZ1'
            fobj = @PF_C1_DTLZ1;
            numObj = 3;
        case 'C2_DTLZ2'
            fobj = @PF_C2_DTLZ2;
            numObj = 3;
        case 'C1_DTLZ3'
            fobj = @PF_C1_DTLZ3;
            numObj = 3;
        case  'C3_DTLZ4'
            fobj = @PF_C3_DTLZ4;
            numObj = 3;
        case 'WFG1'
            fobj = @PF_WFG1;
            numObj = 3;
        case 'WFG2'
            fobj = @PF_WFG2;
            numObj = 3;
       case 'WFG3'
            fobj = @PF_WFG3;
            numObj = 3;
       case 'WFG4'
            fobj = @PF_WFG4;
            numObj = 3;    
       case 'WFG5'
            fobj = @PF_WFG5;
            numObj = 3;
       case 'WFG6'
            fobj = @PF_WFG6;
            numObj = 3;   
        case 'WFG7'
            fobj = @PF_WFG7;
            numObj = 3;
        case 'WFG8'
            fobj = @PF_WFG8;
            numObj = 3;
        case 'WFG9'
            fobj = @PF_WFG9;
            numObj = 3; 
        case 'IMOP1'
            fobj = @PF_IMOP1;
            numObj = 2;
        case 'IMOP2'
            fobj = @PF_IMOP2;
            numObj = 2;
        case 'IMOP3'
            fobj = @PF_IMOP3;
            numObj = 2;
        case 'IMOP4'
            fobj = @PF_IMOP4;
            numObj = 3;
        case 'IMOP5'
            fobj = @PF_IMOP5;
            numObj = 3;
        case 'IMOP6'
            fobj = @PF_IMOP6;
            numObj = 3;
        case 'IMOP7'
            fobj = @PF_IMOP7;
            numObj = 3;
        case 'IMOP8'
            fobj = @PF_IMOP8;
            numObj = 3;
        case 'DC1_DTLZ1'
            fobj = @PF_DC1_DTLZ1;
            numObj = 3;
        case 'LIRCMOP1'
            fobj = @PF_LIRCMOP1;
            numObj = 2;
        case 'LIRCMOP6'
            fobj = @PF_LIRCMOP6;
            numObj = 2;
    end
end



function P =PF_LIRCMOP1()
    N = 10000;
    P(:,1) = linspace(0,1,N)';
    P(:,2) = 1 - P(:,1).^2;
    P      = P + 0.5;
end

function P =PF_LIRCMOP6()
    N = 10000;
    P(:,1) = linspace(0,1,N)';
    P(:,2) = 1 - P(:,1).^2;
    P      = P + 0.7057;
    P(any(Constraint(P)>0,2),:) = [];
end


function P = PF_UF1()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^0.5;
    
    
end

function P = PF_UF2()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^0.5;
end

function P = PF_UF3()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^0.5;
end

function P = PF_UF4()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^2;

end

function P = PF_UF5()
    N         = 10000;
    P(:,1) = (0:1:20)'/20;
    P(:,2) = 1 - P(:,1);
end

function P = PF_UF6()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1);
    P(P(:,1)>0 & P(:,1)<1/4 | P(:,1)>1/2 & P(:,1)<3/4,:) = [];
end

function P = PF_UF7()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1);
end

function P = PF_UF8()
    N         = 10000;
    P = UniformPoint(N,3);
    P = P./repmat(sqrt(sum(P.^2,2)),1,3);
end

function P = PF_UF9()

    N         = 10000;
    P = UniformPoint(N,3);
    P(P(:,1)>(1-P(:,3))/4 & P(:,1)<(1-P(:,3))*3/4,:) = [];
end

function P = PF_UF10()
    N         = 10000;
    P = UniformPoint(N,3);
    P = P./repmat(sqrt(sum(P.^2,2)),1,3);
end

function P = PF_CF1()
    N         = 10000;
    P(:,1) = (0:1/20:1)';
    P(:,2) = 1 - P(:,1);
end

function P = PF_CF2()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - sqrt(P(:,1));
    P(0<P(:,1) & P(:,1)<1/16 | 1/4<P(:,1) & P(:,1)<9/16,:) = [];
end

function P = PF_CF3()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^2;
    P(0<P(:,1) & P(:,1)<1/2 | sqrt(1/2)<P(:,1) & P(:,1)<sqrt(3/4),:) = [];
end

function P = PF_CF4()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1);
    temp1  = 0.5<P(:,1) & P(:,1)<=0.75;
    temp2  = 0.75<P(:,1);
    P(temp1,2) = -0.5*P(temp1,1)+3/4;
    P(temp2,2) = 1 - P(temp2,1)+0.125;
end


function P = PF_CF5()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = (1-P(:,1)).^2;
    temp1  = 0.5<P(:,1) & P(:,1)<=0.75;
    temp2  = 0.75<P(:,1);
    P(temp1,2) = 0.5*(1-P(temp1,1));
    P(temp2,2) = 0.25*sqrt(1-P(temp2,1));
end

function P = PF_CF6()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = (1-P(:,1)).^2;
    temp1  = 0.5<P(:,1) & P(:,1)<=0.75;
    temp2  = 0.75<P(:,1);
    P(temp1,2) = 0.5*(1-P(temp1,1));
    P(temp2,2) = 0.25*sqrt(1-P(temp2,1));

end

function P = PF_CF7()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = (1-P(:,1)).^2;
    temp1  = 0.5<P(:,1) & P(:,1)<=0.75;
    temp2  = 0.75<P(:,1);
    P(temp1,2) = 0.5*(1-P(temp1,1));
    P(temp2,2) = 0.25*sqrt(1-P(temp2,1));
end

function P = PF_CF8()
    N         = 10000;
    N      = ceil(N/5)*5;
    P      = zeros(N,3);
    P(:,3) = repmat(sin((0:1/(N/5-1):1).*pi/2)',5,1);
    for i = 0 : 4
        P(i*N/5+1:(i+1)*N/5,1) = sqrt(i/4*(1-P(i*N/5+1:(i+1)*N/5,3).^2));
    end
    P(:,2) = sqrt(roundn(1-P(:,1).^2-P(:,3).^2,-10));
end

function P = PF_CF9()
    N         = 10000;
    P = UniformPoint(N,3);
    P = P./repmat(sqrt(sum(P.^2,2)),1,3);
    P(1e-5<P(:,1) & P(:,1)<sqrt((1-P(:,3).^2)/4) | sqrt((1-P(:,3).^2)/2)<P(:,1) & P(:,1)<sqrt(3*(1-P(:,3).^2)/4),:) = [];
end

function P = PF_CF10()
    N         = 10000;
    P = UniformPoint(N,3);
    P = P./repmat(sqrt(sum(P.^2,2)),1,3);
    P(1e-5<P(:,1) & P(:,1)<sqrt((1-P(:,3).^2)/4) | sqrt((1-P(:,3).^2)/2)<P(:,1) & P(:,1)<sqrt(3*(1-P(:,3).^2)/4),:) = [];
end

function P = PF_ZDT3()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^0.5 - P(:,1).*sin(10*pi*P(:,1));
    [~,~,~,FrontNo] = Non_ENS(P,1);
    P      = P(FrontNo == 1,:);
end

function P = PF_ZDT2()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^2;
end

function P = PF_ZDT1()
    N         = 10000;
    P(:,1) = (0:1/(N-1):1)';
    P(:,2) = 1 - P(:,1).^0.5;
end

function P = PF_ZDT4()
    N         = 10000;
    P(:,1) = linspace(0,1,N)';
    P(:,2) = 1 - P(:,1).^0.5;
end

function P = PF_ZDT5()
    N         = 10000;
    D         = 30;
    P(:,1) = 1 : 31;
    P(:,2) = (D-30)./5./P(:,1);
end

function P = PF_ZDT6()
    N         = 10000;
    D         = 30;
    minf1  = 0.280775;
    P(:,1) = linspace(minf1,1,N)';
    P(:,2) = 1 - P(:,1).^2;

end

function P = PF_DTLZ1()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M)/2;
end

function P = PF_DTLZ2()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
end
function P = PF_DTLZ3()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
end
function P = PF_DTLZ4()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
end
function P = PF_DTLZ5()
    N         = 10000;
    M         = 3;
    P = [0:1/(N-1):1;1:-1/(N-1):0]';
    P = P./repmat(sqrt(sum(P.^2,2)),1,size(P,2));
    P = [P(:,ones(1,M-2)),P];
    P = P./sqrt(2).^repmat([M-2,M-2:-1:0],size(P,1),1);
end
function P = PF_DTLZ6()
    N         = 10000;
    M         = 3;
    P = [0:1/(N-1):1;1:-1/(N-1):0]';
    P = P./repmat(sqrt(sum(P.^2,2)),1,size(P,2));
    P = [P(:,ones(1,M-2)),P];
    P = P./sqrt(2).^repmat([M-2,M-2:-1:0],size(P,1),1);
end
function P = PF_DTLZ7()
    N         = 10000;
    M         = 3;
    interval     = [0,0.251412,0.631627,0.859401];
    median       = (interval(2)-interval(1))/(interval(4)-interval(3)+interval(2)-interval(1));
    X            = UniformPoint(N,M-1,'grid');
    X(X<=median) = X(X<=median)*(interval(2)-interval(1))/median+interval(1);
    X(X>median)  = (X(X>median)-median)*(interval(4)-interval(3))/(1-median)+interval(3);
    P            = [X,2*(M-sum(X/2.*(1+sin(3*pi.*X)),2))];
end

function P = PF_DTLZ8()
    
    N         = 10000;
    M         = 3;   
    if M == 2
        temp = (0:1/(N-1):1)';
        P    = [(1-temp)/4,temp];
    else
        temp = UniformPoint(N/(M-1),3);
        temp(:,3) = temp(:,3) / 2;
        temp = temp(temp(:,1)>=(1-temp(:,3))/4 & temp(:,1)<=temp(:,2) & temp(:,3)<=1/3,:);
        P    = [repmat(temp(:,2),M-1,M-1),repmat(temp(:,3),M-1,1)];
        for i = 1 : M-1
            P((i-1)*size(temp,1)+1:i*size(temp,1),i) = temp(:,1);
        end
        gap  = sort(unique(P(:,M)));
        gap  = gap(2) - gap(1);
        temp = (1/3:gap:1)';
        P    = [P;repmat((1-temp)/4,1,M-1),temp];
        P    = unique(P,'rows');
    end
end

function P = PF_DTLZ9()
    N         = 10000;
    M         = 2;   
    Temp = (0:1/(N-1):1)';
    P    = [repmat(cos(0.5.*pi.*Temp),1,M-1),sin(0.5.*pi.*Temp)];
end

function P = PF_C1_DTLZ1()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M)/2;
end


function P = PF_C2_DTLZ2()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
    if M == 3
        r = 0.4;
    else
        r = 0.5;
    end
    P(min(min((P-1).^2+repmat(sum(P.^2,2),1,M)-P.^2-r^2,[],2),sum((P-1/sqrt(M)).^2,2)-r^2)>0,:) = [];
end

function P = PF_C1_DTLZ3()
     N         = 10000;
     M = 3;
     P = UniformPoint(N,M);
     P = P./repmat(sqrt(sum(P.^2,2)),1,M);
end

function P = PF_C3_DTLZ4()
     N         = 10000;
     M         = 3;
     P = UniformPoint(N,M);
     P = P./repmat(sqrt(sum(P.^2,2)-3/4*max(P.^2,[],2)),1,M);
end

%% Generate points on the Pareto front
function P = PF_WFG1()
     N         = 10000;
     M         = 3;
     P = UniformPoint(N,M);
     c = ones(size(P,1),M);
     for i = 1 : size(P,1) 
        for j = 2 : M
           temp = P(i,j)/P(i,1)*prod(1-c(i,M-j+2:M-1));
           c(i,M-j+1) = (temp^2-temp+sqrt(2*temp))/(temp^2+1);
        end
     end
     x = acos(c)*2/pi;
     temp = (1-sin(pi/2*x(:,2))).*P(:,M)./P(:,M-1);
     a = 0 : 0.0001 : 1;
     E = abs(temp*(1-cos(pi/2*a))-1+repmat(a+cos(10*pi*a+pi/2)/10/pi,size(x,1),1));
     [~,rank] = sort(E,2);
     for i = 1 : size(x,1)
         x(i,1) = a(min(rank(i,1:10)));
     end
     P      = convex(x);
     P(:,M) = mixed(x);
     P      = repmat(2:2:2*M,size(P,1),1).*P;
end

function P = PF_WFG2()
     N         = 10000;
     M         = 3;
     P = UniformPoint(N,M);
     c = ones(size(P,1),M);
     for i = 1 : size(P,1) 
        for j = 2 : M
           temp = P(i,j)/P(i,1)*prod(1-c(i,M-j+2:M-1));
           c(i,M-j+1) = (temp^2-temp+sqrt(2*temp))/(temp^2+1);
         end
     end
     x = acos(c)*2/pi;
     temp = (1-sin(pi/2*x(:,2))).*P(:,M)./P(:,M-1);
     a = 0 : 0.0001 : 1;
     E = abs(temp*(1-cos(pi/2*a))-1+repmat(a.*cos(5*pi*a).^2,size(x,1),1));
     [~,rank] = sort(E,2);
     for i = 1 : size(x,1)
         x(i,1) = a(min(rank(i,1:10)));
     end
     P      = convex(x);
     P(:,M) = disc(x);
     [~,~,~,FrontNo] = Non_ENS(P,1);
     P      = P(FrontNo == 1,:);
     P      = repmat(2:2:2*M,size(P,1),1).*P;
end
 
function P = PF_WFG3()
     N         = 10000;
     M         = 3;
     X = linspace(0,1,N)';
     X = [X,zeros(N,M-2)+0.5,zeros(N,1)];
     P = linear(X);
     P = repmat(2:2:2*M,size(P,1),1).*P;
end


function P = PF_WFG4()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
    P = repmat(2:2:2*M,size(P,1),1).*P;
end

function P = PF_WFG5()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
    P = repmat(2:2:2*M,size(P,1),1).*P;

end

function P = PF_WFG6()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
    P = repmat(2:2:2*M,size(P,1),1).*P;
end

function P = PF_WFG7()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
    P = repmat(2:2:2*M,size(P,1),1).*P;
end

function P = PF_WFG8()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
    P = repmat(2:2:2*M,size(P,1),1).*P;
end


function P = PF_WFG9()
    N         = 10000;
    M         = 3;
    P = UniformPoint(N,M);
    P = P./repmat(sqrt(sum(P.^2,2)),1,M);
    P = repmat(2:2:2*M,size(P,1),1).*P;
end

function P = PF_IMOP1()
    N         = 10000;
    x = linspace(0.5^4,1,floor(N/2))';
    P(:,1) = [flip((1-x.^0.25).^4);x];
    P(:,2) = [flip(x);(1-x.^0.25).^4]; 
end
function P = PF_IMOP2()
    N         = 10000;
    x = linspace(0,0.5^0.25,floor(N/2))';
    P(:,1) = [x;flip((1-x.^4).^0.25)];
    P(:,2) = [(1-x.^4).^0.25;flip(x)];
end
function P = PF_IMOP3()
    N         = 10000;
    x      = linspace(0,1,N)';
    P(:,1) = 1 + cos(x*pi*10)/5 - x;
    P(:,2) = x;
    [~,~,~,FrontNo] = Non_ENS(P,1);
    P      = P(FrontNo == 1,:);
end
function P = PF_IMOP4()
    N         = 10000;
    P(:,1) = linspace(0,1,N)';
    P(:,2) = P(:,1) + sin(10*pi*P(:,1))/10;
    P(:,3) = 1 - P(:,1);
end
function P = PF_IMOP5()
    N         = 10000;
    [x,y] = meshgrid(linspace(0,1,ceil(sqrt(N/8*1.3))));
    x = [x(:),y(:)] - 0.5;
    x = 0.2*x(sum(x.^2,2)<=0.25,:);
    r = [0.4*cos((1:8)'*pi/4),0.4*sin((1:8)'*pi/4)];
    P = [];
    for i = 1 : size(r,1)
        P = [P;x+repmat(r(i,:),size(x,1),1)];
    end
    P = [P,0.5-sum(P,2)];
end
function P = PF_IMOP6()
    N         = 10000;
    [x,y] = meshgrid(linspace(0,1,ceil(sqrt(N))));
    P = [x(:),y(:)];
    r = max(0,min(sin(3*pi*P(:,1)).^2,sin(3*pi*P(:,2)).^2)-0.05);
    P(:,3) = 1 - sum(P,2)/2;
    P = P + repmat(ceil(r),1,3);
    [~,~,~,FrontNo] = Non_ENS(P,1);
    P      = P(FrontNo == 1,:);
end
function P = PF_IMOP7()
    N         = 10000;
    P = UniformPoint(N,3);
    P = P./repmat(sqrt(sum(P.^2,2)),1,3);
    r = min(min(abs(P(:,1)-P(:,2)),abs(P(:,2)-P(:,3))),abs(P(:,3)-P(:,1)));
    P(r>0.1,:) = [];
end
function P = PF_IMOP8()
    N         = 10000;
    [x,y] = meshgrid(linspace(0,1,ceil(sqrt(N))));
    P = [x(:),y(:)];
    P = [P,3-sum(P.*(1+sin(19*pi.*P)),2)];
    [~,~,~,FrontNo] = Non_ENS(P,1);
    P      = P(FrontNo == 1,:);
end

function P = PF_DC1_DTLZ1()
    N         = 10000; 
    M = 3;
    P = UniformPoint(N,M);
    x = zeros(size(P,1),M-1);
    for i = 1 : size(P,1)
        for j = 2 : M
            temp = P(i,j)./P(i,1).*prod(x(i,M-j+2:M-1));
            x(i,M-j+1) = 1./(1+temp);
        end
    end
    P(cos(3*pi*x(:,1))<0.5,:) = [];
    P = P/2;
end

%************  invoking function of WFG *************************

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
%***************************************************************