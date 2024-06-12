function DrawOld(numObj,costOld,costOpt,testPro,Iteration,maxIter,pf,metric,divImp)
%******************************* DrawOld ****************************
%% Function: DrawOld()
%
% ----------------------------  Function   ---------------------------------
%% Syntax:  
%    DrawOld(numObj,costOld,costOpt,archiveCost,testPro,Iteration,maxIter,pf,metric,divImp)
%% Parameters: 
%   Inputs:
%       numObj:         the number of grey wolves
%       costOld:        function values of  Old grey wolves
%       costOpt:        function values of new grey wolves
%       archiveCost:    function values of first rank level
%       testPro:        the name of test problem
%       Iteration :     current iteration
%       maxIter:        the maximum iteration
%       pf:             the true pareto front
%       metric:         metrices of the algorithm
%       divImp:     	selection methods
%
%%                  Zhang T.
%                
%        Revision: 1.0.0  Date: 2021-04-01
%*************************************************************************    

    %% Plot data  
       global SCORCES;
       
       % Archive diagram
       
       
       iter = Iteration-1;
       numMetric = length(metric);
       for i_metric = 1: numMetric
           metricFun = Metrics(cell2mat(metric(i_metric)));
           Score =  metricFun(costOpt,pf);
           SCORCES(i_metric,iter) = Score;
       end
       
       if numObj == 2

            hold off ;
            subplot(2,1,1)
            plot(costOld(:,1),costOld(:,2),'k.');
            hold on;     
            plot(costOpt(:,1),costOpt(:,2),'rd');
            t = title(['TestProbem: ',testPro,'   Iterations:',num2str(iter)]);
            t.FontName = 'Cambria';
            t.FontSize = 14;
            t.Color = 'red';
            t.FontWeight = 'bold';
            legend({'Grey wolves','Non-dominated solutions'},'Location','northeast','FontSize',10,'TextColor','blue','FontName','Cambria');
            legend('boxoff');
            set(gca,'XTickMode','auto','YTickMode','auto','FontName','Cambria','FontWeight','bold');
            drawnow;
       else 
            
            hold off;
            subplot(2,1,1)
            plot3(costOld(:,1),costOld(:,2),costOld(:,3),'k.');
            hold on;
            plot3(costOpt(:,1),costOpt(:,2),costOpt(:,3),'rd');
            t = title(['TestProblem: ',testPro, '   Iteration:',num2str(iter)]);
            t.FontName = 'Cambria';
            t.FontSize = 14;
            t.Color = 'red';
            t.FontWeight = 'bold';
            legend({'Grey wolves','Non-dominated solutions'},'Location','northeast','FontSize',10,'TextColor','blue','FontName','Cambria');
            legend('boxoff');
%             set(gca,'xdir','reverse','ydir','reverse')
            set(gca,'XTickMode','auto','YTickMode','auto','ZTickMode','auto','View',[-45 20],'FontName','Cambria','FontWeight','bold');
            grid on;
            axis on;
            axis tight;
            drawnow;
         
       end
       
       if iter == maxIter
            subplot(2,1,2)
            if numObj == 2
                plot(pf(:,1),pf(:,2),'o');
                set(gca,'XTickMode','auto','YTickMode','auto','FontName','Cambria','FontWeight','bold');
            else               
                plot3(pf(:,1),pf(:,2),pf(:,3),'o');
                set(gca,'XTickMode','auto','YTickMode','auto','ZTickMode','auto','View',[-45 20],'FontName','Cambria','FontWeight','bold');
                grid on;
                axis on;
                axis tight;
            end
            t = title(['TestProbem: ',testPro]);
            t.FontName = 'Cambria';
            t.FontSize = 14;
            t.Color = 'red';
            t.FontWeight = 'bold';
            legend({'True PF'},'Location','northeast','FontSize',10,'TextColor','blue','FontName','Cambria');
            legend('boxoff');
            dir = ['Images/' divImp '/' testPro '/' datestr(datetime,'yyyymmddTHHMMSS')];
            if ~exist(dir,'dir')
                mkdir(dir);
            end
            figName = [dir '/' testPro];
            strfigure1 = ['print -f1 -djpeg -r600 ' figName];
            eval(strfigure1);
            saveas(gcf,figName);
           
%             set(gca,'xdir','reverse','ydir','reverse')

            for i_metric = 1: numMetric
                figure (i_metric+1);
                plot([1:maxIter],SCORCES(i_metric,:),'-*');
                t = title(metric(i_metric));
                t.FontName = 'Cambria';
                t.FontSize = 14;
                t.Color = 'red';
                t.FontWeight = 'bold';
                metricName = cell2mat(metric(i_metric));
                legend({metricName},'Location','northeast','FontSize',10,'TextColor','blue','FontName','Cambria');
                legend('boxoff');
                set(gca,'XTickMode','auto','YTickMode','auto','FontName','Cambria','FontWeight','bold');
                if ~exist(dir,'dir')
                    mkdir(dir);
                end
                str = ['print -f' mat2str(i_metric+1) ' -djpeg -r600 ' dir '/' metricName];
                eval(str);
                figName = [dir '/' metricName];
                saveas(gcf,figName);
            end
            path =[dir '/SCORCES.mat'];
            save (path,'SCORCES');
            clear global SCORCES; % note 
            clear costOpt;
            clear costOld;
            clear pf;
            close all;
            t= toc;
            timePath = [dir  '/time.mat'];
            save(timePath,'t');
       end
           
end