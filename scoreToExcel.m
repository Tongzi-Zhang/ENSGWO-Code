function scoreToExcel
  s(1) = load('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Images\refPoints\ZDT6\20211220T093940\SCORCES.mat');
  s(2) = load('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Images\refPoints\ZDT6\20211220T093833\SCORCES.mat');
  s(3) = load('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Images\refPoints\ZDT6\20211220T093548\SCORCES.mat');
  s(4) = load('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Images\refPoints\ZDT6\20211105T180259\SCORCES.mat');
  s(5) = load('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Images\refPoints\ZDT6\20211220T102353\SCORCES.mat');
  sheetName = 'ZDT6';
  number = 5;
  igd = 1;
  hv = 11;
  gd = 21;
  cpf =31;
  for i = 1 : number
      igd = igd +1;
      hv = hv +1;
      gd = gd +1;
      cpf = cpf+1;
      igdCell = ['D',num2str(igd),':ALO',num2str(igd)];
      hvCell  = ['D',num2str(hv),':ALO',num2str(hv)];
      gdCell  = ['D',num2str(gd),':ALO',num2str(gd)];
      cpfCell = ['D',num2str(cpf),':ALO',num2str(cpf)];
      metricExcel =[ s(i).SCORCES(1,:); s(i).SCORCES(2,:); s(i).SCORCES(4,:); s(i).SCORCES(10,:)];
      xlswrite('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Metrics for ENS-MOGWO.xlsx',metricExcel(1,:),sheetName,igdCell)
      xlswrite('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Metrics for ENS-MOGWO.xlsx',metricExcel(2,:),sheetName,hvCell)
      xlswrite('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Metrics for ENS-MOGWO.xlsx',metricExcel(3,:),sheetName,gdCell)
      xlswrite('E:\水-能源-粮食\水-能源-粮食系统\多目标神经网络\Grey wolf\Modefied MOGWO\ENSGWO 0530\Metrics for ENS-MOGWO.xlsx',metricExcel(4,:),sheetName,cpfCell)
  end 
end