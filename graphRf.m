function pf = graphRf()
    N         = 10;
    M         = 3;
    pf = UniformPoint(N,M)/2;
    plot3(pf(:,1),pf(:,2),pf(:,3),'o','Color',[0,0.74902,1]);
end