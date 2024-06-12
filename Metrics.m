function fobj =  Metrics(name)
%************************************************************************* 
%%                  Zhang T.
%                
%        Revision: 1.0.0  Date: 2021-04-01
%************************************************************************* 
    switch name
        case 'IGD'
            fobj = @IGD;
        case 'HV'
            fobj = @HV; 
        case 'Coverage'
            fobj = @Coverage;
        case 'GD'
            fobj = @GD;
        case 'Spread'
            fobj = @Spread;
        case 'Spacing'
            fobj = @Spacing;
        case 'PD'
            fobj = @PD;
        case 'DeltaP'
            fobj = @DeltaP;
        case 'DM'
            fobj = @DM;
        case 'CPF'
            fobj = @CPF;
    end
end