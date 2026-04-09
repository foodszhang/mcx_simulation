function  [uax, Dx,uam, Dm, An,d]= Initialization1
%=====激发光方程的参数设置
    disp('                   *** INITIALIZATION ***                   ')
    ua1 = [0.0052  0.0114  0.0133  0.0329  0.0083  0.066];     % 吸收系数 Muscle stomach Lung Liver Heart kidney 
    us1 = [10.8   17.96    19.7    7.0     6.733  16.09];    % 散射系数
    g   = [0.9 0.92 0.9 0.9 0.85 0.86];                      % 各向异性参数
    I=ones(size(g));
    us_1=(I-g).*us1;
    D1 = (3.*(ua1+(I-g).*us1)).^(-1);
    ratio_x=ua1./us1;
    %   Comsol刨分得到各个组织的标识符
    %   Musical：1    Heart：2    stomach：3    Liver：4    kedney：5    Lung2：6    flurophore：7
    uax= [ua1(1)   ua1(5)    ua1(2)  ua1(4)  ua1(6)  ua1(3)  ua1(4)];
    Dx = [ D1(1)    D1(5)     D1(2)   D1(4)   D1(6)   D1(3)   D1(4)];
    disp('Muscle    Heart      stomach   Liver   kidney  Lung  flurophore ')
    disp(['uax =  ',num2str(uax)])
    disp(['Dx  =  ',num2str(Dx)])
    uax_s=uax(1);
    usx_s=us_1(1);
    d=1/(uax_s+usx_s)
 %=====发射光方程的参数设置   
    ua2 = [0.0068  0.007   0.0203   0.0176   0.0104    0.038];      % 吸收系数 Muscle stomach Lung Liver Heart kedney
    us2 = [10.3    16.98    19.5     6.48      6.6    14.74];      % 散射系数
    g   = [0.9 0.92 0.9 0.9 0.85 0.86];                      % 各向异性参数
    n = 1.37;                                       % 折射系数
    D2 = (3.*(ua2+(I-g).*us2)).^(-1);
    R = -1.4399.*n.^(-2)+0.7099.*n.^(-1)+0.6681+0.0636.*n;
    An = (1+R)/(1-R);
    save An.mat An
    ratio_m=ua2./us2;
    %    Comsol刨分得到各个组织的标识符
    %   Musical：1    Heart：2    stomach：3    Liver：4    kedney：5    Lung2：6  flurophore：7
    uam=[ua2(1)     ua2(5)   ua2(2)   ua2(4)  ua2(6)  ua2(3)  ua2(4) ];
    Dm =[ D2(1)      D2(5)    D2(2)    D2(4)   D2(6)   D2(3)   D2(4) ];
     disp('Muscle    Heart      stomach   Liver   kidney  Lung  flurophore ')
    disp(['Dm  =  ',num2str(Dm)])
    disp(['uam =  ',num2str(uam)])
    disp([' An =  ',num2str(An)])
