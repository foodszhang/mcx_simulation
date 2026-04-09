function  [uam, Dm, An]= Initialization
%=====激发光方程的参数设置
%=====发射光方程的参数设置   
    ua2 = [0.08697  0.05881   0.01304   0.35182   0.06597    0.19639];      % 吸收系数 Muscle Heart stomach Liver kedney Lung  
    us2 = [4.29071  6.42581    17.9615  6.78066   16.09293   36.52133];      % 散射系数
    g   = [0.9 0.85 0.92 0.9 0.85 0.94];                      % 各向异性参数
    I=ones(size(g));
    n = 1.37;                                       % 折射系数
    D2 = (3.*(ua2+(I-g).*us2)).^(-1);
    R = -1.4399.*n.^(-2)+0.7099.*n.^(-1)+0.6681+0.0636.*n;
    An = (1+R)/(1-R);
%save An.mat An
%     ratio_m=ua2./us2;
    %    Comsol刨分得到各个组织的标识符
    %   Musical：1    Heart：2    stomach：3    Liver：4    kedney：5    Lung2：6  flurophore：7
    uam=[ua2(1)     ua2(2)   ua2(3)   ua2(4)  ua2(5)  ua2(6)  ua2(4) ];
    Dm =[ D2(1)      D2(2)    D2(3)    D2(4)   D2(5)   D2(6)   D2(4) ];
     disp('Muscle    Heart      stomach   Liver   kidney  Lung  flurophore ')
    disp(['Dm  =  ',num2str(Dm)])
    disp(['uam =  ',num2str(uam)])
    disp([' An =  ',num2str(An)])
