function [po, g, An, A1, B1, J0, J1, J2, J3, Cb, Cb_DE, a11, a12, a21, a22, c1, c2] = caculate
global Dm  uam us
    po=10^6;
    ni=1.37;
    nt=1;
    g = [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90];      

    R1=-1.4399*ni^-2+0.7099*ni^-1+0.6681+0.0636*ni;
    An=(1+R1)/(1-R1);
    NSP=3;
    R=zeros(1,NSP*2);
    R2=R;
    sinc=nt/ni;  % 临界角cos
    sinc=min(sinc,1);
    cosc=sqrt(1-sinc^2);
    a=cosc;b=1;
    tol=1e-10;
    for i=1:NSP*2
        fp=@(x)(x.^i).*(0.5*((ni.*x-nt.*sqrt(1-(ni.*sqrt(1-x.^2)./nt).^2))./(ni.*x+nt.*sqrt(1-(ni.*sqrt(1-x.^2)./nt).^2))).^2+0.5*((ni.*sqrt(1-(ni.*sqrt(1-x.^2)./nt).^2)-nt.*x)/(ni.*sqrt(1-(ni.*sqrt(1-x.^2)./nt).^2)+nt.*x)).^2);
        [q,errbnd]=quadgk(fp,a,b,'RelTol',1e-8,'AbsTol',1e-12);%tol代表误差，Reltol是相对误差,Abstol是绝对误差.
        R(i)=q+cosc^(i+1)/(i+1);%R为展开矩
    end
    %求解参数系数A，B,C,D
    A1=-R(1);
    A2=-9*R(1)/4+15*R(3)/2-25*R(5)/4;
    B1=3*R(2);
    B2=63*R(2)/4-105*R(4)/2+175*R(6)/4;
    C1=-3*R(1)/2+5*R(3)/2;
    C2=-3*R(1)/2+5*R(3)/2;
    D1=3*R(2)/2-5*R(4)/2;
    D2=3*R(2)/2-5*R(4)/2;
    J0=-1*R(1)/2;
    J1=-3*R(2)/2;
    J2=5*R(1)/4-15*R(3)/4;
    J3=21*R(2)/4-35*R(4)/4;
    Mbg=zeros(2);
    Mbm=zeros(2);
    Mbg(1,:)=[(1+B1),-7*D1];
    Mbm(1,:)=[-(1+2*A1)/2,(1+8*C1)/8];
    Mbg(2,:)=[-3*D2,(1+B2)];
    Mbm(2,:)=[(1+8*C2)/8,-(7+24*A2)/24];
    Cb=Mbg\(Mbm);  %% SP3
    Cb_DE=Mbg(1,1)\Mbm(1,1); %% SP1
     us = Dm ./ (1-g);
     c1 = 1./(3.*(uam+(1-g).*us));
     c2 = 1./(7.*(uam+(1-g.^3).*us));
     a11 = uam; a12 = -2*uam/3; a21 = a12; a22 =  4*uam/9 + 5*(uam+(1-g.^2).*us)/9;