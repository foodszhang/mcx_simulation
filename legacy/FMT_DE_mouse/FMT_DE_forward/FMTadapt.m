clear
close all                        
format long
global nodes elementInside elementSurface sizeNode sizeInsideElement sizeSurfaceElement Dx  Dm  uax  uam An g c1 c2 a11 a12 a21 a22 Cb Cb_DE J0 J1 J2 J3 A1 B1 po
tic;
fprintf('\n')
disp('*********************************************************************')
disp('               *** S T A R T I N G R U N ***          ')
disp('********************************************************************')
time = clock;
disp([num2str(time(1)),'-',num2str(time(2)),'-',num2str(time(3)),' ',num2str(time(4)),'.',num2str(time(5)),'.',num2str(round(time(6)))])
disp(['               START           :    ',num2str(toc)])
%**************************************************************************
%                       参数设置区
% *****************************读入数据********************************************
%File_1  = '19468_liver_signle_source.grid.am'; % amira输出的体网格
%File_2  = '19468_liver_signle_source.grid.txt'; 
File_1  = '19468_liver_signle.grid.am'; % amira输出的体网格
File_2  = '19468_liver_signle.grid.txt'; 
excite_nodes=importdata('single_15.txt');%%%读数据
min_boundary=0.3;
max_boundary=33.3;
num_esource = 1;
%*************************数据预处理**************************************%
[sizeNode, nodes, sizeInsideElement, elementInside, sizeSurfaceElement, elementSurface] = preAmiraMesh(File_1, File_2);
elementSurface=getsurface(min_boundary,max_boundary,elementSurface);
[uax, Dx,uam, Dm, An,d] = Initialization1;
[po, g, An, A1, B1, J0, J1, J2, J3, Cb, Cb_DE, a11, a12, a21, a22, c1, c2] = caculate;
%**************************激发过程***********************%
     [ Mx ] = FEMLinear(Dx, uax) ;
     [ Lx] = sorcesetup(num_esource,excite_nodes);%%%%%%% 这里需要根据光源的位置给出光源对应的向量,ok了 %%%%
     for i1=1:num_esource
          fyx(:,i1)=Mx\Lx(:,i1); %%% 前向问题fy，也即第一个耦合方程的解%%%
     end
     clear Mx; clear Lx;
%**************************发射过程***********************%
     sign=ones(length(elementInside(:,1)),1);
     [ Mm ] = FEMLinear(Dm, uam);
     fym=zeros(length(nodes(:,1)),num_esource);
     Bm=cell(num_esource,1);
     X=given(elementInside,nodes);
     for i2=1:num_esource
     [ F ] = FormMatrixF( fyx(:,i2),sign);
          Bm{i2}=F*X;
          fym(:,i2)=Mm\Bm{i2};
     end
       clear Mm; clear F;
%*****************************表面能量提取*********************************%
     surf_fym=get_surface_energy(fym,elementSurface,num_esource);
%************************前向结果展示*****************************
     tecplot_forward(nodes,surf_fym);
%******************结果保存*********************
     save  surf_fym  surf_fym
     
