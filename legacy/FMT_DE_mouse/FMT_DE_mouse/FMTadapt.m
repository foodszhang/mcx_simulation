clear
close all                        
format long
global nodes elementInside elementSurface sizeNode sizeInsideElement sizeSurfaceElement Dx  Dm  uax  uam An c1 c2 a11 a12 a21 a22 Cb Cb_DE J0 J1 J2 J3 A1 B1 po
tic;
fprintf('\n')
disp('*********************************************************************')
disp('               *** S T A R T I N G R U N ***          ')
disp('********************************************************************')
time = clock;
disp([num2str(time(1)),'-',num2str(time(2)),'-',num2str(time(3)),' ',num2str(time(4)),'.',num2str(time(5)),'.',num2str(round(time(6)))])
disp(['               START           :    ',num2str(toc)])
%*********************************************************************************
%                       参数设置区
% *****************************读入数据********************************************
arpha = 0.0 ;          % 显示能量大于最大值arpha的单元
delta = 0.0;          % 设定的阈值,能量大于最大能量delta%的单元,need to be refinement 对重建结果影响很大！！
gama = 0.00  ;        % 噪声的百分比
S_site = [15 8 15];% 真实光源位置，若有改动，则refinemen_rule也要改
% center_m = [17.5  10.4  16.5];
min_boundary = 0.3;
max_boundary = 33.3;
num_esource = 1;
num_source = 1 ;            % 光源数目
forwardFile = '19468_liver_signle.grid.txt' ;
load  surf_fym
forwardfym=surf_fym;
File_1 = 'N_zlz(235,80,165)3.grid.am'; % amira输出的体网格
File_2 = 'N_zlz(235,80,165)3.grid.txt'; 
excite_nodes=importdata('single_15.txt');
%*************************数据预处理**************************************%
[sizeNode, nodes, sizeInsideElement, elementInside, sizeSurfaceElement, elementSurface] = preAmiraMesh(File_1, File_2);
[uax, Dx,uam, Dm, An,d] = Initialization1;
elementSurface=getsurface(min_boundary,max_boundary,elementSurface);
[po, g, An, A1, B1, J0, J1, J2, J3, Cb, Cb_DE, a11, a12, a21, a22, c1, c2] = caculate;
%*********************************激发过程********************************%
     [ Mx ]= FEMLinear(Dx, uax);
      [ Lx] = sorcesetup (num_esource,excite_nodes);%%%%%%% 这里需要根据光源的位置给出光源对应的向量,ok了 %%%%
     fyx = zeros(length(nodes(:,1)),num_esource);
     for i1 = 1:num_esource
          fyx(:,i1) = Mx\Lx(:,i1); %%% 前向问题fy，也即第一个耦合方程的解%%%
     end
     clear Mx; clear Lx;
%************************************发射过程*****************************%
     sign = ones(length(elementInside(:,1)),1);
       [b_analytic] = mosetomesh1(nodes,num_esource,elementSurface,forwardFile,forwardfym);%%%%%把mose的值映射到网格点上
%       [b_analytic] = mosetomesh(nodes,num_esource,forwardFile,forwardfym);%%%%%把mose的值映射到网格点上
%      [flagFOV,  region0] = MoseSurface( nodes, elementSurface,num_esource,center_m);%%%%%表面的数据处理成外围边界的值 
     [ Mm ] = FEMLinear(Dm, uam);
     [ F ] = FormMatrixF(sign,fyx);
     B = Mm\F;
     b1=b_analytic;
     A=[];
     fym = [];
          for i2=1:num_esource        
            measure_sur=b1(:,i2);
            index=find(measure_sur>0);  
%             measure_sur=b1.*flagFOV(:,i2);
%             noise = ((sum(measure_sur.*measure_sur)).^0.5/length(measure_sur)).*randn(length(measure_sur),1);
%             measure_sur = measure_sur+gama*noise;
            fym = cat(1,fym,measure_sur(index,:));
            A=cat(1,A,B(index,:));
          end
     clear Mm; clear F;
%*****************************优化处理***********************************%
        t1 = cputime;
        tau = 1e-4;
        err = 1e-10;
        [output,nv_ITCG_vs,T_ITCG_vs,iter_vs,NZ_vs] =ITCG_vs(fym,A,tau,err);
%         E=output;
        t0 = cputime-t1;
        [vector1,sign1,elementInside,phantom_point0] = norm_refinement_rule(output,sign,elementInside,nodes,delta,arpha,num_source,S_site ); 
        disp(['No mesh cost      :    ',num2str(toc)])
        disp(['第一次要细分的单元数目为  ：  ',num2str(length(vector1))])