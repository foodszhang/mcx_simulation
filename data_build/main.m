clc;clear;close all;
global nodes elementInside elementSurface sizeNode sizeInsideElement sizeSurfaceElement Dx  Dm  uax  uam An g
% % *****************************读入数据********************************************
File_1  = '9029_with_cancer.am'; % amira输出的体网格（）
File_2  = '9029_with_cancer.txt'; % 输出的文件名
min_boundary=0.3;
max_boundary=33.3;
%*************************数据预处理**************************************%
[sizeNode, nodes, sizeInsideElement, elementInside, sizeSurfaceElement, elementSurface] = preAmiraMesh(File_1, File_2);
% [uax, Dx,uam, Dm, An,d] = Initialization;
[uam, Dm, An]= Initialization;
elementSurface=getsurface(min_boundary,max_boundary,elementSurface,nodes);
fyx=ones(length(nodes(:,1)),1);
sign=ones(length(elementInside(:,1)),1);
[ F ] = FormMatrixF(fyx(:,1),sign);
[ Mm ] = FEMLinear(Dm, uam,nodes ,elementInside ,elementSurface, An);
A =full(Mm\F);
savePath=['recon_data_9029_with_cancer'];
save(savePath,'elementInside','A','nodes','elementSurface','sizeInsideElement','sizeNode');
%*********************标签*********************
% load recon_head_data.mat
load recon_data_9029_with_cancer;
%*******i=1单光源 i = 2 单双光源*******
i=1;
test=1;
index = unique(elementSurface(:,2:4));
A = A(index,:);

% clear Label_Data1; clear test_b1; clear Label_Data2; clear test_b2;
%*********************拉普拉斯矩阵*********************
% 
Lap=getlaplace;
[n_Lap0,n_Lap1, n_Lap2, n_Lap3]=new_getlaplace_GHB;
%******************结果保存*********************
savePath='recon_info_9029_with_cancer';
save(savePath,'A','Lap','n_Lap0','n_Lap1', 'n_Lap2', 'n_Lap3','nodes','elementInside','elementSurface','sizeInsideElement');
