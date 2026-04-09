% -------------------------------------------------------------------------
% 步骤 1: 从体表元素中提取所有体表节点的全局索引 S
% -------------------------------------------------------------------------
clc;clear all;close all;
tic
global nodes elementInside elementSurface sizeNode sizeInsideElement sizeSurfaceElement Dx  Dm  uax  uam An g
% % *****************************读入数据********************************************
File_1  = '9029.am'; % amira输出的体网格（）
File_2  = '9029.txt'; % 输出的文件名
min_boundary=0.3;
max_boundary=33.3;
%*************************数据预处理**************************************%
[sizeNode, nodes, sizeInsideElement, elementInside, sizeSurfaceElement, elementSurface] = preAmiraMesh(File_1, File_2);
% [uax, Dx,uam, Dm, An,d] = Initialization;
[uam, Dm, An]= Initialization;
elementSurface=getsurface(min_boundary,max_boundary,elementSurface,nodes);
% 提取 elementSurface 中第 2 到第 4 列的所有节点索引
surface_nodes_raw = elementSurface(:, 2:4);
surface_nodes_vector = surface_nodes_raw(:);

% --- 关键步骤 2: 获取不重复的节点索引并排序 ---
% 使用 unique 函数获取所有不重复的体表节点全局索引。
% MATLAB的 unique 函数默认会从小到大排序。
S_global_indices = unique(surface_nodes_vector);

save('S_global_indice_9029.mat','S_global_indices');