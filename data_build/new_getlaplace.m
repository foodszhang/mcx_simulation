function [n_Lap1, n_Lap2, n_Lap3]=new_getlaplace
global nodes elementInside  
sizeNode=size(nodes,1);
adjacency_matrix1 = zeros(sizeNode,sizeNode);
adjacency_matrix2 = zeros(sizeNode,sizeNode);
adjacency_matrix3 =  zeros(sizeNode,sizeNode);
degree_matrix1 = zeros(sizeNode,sizeNode);
degree_matrix2 =  zeros(sizeNode,sizeNode);
degree_matrix3 =  zeros(sizeNode,sizeNode);
r1 =0.5;
r2 =1;
r3 = 1.5;
a = 1;

for i = 1:sizeNode
    r1 =0.1;
    flag_tru1 = 1;
    flag_tru2 = 1;
    while (flag_tru1 <4)
        flag_true1 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=r1);
        flag_tru1 = sum(length(flag_true1(:)));
       % if flag_tru1 <=4
            r1 = r1 + 0.01;
        %end
    end
    flag_true1 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=r1);
    flag_true2 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=1.5*r1);
    flag_true3 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=2*r1);
    adjacency_matrix1(i,flag_true1) = 1;
    adjacency_matrix1(flag_true1,i) = 1;
    adjacency_matrix2(i,flag_true2) = 1;
    adjacency_matrix2(flag_true2,i) = 1;
    adjacency_matrix3(i,flag_true3) = 1;
    adjacency_matrix3(flag_true3,i) = 1;
    degree_matrix1(i,i) = length(flag_true1) - 1;
    degree_matrix2(i,i) = length(flag_true2) - 1;
    degree_matrix3(i,i) = length(flag_true3) - 1;
end
adjacency_matrix1 = adjacency_matrix1-diag(diag(adjacency_matrix1));
adjacency_matrix2 = adjacency_matrix2-diag(diag(adjacency_matrix2));
adjacency_matrix3 = adjacency_matrix3-diag(diag(adjacency_matrix3));
n_Lap1 = eye(sizeNode,sizeNode) - (degree_matrix1^-0.5) * adjacency_matrix1 * (degree_matrix1^-0.5);
n_Lap2 = eye(sizeNode,sizeNode) - (degree_matrix2^-0.5) * adjacency_matrix2 * (degree_matrix2^-0.5);
n_Lap3 = eye(sizeNode,sizeNode) - (degree_matrix3^-0.5) * adjacency_matrix3 * (degree_matrix3^-0.5);

% save n_Lap1 n_Lap1
% save n_Lap2 n_Lap2
% save n_Lap3 n_Lap3

