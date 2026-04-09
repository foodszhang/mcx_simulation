function A = getA(nodes,P)
sizeNode = length(nodes(:,1));
P11 = P(1:sizeNode,1:sizeNode);
P12 = P(1:sizeNode,sizeNode+1:2*sizeNode);
P21 = P(sizeNode+1:2*sizeNode,1:sizeNode);
P22 = P(sizeNode+1:2*sizeNode,sizeNode+1:2*sizeNode);
clear P;
A = (P11-2/3*P12) + (P21-2/3*P22);
save A.mat A;
