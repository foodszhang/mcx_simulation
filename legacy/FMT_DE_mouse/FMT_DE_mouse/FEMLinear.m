function [  Mm ] = FEMLinear ( D, U)
%FEMLINEAR Summary of this function goes here
%   Detailed explanation goes here
global nodes elementInside elementSurface An 
%初始化矩阵
M  =sparse(length(nodes(:,1)),length(nodes(:,1)));
% ***************************************************************
%                      将K和C加入到M中
% ***************************************************************

for l=1:length(elementInside(:,1))
    node=elementInside(l,2:5);
    sign1=elementInside(l,1);
 % 每个小四面体单元中的ai,bi,ci,di
    A(2:4,:)=nodes(node,:)';A(1,:)=ones(1,4);
    D0=abs(det(A));
    av1=-det(A([1 3 4],[2 3 4]));bv1=det(A([1 2 4],[2 3 4]));cv1=-det(A([1 2 3],[2 3 4]));
    av2=det(A([1 3 4],[1 3 4]));bv2=-det(A([1 2 4],[1 3 4]));cv2=det(A([1 2 3],[1 3 4]));
    av3=-det(A([1 3 4],[1 2 4]));bv3=det(A([1 2 4],[1 2 4]));cv3=-det(A([1 2 3],[1 2 4]));
    av4=det(A([1 3 4],[1 2 3]));bv4=-det(A([1 2 4],[1 2 3]));cv4=det(A([1 2 3],[1 2 3]));
    kl=(D(sign1)/(6.*D0))*[av1.*av1+bv1.*bv1+cv1.*cv1 av1.*av2+bv1.*bv2+cv1.*cv2 av1.*av3+bv1.*bv3+cv1.*cv3 av1.*av4+bv1.*bv4+cv1.*cv4
        av2.*av1+bv2.*bv1+cv2.*cv1 av2.*av2+bv2.*bv2+cv2.*cv2 av2.*av3+bv2.*bv3+cv2.*cv3 av2.*av4+bv2.*bv4+cv2.*cv4
        av3.*av1+bv3.*bv1+cv3.*cv1 av3.*av2+bv3.*bv2+cv3.*cv2 av3.*av3+bv3.*bv3+cv3.*cv3 av3.*av4+bv3.*bv4+cv3.*cv4
        av4.*av1+bv4.*bv1+cv4.*cv1 av4.*av2+bv4.*bv2+cv4.*cv2 av4.*av3+bv4.*bv3+cv4.*cv3 av4.*av4+bv4.*bv4+cv4.*cv4];
    cl=(D0)/120*ones(4);
    cl=cl+diag(diag(cl));%%%%%对角线上的值就是fyi=gyj时，是之前的两倍%%%%%
    for i = 1:4
        for j = 1:4
           M(node(i),node(j)) = M(node(i),node(j))+kl(i,j)+U(sign1)*cl(i,j);
        end
    end
end
% *****************************************************
% *********************加入边界条件*********************
% ****************************************************
for l=1:length(elementSurface(:,1))
    node=elementSurface(l,2:4);
    x=nodes(node,:);
    Ax=[1 1 1;x(1,1) x(2,1) x(3,1);x(1,2) x(2,2) x(3,2)];
    Ay=[1 1 1;x(1,1) x(2,1) x(3,1);x(1,3) x(2,3) x(3,3)];
    Az=[1 1 1;x(1,2) x(2,2) x(3,2);x(1,3) x(2,3) x(3,3)];
    D0=sqrt((det(Ax)).^2+(det(Ay)).^2+(det(Az)).^2);
    bl=(D0/(48.*An)).*ones(3);
    bl=bl+diag(diag(bl));
    bl=sparse(bl);
    for i = 1:3
        for j = 1:3
           M(node(i),node(j)) = M(node(i),node(j))+bl(i,j);
        end
    end
end
Mm=M;
