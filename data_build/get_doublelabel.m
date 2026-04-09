function [Label_Data,d ]=get_doublelabel
global nodes   sizeNode
X = zeros(sizeNode,1);
x = 6.5:0.5:9;
y =12.5:0.1:13.5;
z =18.5;

x1 = 6.5:0.5:9.5;
y1 =12.5:0.1:13.5;
z1 =18.5;

% x = 12:0.6:15;
% y = 7:0.1:7.5;
% z = 15.5:0.2:17.5;
%
% x1 = 21:0.2:22;
% y1 = 7:0.1:7.5;
% z1 = 15.5:0.2:17.5;
r =0.8:0.2:1.6;

Len_x = length(x);
Len_y = length(y);
Len_z = length(z);
Len_r = length(r);
dis=2;
f=waitbar(0,'双光源数据生成中...');
for i = 1 : Len_x
    site1(1,1) =  x(1,i);
    site2(1,1) =  x1(1,i);
    for j = 1 : Len_y
        site1(1,2) =   y(1,j);
        site2(1,2) =   y1(1,j);
        for k = 1 : Len_z
            site1(1,3) =   z(1,k);
            site2(1,3) =   z1(1,k);
            for l = 1 : Len_r
                hh=[fix(2*rand) fix(2*rand) fix(2*rand)];
                if norm(hh)>0
                    S_site1=S_site+hh.*(dis+r(1,l));

                    flag_true1 = find(sum((nodes(:,1:3)-site1).^2,2).^0.5<=r(1,l));
                    flag_true2 = find(sum((nodes(:,1:3)-site2).^2,2).^0.5<=r(1,l));
                    index = cat(1,flag_true1,flag_true2);
                    X(index) = 1;
                    d = d + 1;
                    S_site1(d,1:3) = site1;
                    S_site1(d,4:6) = site2;
                    S_site1(d,7) =  r(1,l);
                    Label_Data(d,:) = X;
                    X = zeros(sizeNode,1);
                end
            end
        end
    end
    str=['计算中已完成',num2str(100*i/Len_x),'%'];
    waitbar(i/Len_x,f,str);
end
close(f);
c = sum(Label_Data,2);
index = find(c == 0);
Label_Data(index,:) = [];
S_site1(index,:) = [];