function [Label_Data,test_b]=get_signlelabel(A,test)
global nodes   sizeNode
X=zeros(sizeNode,1);
if test==0
x=10.5:0.5:12.5;
% x=10.5:0.5:12.0;
y=10:0.5:14;
% y=10:0.5:12;
z=16:0.5:19;
% z=16:0.5:18;
r =1.0:0.20:2.0;
% r =1.0:0.20:1.6;
else
x=11.5;
% y=[11 12 13];
y = 11;
z=17;
r=1.0;
end

Label_Data=[];
f=waitbar(0,'单光源数据生成中...');
for nn=1:length(z)
    c=0;
    S_site(1,3) = z(nn);
    Label_Data1=[];
    % z=12:0.5:22+nn;
    %z=22:1:32;
    % x=13:0.5:23; y=6:0.2:8; z=15:0.5:18;
    Len_x = length(x);
    Len_y = length(y);
    % Len_z = length(z); z = [16.2  16.8  17.2 17.8  18.2];

% 
    Len_r = length(r);

    for i = 1 : Len_x
        S_site(1,1) = x(1,i);
        for j = 1 : Len_y
            S_site(1,2) = y(1,j);
            %         for k = 1 : Len_z
            for l = 1 : Len_r
                flag_true1 = find(sum((nodes(:,1:3)-S_site).^2,2).^0.5<=r(1,l));
                index = flag_true1;
                
                X(index) = 1;
                c = c + 1;
                S_site1(c,1:3) = S_site;
                S_site1(c,4) =  r(1,l);
                Label_Data1(c,:) = X;
                X = zeros(sizeNode,1);
                %             end
            end
        end

    end
    Label_Data=sparse([Label_Data;Label_Data1]);
    str=['计算中已完成',num2str(100*nn/length(z)),'%'];
    waitbar(nn/length(z),f,str);
end
%save Lable_Data Label_Data close(f);
Label_Data=full(Label_Data);
test_b=max(A*Label_Data',0);
% [~,index1]=find(test_b<0)
c = sum(Label_Data,2);c1=sum(test_b,1);
index= find(c <= 0 & c1'<=0 );
Label_Data(index,:) = []; test_b(:,index)=[];
