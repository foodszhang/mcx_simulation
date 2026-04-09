function [flagFOV,  region0] = MoseSurface( nodes_b, elementSurface_b,num_esource,center_m)
nodeSign=zeros(length(nodes_b(:,1)),1);
a=elementSurface_b(:,2:4);
nodeSign(a)=1;
region0=find(nodeSign==1);
N=num_esource;
n=2*pi/N;
center0=center_m;
nodes1=nodes_b; 
nodes_count=length(nodes1(:,1));
sur_theta=zeros(length(region0(:,1)),1);
sur_nodes=nodes1(region0,:);
cen_nodes=zeros(size(sur_nodes));
cen_nodes(:,1)=center0(1);
cen_nodes(:,2)=center0(2);
cen_nodes(:,3)=center0(3);
nodes2=sur_nodes-cen_nodes;
flagFOV=zeros(nodes_count,N);

re1=find(nodes2(:,2)>=0 & nodes2(:,1)>0);
re2=find(nodes2(:,2)>=0 & nodes2(:,1)<0);
re3=find(nodes2(:,2)<0 & nodes2(:,1)<0);
re4=find(nodes2(:,2)<0 & nodes2(:,1)>0);
re5=find(nodes2(:,1)==0 & nodes2(:,2)>0);
re6=find(nodes2(:,1)==0 & nodes2(:,2)<0);
tan_y_x=nodes2(:,2)./nodes2(:,1);
tan_theta=atan(tan_y_x);

sur_theta(re1)=tan_theta(re1);
sur_theta(re2)=tan_theta(re2)+pi;
sur_theta(re3)=tan_theta(re3)+pi;
sur_theta(re4)=tan_theta(re4)+2*pi;
sur_theta(re5)=tan_theta(re5);
sur_theta(re6)=tan_theta(re6)+2*pi;
% flag=[];
for i=1:N
    single_theta=(i-1)*n;
    single_theta1=single_theta+2*pi/3;
    single_theta2=single_theta+4*pi/3;
    if single_theta2>2*pi
        single_theta2=single_theta2-2*pi;
    end
    if single_theta1>2*pi
        single_theta1=single_theta1-2*pi;
    end   
    if single_theta1>single_theta2
        sur_region=find((sur_theta>=single_theta1 & sur_theta<=2*pi)|(sur_theta>0 & sur_theta<=single_theta2));
        flag=region0(sur_region);
        flagFOV(flag,i)=1;
    else 
        sur_region=find(sur_theta>=single_theta1 & sur_theta<=single_theta2);
        flag=region0(sur_region);
        flagFOV(flag,i)=1;
    end
end


