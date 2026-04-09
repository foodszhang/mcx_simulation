function [vector1,sign1,elementInside,phantom_point0]=refinement_rule(output,sign,elementInside,nodes,delta,arpha,num_source, S_site )
    s =   output;
%    S_site_above=S_site11;
    S_site_underside=S_site;
    phantom_point_num = length(nodes(:,1));
    elementInside(:,12) = sign;   
    sign1 = zeros(length(elementInside(:,1)),1);
    % =========================== 9-35是将产额赋予到对应点坐标上   ==================== %
    P_E=zeros(phantom_point_num,1);   % 各点的能量值
    sSign=zeros(length(nodes(:,1)),1);
    for i=1:length(elementInside(:,1))
        if elementInside(i,12) == 1
            b = elementInside(i,2:5);
            sSign(b) = 1;
        end
    end
    sSign = find(sSign==1);
    P_E(sSign) = s;            % sSign为光源可行区节点编号，s是计算出的能量
    phantom_point =  zeros(phantom_point_num,4);
    phantom_point(:,1:3) = nodes(:,1:3);    % 仿体各点的坐标值
    phantom_point(:,4) = P_E;               % 仿体各点的能量值（光流密度）
    phantom_point0=phantom_point;
%     save phantom_point.mat phantom_point
    nodes = phantom_point; 
    nodes(:,4) = [];
    if num_source == 2
        nodes_1 = nodes;
        nodes_2 = nodes;
        for i = 1:length(nodes(:,1))
            if nodes(i,2) >= 8.65
                nodes_1(i,4) = phantom_point(i,4);
            elseif nodes(i,2) < 8.65
                nodes_2(i,4) = phantom_point(i,4);
            end
        end
    end    
    
    plantom_element = elementInside(:,2:5);
%     fe_point_num = length(sSign);
%     fe_point = phantom_point(sSign,:);        %fe_point包括坐标和能量值
    flag4 = find(elementInside(:,12)==1);                     %%%%%                               可能有问题
%     fe_element_num = length(flag4);           % 可行光源区的单元数
    fe_element = plantom_element(flag4,:);
     % =====================42-end判断能量大于某阈值的的数据单元 ================= %
    S_Dense = zeros(length(fe_element(:,1)),1);
    S_Dense_1 = zeros(length(fe_element(:,1)),1);
    S_Dense_2 = zeros(length(fe_element(:,1)),1);
    for i = 1:length(elementInside(:,1))
            if elementInside(i,12) == 1
                node = elementInside(i,2:5);
                A0(2:4,:) = nodes(node,1:3)';
                A0(1,:) = ones(1,4);
%                 V = abs(det(A0))/6;
                S_Dense(i) = 0.25*sum(phantom_point(node,4)); % S_Dense为可行区每个单元的荧光产额，取了各个顶点的平均值
                if num_source == 2
                    if mean(A0(3,:)) >= 8.65
                        S_Dense_1(i) = 0.25*sum(phantom_point(node,4));
                    elseif mean(A0(3,:)) < 8.65
                        S_Dense_2(i) = 0.25*sum(phantom_point(node,4));
                    end
                end
            end
    end
    if num_source == 1
        vector1 = find(S_Dense>delta*max(S_Dense));   %%%%%%返回的vector1是四面体单元编号%%%%
        subnonz_index = find (S_Dense>arpha*max(S_Dense));
    elseif num_source == 2
        vector1_1 = find (S_Dense_1>delta*max(S_Dense_1));
        vector1_2 = find (S_Dense_2>delta*max(S_Dense_2));
        vector1 = [vector1_1;vector1_2];
        a=find (S_Dense_1>arpha*max(S_Dense_1));
        b=find (S_Dense_2>arpha*max(S_Dense_2));
        subnonz_index =[a;b];
    end
    save S_Dense.mat S_Dense
%     subnonz_point_num = phantom_point_num;
%     subnonz_point = phantom_point;
    subnonz_element = elementInside(subnonz_index,2:5);
    subnonz_element_num =  length(subnonz_index);     % 单元数
    sign2=zeros(length(nodes(:,1)),1);
    sign2(subnonz_element)=1;
    phantom_point(:,4)=phantom_point(:,4).*sign2;
    %%%%%79-97下面是将 vector1单元及其周围的单元都作为可行区
    temp1 = zeros(length(nodes(:,1)),1);
    for i = 1:length(vector1)
          a = elementInside(vector1(i),2:5);
          temp1(a) = 1;
    end
    for i = 1:length(elementInside(:,1))
         if elementInside(i,12) == 1
             b = sum(temp1(elementInside(i,2:5)));
             if b>1
                 sign1(i) = 1;
             end
        end
    end
    elementInside(:,12) = sign1;
    vector1 = find(sign1==1);
    sign1 = zeros(length(elementInside(:,1)),1);
    sign1(vector1) = 1;
%     save vector1.mat vector1
%     save sign.mat sign1
    % =====================读入实际的荧光数据=========================== %
  if num_source == 1
        fid = fopen('19468_liver_signle_source.grid.txt','r');
        S_point_num = fscanf(fid,'%f',1);                 % 光源区域的点数
        S_point_coord = fscanf(fid,'%f',[3,S_point_num]); % 读入光源点坐标
        S_point_coord = S_point_coord.';
        S_element_num = fscanf(fid,'%f',1);               % 光源区域的单元数
        S_element = fscanf(fid,'%f',[5,S_element_num]);   % 读入光源单元数据
        S_element = S_element.';
        S_element = S_element(:,2:5);
        fclose(fid);
  elseif num_source == 2
        fid = fopen('double.geo','r');
        S_point_num = fscanf(fid,'%f',1);                 % 光源区域的点数
        S_point_coord = fscanf(fid,'%f',[3,S_point_num]); % 读入光源点坐标
        S_point_coord = S_point_coord.';
        S_element_num = fscanf(fid,'%f',1);               % 光源区域的单元数
        S_element = fscanf(fid,'%f',[5,S_element_num]);   % 读入光源单元数据
        S_element = S_element.';
        S_element = S_element(:,2:5);
        fclose(fid);
  end
    S_point = [S_point_coord zeros(S_point_num,1)];       % 最后一列表示能量密度
    %%%%%% 以下是将荧光数据写成tecplot的必须的形式，从而用tecplot读取显示 %%%%%
    % ==================   文件输出   ============== %
    fid = fopen('mouseinverse.dat','w');
    fprintf(fid,'%s\n','TITLE = "Example: FE-Volume Tetrahedral Data"  ');
    fprintf(fid,'%s\n','VARIABLES = "X", "Y", "Z", "Density"');
     % ==================  输出光源信息  ============ %
    fprintf(fid,'%s','ZONE N=');
    fprintf(fid,'%d',S_point_num);
    fprintf(fid,'%s',', E=');
    fprintf(fid,'%d',S_element_num);
    fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
    fprintf(fid,'%f\t %f\t %f\t %f\n',S_point.');         % 点坐标和能量值
    fprintf(fid,'\n \n');
    fprintf(fid,' %d\t %d\t %d\t %d\n',S_element.');      % 单元信息
    % ================ 能量大于某个阈值的数据 =============== %
    fprintf(fid,'%s','ZONE N=');
    fprintf(fid,'%d',phantom_point_num);
    fprintf(fid,'%s',', E=');
    fprintf(fid,'%d',subnonz_element_num);
    fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
    fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point.');    % 点坐标和能量值
    fprintf(fid,'\n \n');
    fprintf(fid,' %d\t %d\t %d\t %d\n',subnonz_element.'); % 单元信息
    fprintf(fid,'\n \n');
    %=========================================================================%
    % 计算光源位置
    if num_source == 1
        num = find(S_Dense==max(S_Dense));
%         S_site0 = 0.25*(1/length(num))*sum(nodes(elementInside(num,2:5),:));
        temp = phantom_point(elementInside(num,2:5)',4);%% phantom_point包括坐标和能量值
        S_site1 = 0;
        temp2 = reshape(elementInside(num,2:5)',4*length(num),1);%%% 变成一列元素
        for i = 1 : length(temp2)
            S_site1 = S_site1+temp(i).*nodes(temp2(i),:);
        end
        S_site1 = S_site1/sum(temp);
        disp(['荧光产额加权中心点坐标            ：  ',num2str(S_site1)])
        [a b] = max(phantom_point(:,4));%%% 显示最大值和所在行
        S_site3 = nodes(b,1:3);
        disp(['荧光产额最大单元的编号为          ： ',num2str(num')])
        disp(['maxS = ',num2str(max(s))])
        disp(['荧光产额最大点的坐标              ： ',num2str(S_site3)])
        disp(['荧光团的真实位置为              ： ',num2str(S_site)])
%         distence0 = norm(S_site-S_site0);
        distence1 = norm(S_site-S_site1);
        distence3 = norm(S_site-S_site3);
        disp(['荧光产额最大点到真实光源的距离     ： ',num2str(distence3)])
        disp(['中心点到荧光团的距离     ： ',num2str(distence1)])
        % 计算总能量
        sumS =0;  % 总能量
%         disp(['重建后的总能量为              ： ',num2str(sumS)])
        V2=0;
         for i = 1:length(S_Dense)
                if S_Dense(i) ~=0
                    node = elementInside(i,2:5);
                    A0(2:4,:) = nodes(node,:)';
                    A0(1,:)=ones(1,4);
                    V0=abs(det(A0))/6;
                    V2 = V2+V0;
                    sumS = sumS+ S_Dense(i)*V0;
                end
         end
         disp(['重建后平均荧光产额为              ： ',num2str(sumS/V2)])
         
     elseif num_source == 2
        disp(['位置为',num2str(S_site_above),'的荧光团的重建信息  ： '])
        [a b] = max(nodes_1(:,4));
        disp(['maxS = ',num2str(a)])
        disp(['荧光产额最大点的坐标              ：  ',num2str(nodes_1(b,1:3))])
        disp(['荧光产额最大点到荧光团的距离         ：  ',num2str(norm(nodes_1(b,1:3)-S_site_above))])
        num1 = find(S_Dense_1==max(S_Dense_1));
%         S_site0 = 0.25*(1/length(num))*sum(nodes(elementInside(num,2:5),:));
        temp = phantom_point(elementInside(num1,2:5)',4);
        S_site1 = 0;
        temp2 = reshape(elementInside(num1,2:5)',4*length(num1),1);
        for i = 1 : length(temp2)
            S_site1 = S_site1+temp(i).*nodes(temp2(i),:);
        end
        S_site1 = S_site1/sum(temp);
        disp(['荧光产额加权中心点坐标            ：  ',num2str(S_site1)])
        disp(['中心点1到荧光团的距离         ：  ',num2str(norm(S_site1-S_site_above))])
        disp(['荧光团的真实位置为              ： ',num2str(S_site_above)])
        
        disp('================================================================')
        disp(['位置为',num2str(S_site_underside),'的荧光团的重建信息  ： '])
        [a1 b1] = max(nodes_2(:,4));
        disp(['maxS = ',num2str(a1)])
        disp(['荧光产额最大点的坐标              ：  ',num2str(nodes_2(b1,1:3))])
        disp(['荧光产额最大点到荧光团的距离         ：  ',num2str(norm(nodes_2(b1,1:3)-S_site_underside))])
        num2 = find(S_Dense_2==max(S_Dense_2));
%         S_site0 = 0.25*(1/length(num))*sum(nodes(elementInside(num,2:5),:));
        temp = phantom_point(elementInside(num2,2:5)',4);
        S_site2 = 0;
        temp2 = reshape(elementInside(num2,2:5)',4*length(num2),1);
        for i = 1 : length(temp2)
            S_site2 = S_site2+temp(i).*nodes(temp2(i),:);
        end
        S_site2 = S_site2/sum(temp);
        disp(['荧光产额加权中心点坐标            ：  ',num2str(S_site2)])
        disp(['中心点2到荧光团的距离         ：  ',num2str(norm(S_site2-S_site_underside))])
        disp(['荧光团的真实位置为              ： ',num2str(S_site_underside)])
    else
        error('The num_source is wrong! Please input again');
    end
 %    num=[num1;num2];
    % ================ 最大四面体位置 ========= %
    max_element=elementInside(num,2:5);
    max_element_num = length(max_element(:,1));
    fprintf(fid,'%s','ZONE N=');
    fprintf(fid,'%d',phantom_point_num);
    fprintf(fid,'%s',', E=');
    fprintf(fid,'%d',max_element_num);
    fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
    fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point.');             % 点坐标和能量值
    fprintf(fid,'\n \n');
    fprintf(fid,' %d\t %d\t %d\t %d\n',max_element.');          % 单元信息
    fprintf(fid,'\n \n');
  % ================  可行区的数据  ============ %
    fe_element1 = elementInside(elementInside(:,12)==1,2:5);
    fe_element1_num = length(fe_element1(:,1));
    fprintf(fid,'%s','ZONE N=');
    fprintf(fid,'%d',phantom_point_num);
    fprintf(fid,'%s',', E=');
    fprintf(fid,'%d',fe_element1_num);
    fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
    fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point.');             % 点坐标和能量值
    fprintf(fid,'\n \n');
    fprintf(fid,' %d\t %d\t %d\t %d\n',fe_element1.');          % 单元信息
    fprintf(fid,'\n \n');
    % ================ 仿体数据 =============== %
    phantom_point_num = length(nodes(:,1));
    phantom_element = elementInside(:,2:5);
    fprintf(fid,'%s','ZONE N=');
    fprintf(fid,'%d',phantom_point_num);
    fprintf(fid,'%s',', E=');
    fprintf(fid,'%d',length(phantom_element(:,1)));
    fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
    fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point(:,1:4).');   % 点坐标和能量值
    fprintf(fid,'\n \n');
    fprintf(fid,' %d\t %d\t %d\t %d\n',phantom_element.');          % 单元信息
    fprintf(fid,'\n \n');
        % ================ 2 =============== %        
        heart_element = phantom_element(elementInside(:,1)==2,:);
        if ~isempty(heart_element)
            fprintf(fid,'%s','ZONE N=');
            fprintf(fid,'%d',phantom_point_num);
            fprintf(fid,'%s',', E=');
            fprintf(fid,'%d',length(heart_element(:,1)));
            fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
            fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point(:,1:4).');   % 点坐标和能量值
            fprintf(fid,'\n \n');
            fprintf(fid,' %d\t %d\t %d\t %d\n',heart_element.');          % 单元信息
            fprintf(fid,'\n \n');
        end
        % ================ 3 =============== %
        stomach_element = phantom_element(elementInside(:,1)==3,:);
        if ~isempty(stomach_element)
            fprintf(fid,'%s','ZONE N=');
            fprintf(fid,'%d',phantom_point_num);
            fprintf(fid,'%s',', E=');
            fprintf(fid,'%d',length(stomach_element(:,1)));
            fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
            fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point(:,1:4).');   % 点坐标和能量值
            fprintf(fid,'\n \n');
            fprintf(fid,' %d\t %d\t %d\t %d\n',stomach_element.');          % 单元信息
            fprintf(fid,'\n \n');
        end
        % ================ 4 =============== %
        liver_element = phantom_element(elementInside(:,1)==4,:);
        if ~isempty( liver_element)
            fprintf(fid,'%s','ZONE N=');
            fprintf(fid,'%d',phantom_point_num);
            fprintf(fid,'%s',', E=');
            fprintf(fid,'%d',length(liver_element(:,1)));
            fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
            fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point(:,1:4).');   % 点坐标和能量值
            fprintf(fid,'\n \n');
            fprintf(fid,' %d\t %d\t %d\t %d\n',liver_element.');          % 单元信息
            fprintf(fid,'\n \n');
        end
        % ================ 5 =============== %
        kidney_element = phantom_element(elementInside(:,1)==5,:);
        if ~isempty(kidney_element)
            fprintf(fid,'%s','ZONE N=');
            fprintf(fid,'%d',phantom_point_num);
            fprintf(fid,'%s',', E=');
            fprintf(fid,'%d',length(kidney_element(:,1)));
            fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
            fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point(:,1:4).');   % 点坐标和能量值
            fprintf(fid,'\n \n');
            fprintf(fid,' %d\t %d\t %d\t %d\n',kidney_element.');          % 单元信息
            fprintf(fid,'\n \n');
        end
        % ================ 6 =============== %
        lung_element = phantom_element(elementInside(:,1)==6,:);
        if ~isempty(lung_element)
            fprintf(fid,'%s','ZONE N=');
            fprintf(fid,'%d',phantom_point_num);
            fprintf(fid,'%s',', E=');
            fprintf(fid,'%d',length(lung_element(:,1)));
            fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
            fprintf(fid,'%f\t %f\t %f\t %f\n',phantom_point(:,1:4).');   % 点坐标和能量值
            fprintf(fid,'\n \n');
            fprintf(fid,' %d\t %d\t %d\t %d\n',lung_element.');          % 单元信息
            fprintf(fid,'\n \n');
        end
        fclose(fid);
% =========================================================================
    disp('********************************************************************')
    disp('                 *** E N D O F R U N ***                            ')
    disp('********************************************************************')
% end
fprintf('\n')
