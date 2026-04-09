function tecplot_forward(nodes,surf_fym,elementInside)
% global  elementInside
        fid = fopen('mouseforward.dat','w');
        fprintf(fid,'%s\n','TITLE = "Example: FE-Volume Tetrahedral Data"  ');
        fprintf(fid,'%s\n','VARIABLES = "X", "Y", "Z", "Density"');
  % ================ 1 =============== %
        phantom_point(:,1:3)=nodes;
        phantom_point(:,4)=surf_fym(:,1);
        phantom_point_num = length(nodes(:,1));
        phantom_element = elementInside(:,2:5);
        fprintf(fid,'%s','ZONE N=');
        fprintf(fid,'%d',length(phantom_point(:,1)));
        fprintf(fid,'%s',', E=');
        fprintf(fid,'%d',length(phantom_element(:,1)));
        fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
        fprintf(fid,'%f\t %f\t %f\t %g\n',phantom_point(:,1:4).');
        fprintf(fid,'\n \n');
        fprintf(fid,' %d\t %d\t %d\t %d\n',phantom_element.');
        fprintf(fid,'\n \n');
        % ================ 2 =============== % 
        for organ=1:max(elementInside(:,1))
        heart_element = phantom_element(elementInside(:,1)==organ,:);
        if ~isempty(heart_element)
            fprintf(fid,'%s','ZONE N=');
            fprintf(fid,'%d',phantom_point_num);
            fprintf(fid,'%s',', E=');
            fprintf(fid,'%d',length(heart_element(:,1)));
            fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
            fprintf(fid,'%f\t %f\t %f\t %g\n',phantom_point(:,1:4).');   % 点坐标和能量值
            fprintf(fid,'\n \n');
            fprintf(fid,' %d\t %d\t %d\t %d\n',heart_element.');          % 单元信息
            fprintf(fid,'\n \n');
        end
        end
        fclose(fid);
