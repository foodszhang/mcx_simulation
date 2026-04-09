function    tecplot(nodes,elementInside,boundary)
fym_surf=boundary;
S_point=[nodes fym_surf(:,1)];
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 fid = fopen('mouse_map.dat','w');
 fprintf(fid,'%s\n','TITLE = "Example: FE-Volume Tetrahedral Data"  ');
 fprintf(fid,'%s\n','VARIABLES = "X", "Y", "Z", "Density"');
     % ==================  输出光源信息  ============ %
 fprintf(fid,'%s','ZONE N=');
 fprintf(fid,'%d', length(nodes));
 fprintf(fid,'%s',', E=');
 fprintf(fid,'%d',length(elementInside));
 fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
 fprintf(fid,'%f\t %f\t %f\t %g\n',S_point.');         % 点坐标和能量值
 fprintf(fid,'\n \n');
  fprintf(fid,' %d\t %d\t %d\t %d\n',elementInside(:,2:5).');  
  