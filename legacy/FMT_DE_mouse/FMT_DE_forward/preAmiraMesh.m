function  [sizeNode, nodes, sizeInsideElement, elementInside, sizeSurfaceElement, elementSurface] = preAmiraMesh(File_1, File_2)

%输入Amira文件格式  输出netgen文件格式
%%从rm文件中提取nodes 和elementInside



fidr = fopen(File_1, 'rb');
while 1
    tline = fgetl(fidr);
    if ~ischar(tline),
        break,
    end
    [token,rem]=strtok(tline);
    if strcmp(token,'nNodes')
        numofNodes = str2double(rem);
        nodes = zeros(numofNodes,3);
    end
    if strcmp(token,'nTetrahedra')
        numofElement = str2double(rem);
        elementInside = zeros(numofElement,5);
    end
    if strcmp(token,'@1')
        for i = 1:numofNodes
            tline = fgetl(fidr);
            nodes(i,:) = str2num(tline);
        end
    end
    if strcmp(token,'@2')
        for i = 1:numofElement
            tline = fgetl(fidr);
            elementInside(i,2:5) = str2num(tline);
        end
    end
    if strcmp(token,'@3')
        for i = 1:numofElement
            tline = fgetl(fidr);
            elementInside(i,1) = str2double(tline);
            elementInside(i,1) = elementInside(i,1) + 1;
        end
    end
end
fclose(fidr);
disp(['       load .rm data end       ：   ',num2str(toc)])

%从elementInside中找到elementSurface 只属于一个elementInside的三角形是表面面片 
%生成netgen文件格式的数据 包含三个信息nodes elementInside elementSurface
p1 = [1 2 3;1 2 4;1 3 4;2 3 4];
elementSurface = [];
temp4 = sum(elementSurface,2);
[numofSurface,a] = size(elementSurface);
for i = 1:length(elementInside(:,1))
    temp1 = elementInside(i,2:5);
    temp2 = temp1(p1);
    temp3 = sum(temp2,2);
    for ii = 1:length(temp3)
        temp5 = find(temp4 == temp3(ii));
        sign = 0;
        for iii = 1:length(temp5)
            if sort(temp2(ii,:)) == sort(elementSurface(temp5(iii),1:3))
                elementSurface(temp5(iii),:) = [];
                temp4(temp5(iii)) = [];
                numofSurface = numofSurface-1;
                sign = 1;
                break;
            end
        end
        if sign == 0
            numofSurface = numofSurface+1;
            elementSurface(numofSurface,1:3) = temp2(ii,:);
            temp4(numofSurface) = sum(temp2(ii,:),2);
        end
    end
end

fidw1 = fopen(File_2,'w');
fprintf(fidw1,'%d\n',length(nodes(:,1)));
fprintf(fidw1,'%f\t %f\t %f\n',nodes');
fprintf(fidw1,'%d\n',length(elementInside(:,1)));
fprintf(fidw1,'%d\t %d\t %d\t %d\t %d\n', elementInside');
fprintf(fidw1,'%d\n',length(elementSurface(:,1)));
fprintf(fidw1,'%d\t %d\t %d\n',elementSurface');
fclose(fidw1);
disp(['       FIND SURFACE END        :  ',num2str(toc)])
MeshFile           =  File_2 ;
disp(['  data file is                 ：    ',num2str(MeshFile)])
fp = fopen(MeshFile);
sizeNode = fscanf(fp,'%d',1);
nodes = fscanf(fp,'%f %f %f',[3,sizeNode]);
nodes = nodes';
sizeInsideElement = fscanf(fp,'%d',1);
elementInside = fscanf(fp,'%d %d %d %d %d',[5,sizeInsideElement]);
elementInside = elementInside';
sizeSurfaceElement = fscanf(fp,'%d',1);
elementSurface = fscanf(fp,'%d %d %d ',[3,sizeSurfaceElement]);
elementSurface = elementSurface';
fclose(fp);