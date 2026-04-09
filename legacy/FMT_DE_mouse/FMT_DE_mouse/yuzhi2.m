function [output]=yuzhi2(output,GESHU)
         GESHU1=GESHU-2;      %最后结果不为0的结果应该大于n-2，此处设为2，只要有一个能量值就可以
          TZ=1;
        outputsum=0;
        number1=length(find(output(:,1)~=0)) 
        outputp=output;
        for i=1:length(outputp)
            outputp(i,2)=i;     %标记每个能量值的位置
        end
    
        outputC=sortrows(outputp,-1);      %排序
        for i=1:length(outputC(:,1))-1
            Dvalue(i,1)=outputC(i,1)-outputC(i+1,1);     %求差值
        end
        LEN=length(Dvalue);
        outputD=outputC;
        I=0;
        II=0;
%         
        for i=1:LEN
             II=I;
             outputD=outputC;
             [M,I]=max(Dvalue);      %查找最大的差值
             for j=1:length(outputD(:,1))
                 if(j>I)
                      outputD(j,1)=0;         %保留阈值内的点
                    
                 end
             end
            number2=length(find(outputD(:,1)~=0));
            
            
%%%%%%%%%判断能量值的个数是否符合要求%%%%%%%%%%%
             if number2<=i+GESHU1
                Dvalue(I,1)=0;           
%      
             else
                for j=1:length(outputC(:,1))
                     if j>I
%                 
                         outputC(j,1)=0;         
                     end      
                end
                break
%                 Mvalue = outputC(I,1);
%                  i=LEN;
             end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
%        
        outputA1=sortrows(outputC,2);      %回到初始位置
        output=outputA1(:,1);
        number2=length(find(output(:,1)~=0))