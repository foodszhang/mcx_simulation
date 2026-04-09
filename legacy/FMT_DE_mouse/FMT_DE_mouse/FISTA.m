function [x_ITCG_vs,T_ITCG_vs,iter_vs,NZ_vs,yy,L,ferrr]=FISTA(y,A1,tau,err)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a code for sparse reconstruction through applying 
% programming min 1/2||Ax-y||_2^2+tau ||x||_1.
% %% imput %%%%%%%%%%
% y: the length-K observation vector 
%       (y=Af, f is the original sparse signal),
% A: the measurement matrix with size KxN,
% tau: the regularization parameter,
% err: the termination tolerance
%       (when ||v||_2<=err, the ITCG-vs stops)
%       (see the paper for the definition of v).
% %% output %%%%%%%%%%%
% x_ITCG_vs: the solution of min 1/2||Ax-y||_2^2+tau ||x||_1
%       computed by ITCG-vs, it is an approximation of f,
% nv_ITCG_vs: the value of ||v^k||_2 when ITCG-vs stops,
% T_ITCG_vs: the CPU time consumed by ITCG-vs,
% iter_vs: the number of iterations needed by ITCG-vs,
% NZ_vs: the number of nonzero entries of x_ITCG_vs.





maxIter = 2000;

[m,n]=size(A1);
A = A1./repmat(sqrt(sum(A1.^2,1)),m,1);
%b=A'*y;%相当于FISTA中的c

%A=[A,-A];

t_kF=1;
t_km1 = 1 ;
L0 = 1 ;
G = A'*A; 
nIter = 0 ;
c_F = A'*y ;
lambda0 = 0.5*L0*norm(c_F,inf) ;
eta = 0.95 ;
xk = zeros(n,1);
lambdaF = lambda0 ;
L_F =L0 ;
beta_F = 1.5;

keep_going = 1 ;
%nz_x = (abs(xk)> eps*10);
f = 0.5*norm(y-A*xk)^2 + tau * norm(xk,1);
xkm1 = xk;
t0 = tic ;

while keep_going && (nIter < maxIter)
    nIter = nIter + 1 ;
    
    yk = xk + ((t_km1-1)/t_kF)*(xk-xkm1) ;
    
    stop_backtrack = 0 ;
    
    temp = G*yk - c_F ; % gradient of f at yk
    % gk = yk - (1/L_F)*temp ;
%     iter_sign=1;
%     i=1;
    while ~stop_backtrack
        
        gk = yk - (1/L_F)*temp ;
        
        xkp1 = soft(gk,lambdaF/L_F) ;
        
        temp1 = 0.5*norm(y-A*xkp1)^2 ;
        temp2 = 0.5*norm(y-A*yk)^2 + (xkp1-yk)'*temp + (L_F/2)*norm(xkp1-yk)^2 ;
        
       if temp1 <= temp2
            stop_backtrack = 1 ;
%            if iter_sign == 1
%                L_F = 1;
%            end
       else
%                LLL(i)=L_F;
%                i=i+1;
            L_F = L_F*beta_F ;
         
       end
%           iter_sign=iter_sign+1;
    end
    
   prev_f = f;
     f = 0.5*norm(y-A*xkp1)^2 + tau * norm(xk,1);
     ferrr(nIter)=0.5*norm(y-A*xkp1)^2 ;
     keep_going =  (abs(f-prev_f)/(prev_f)> err) ;
%     keep_going = 1 ;
     
   
     yy(nIter)=(abs(f-prev_f)/(prev_f));
      L(nIter)=L_F;
%      plot(nIter,log(abs(f-prev_f)/(prev_f)),'-ok');
   
     
     %disp(keep_going);
    %lambdaF = max(eta*lambdaF,tau) ;
     lambdaF = tau ;
    t_kp1 = 0.5*(1+sqrt(1+4*t_kF*t_kF)) ;
    
    t_km1 = t_kF ;
    t_kF = t_kp1 ;
    xkm1 = xk ;
    xk = xkp1 ;
%     if nIter >100
%         break;
%     end
end
 T_ITCG_vs(nIter) = toc(t0) ;%timeSteps
  xk=abs(xk./sqrt(sum(A1.^2,1))');
 x_ITCG_vs= xk ;%x_hat
   
NZ=sum(x_ITCG_vs~=0);

iter_vs=nIter;
NZ_vs=NZ;
 disp(nIter);
%  disp(T_ITCG_vs);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 function y = soft(x,T)
if sum(abs(T(:)))==0
    y = x;
else
    y = max(abs(x) - T, 0);
    y = sign(x).*y;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%