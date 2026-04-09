function [x_ITCG_vs,nv_ITCG_vs,T_ITCG_vs,iter_vs,NZ_vs]=norm_ITCG_vs(y,A1,tau,err)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err_sub=1e-40;
[M,N]=size(A1);
A = A1./repmat(sqrt(sum(A1.^2,1)),M,1);
b=A1'*y;

if tau>=max(abs(b))
    x_ITCG_vs=zeros(N,1);
    nv_ITCG_vs=0;
    T_ITCG_vs=0;
    iter_vs=0;
    NZ_vs=0;
    return
end
A0=A;
A=[A,-A];
c=tau+[-b;b];

Ns=fix(M/4);
N_max=Ns+fix(Ns/8);


gamma=0.9;
beta=0.01;
delta=7;
alpha_max=10^10;
ind=0;
z=zeros(2*N,1);
%%%%%%%%%%%%%%%%%%%%%%%% main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%
t1=cputime;

g=c;
v=min(0,g);
nv_ITCG=norm(v);

for iter=0:1000    

    if nv_ITCG<err
        break
    end    
      
    Gamma=((z>0)&(g~=0))|((z==0)&(g>0));    

    abs_g=abs(g);
    z_abs_g=z./(abs_g+eps);
    [Y,I_hat]=sort(((z>0)&(z_abs_g>delta)).*abs_g,'descend');   
    n_Y=sum(Y~=0);
    if n_Y<=Ns
        I=I_hat(1:n_Y);
    else
        I=I_hat(1:Ns);
    end     
    
    if n_Y==0
        ind=1;
    else
        A_I=A(:,I);
        b_I=g(I);
        z_I=z(I);    
        %%%%%%%%%%%% to solve the subproblem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        [MI,NI]=size(A_I);
        s=zeros(NI,1);
        gI=b_I;
        d_bar=-gI;        
                       
        q=gI'*gI;
        
        for iter_sub=1:NI             
            q1=q;                     
            if q1<err_sub           
                break;
            end
            t=A_I*d_bar;
            norm_t2=t'*t;
            if q1>=alpha_max*norm_t2 
                alpha_I=alpha_max;
                disp('???alpha_k>=alpha_max or (bar(d)^i)^T*BSl*bar(d)^i=0');
            else
                alpha_I=q1/norm_t2;
            end
            alpha_I_d_bar=alpha_I*d_bar;
            z_bar=z_I+alpha_I_d_bar;     
            if sum(z_bar<0)>0
                z_I_d_bar=-z_I./d_bar;
                Y_sub=sort(z_I_d_bar.*(z_I_d_bar>0),'descend'); 
                n_Y_sub=sum(Y_sub~=0);
                alpha_star=Y_sub(n_Y_sub);
                s=s+alpha_star*d_bar;                
                break;
            else
                z_I=z_bar;
                s=s+alpha_I_d_bar;
                gI=gI+alpha_I*(A_I'*t);
                q=gI'*gI;
                beta_bar=q/q1;
                d_bar=-gI+beta_bar*d_bar;
            end
        end
        d_I=s;
    end
    %%%%%%%%%%% the subproblem has solved %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
             
    I_J=ones(N+N,1);
    I_J(I)=zeros(size(I));
%     [Y,J_hat]=sort(((abs(v).*Gamma).*I_J),'descend');
    [Y,J_hat]=sort(((abs(g).*Gamma).*I_J),'descend');
    
    n_Y=sum(Y~=0);
    if n_Y<=N_max-Ns
        J=J_hat(1:n_Y);
    else
        J=J_hat(1:N_max-Ns);
    end    

    d=zeros(N+N,1);
    if ind==1
        d(J)=-v(J);
        ind=0;
    else
        d(I)=d_I;
        d(J)=-v(J);
    end      

    d_sparse=sparse(d);
    g_d=g'*d_sparse;
    if g_d>=0
        disp('error!d is not a descent direction')
        d_sparse=-v;
        A_d_Gamma=A* d_sparse;
        d_B_d=A_d_Gamma'*A_d_Gamma;        
        for l=0:100                                               
            if d_B_d<=2*(beta-1)*g_d
%                 if l==0
%                     disp('the iteration point has become sparser')
%                 end
                arpha=gamma^l;
                break
            end
            d_B_d=gamma*d_B_d;
        end
    else
        A_d_Gamma=A* d_sparse;
        d_B_d=A_d_Gamma'*A_d_Gamma;        
        for l=0:100                                               
            if d_B_d<=2*(beta-1)*g_d
%                 if l==0
%                     disp('the iteration point has become sparser')
%                 end
                arpha=gamma^l;
                break
            end
            d_B_d=gamma*d_B_d;
        end  
    end
  
    z0=z;
    d_sparse=arpha*d_sparse;
    z=z+d_sparse;
    z=z.*(z>0);
    z_min=min(z(1:N),z(N+1:N+N));
    z(1:N)=z(1:N)-z_min;
    z(N+1:N+N)=z(N+1:N+N)-z_min;
    d1=sparse(z(1:N)-z0(1:N));
    d2=sparse(z(N+1:N+N)-z0(N+1:N+N));
    A0_d1_d2=A0*(d1-d2);   
    A0T_A0_d1_d2=A0'*A0_d1_d2;
    g(1:N)=g(1:N)+A0T_A0_d1_d2;
    g(N+1:N+N)=g(N+1:N+N)-A0T_A0_d1_d2;
    v=min(z,g);
    nv_ITCG=norm(v);   
end
T_ITCG=cputime-t1;
x_ITCG=z(1:N)-z(N+1:N+N);   
NZ=sum(x_ITCG~=0);



x_ITCG_vs=x_ITCG;
nv_ITCG_vs=nv_ITCG;
T_ITCG_vs=T_ITCG;
iter_vs=iter;
NZ_vs=NZ;
 x_ITCG_vs=abs(x_ITCG_vs./sqrt(sum(A1.^2,1))');