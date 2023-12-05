clear all
clear
[labels, features] = libsvmread('../Data/german_numer_scaled.txt');


N = length(labels);
d = length(features(1, :));
epochs = 100;
gstop = 1e-10;
lambda = 1 / N; 

X = full(features');
disp(N); disp(d);
Y = ones(N, 1);
for i = 1 : N
    if labels(i) ~= 1
        Y(i) = 0;
    end
end
%% Initialization
% common initializations (to be used accross all the algorithms)
w0 = 0.5 * randn(d,1); % Initialize x0
H0 = zeros(d, d, N); % store all the n hessian approximation matrices in a 3D tensor
for i = 1 : N
    H0(:, :, i) = gethess(X(:, i), w0, lambda);
end
%% Find the minima using IQN
w = w0;
H = H0;
ta = repmat(w,[1 N]);

Hw = squeeze(sum(H.*repmat(w,[1 d N]),1)); 
u = sum(H, 3) * w;

oldg = zeros(d,N);
for i = 1 : N
    oldg(:, i) = getgrad(X(:, i), Y(i), w, lambda); % nabla f_i(x)
end
g = sum(oldg, 2); 

Binv = inv(sum(H, 3));

flag = 0; % used as a flag to check the stopping condition


fprintf("#############Started finding minima!#############\n");
tic;
for ep = 1:epochs
    for i = 1:N
        % recall old data
        oldw = ta(:,i);
        D = H(:,:,i);
        Hwold = Hw(:,i);


        % calculate x and s
        w = Binv * (u - g); 
        s = w - oldw;


        % calculate yy = \nabla f_(i_t)(w)-\nabla f_(i_t)(oldw)
        grad = getgrad(X(:,i),Y(i),w,lambda);
        yy = grad - oldg(:,i);

        % update B
        Ds = D*s;
        sys = yy'*s;
        sDs = s'*Ds;
        Hi = D + (yy*yy')/(yy'*s) - (Ds*Ds')/(s'*Ds); % BFGS update
        Binv = rank2update(Binv,yy,sys,Ds,sDs);

        % update the variables 
        Hiw = Hi*w; 
        u = u + Hiw - Hwold; 
        g = g + grad - oldg(:,i);

        % maintain old gradients and iterates
        oldg(:,i) = grad;
        ta(:,i) = w;
        H(:,:,i) = Hi;
        Hw(:,i) = Hiw;
    end
    grad_norm = norm(getgrad(X,Y,w,lambda)); % gradient of the overall function at current point
    if grad_norm < gstop % check stopping condition
        conv_iters = ep;
        fprintf("Convergence attained after %d epochs with gradient norm: %0.4e\n", ep, grad_norm);
        flag = 1;
        x_star = w; % minima stored in x_star
        break;
    end
    if flag == 1
        break;
    end
    fprintf('(IQN) epoch: %4d, gradient norm: %.1e\n', ep, grad_norm);
end
fprintf("#############Finished finding minima!#############\n");

%% Implement IQN again to store the normalized error
w = w0;
H = H0;
ta = repmat(w,[1 N]); 

Hw = squeeze(sum(H.*repmat(w,[1 d N]),1));  
u = sum(H, 3) * w;

oldg = zeros(d,N); 
for i = 1 : N
    oldg(:, i) = getgrad(X(:, i), Y(i), w, lambda);
end
g = sum(oldg, 2);
Binv = inv(sum(H, 3));

y_axis_iqn = zeros(epochs,1);


fprintf("#############Started IQN!#############\n");
tic;
for ep = 1:conv_iters
    for i = 1:N
        % recall old data
        oldw = ta(:,i);
        D = H(:,:,i);
        Hwold = Hw(:,i); 


        % calculate x and s
        w = Binv*(u-g);
        s = w - oldw; 


        % calculate yy = \nabla f(w)-\nabla f(oldw)
        grad = getgrad(X(:,i),Y(i),w,lambda);
        yy = grad - oldg(:,i); 


        % update B
        Ds = D*s;
        sys = yy'*s;
        sDs = s'*Ds;
        Hi = D + (yy*yy')/(yy'*s) - (Ds*Ds')/(s'*Ds); % BFGS update
        Binv = rank2update(Binv,yy,sys,Ds,sDs);

        % update the variables
        Hiw = Hi*w;
        u = u + Hiw - Hwold;
        g = g + grad - oldg(:,i);

        % maintain old gradients and iterates
        oldg(:,i) = grad;
        ta(:,i) = w;
        H(:,:,i) = Hi;
        Hw(:,i) = Hiw;
    end
    y_axis_iqn(ep) = norm(w - x_star) / norm(w0 - x_star);
end
fprintf("#############Finished IQN!#############\n");

%% Implement IGS
w = w0; 
H = H0;
ta = repmat(w,[1 N]);

Hw = squeeze(sum(H.*repmat(w,[1 d N]),1));  
u = sum(H, 3) * w;

oldg = zeros(d,N);
for i = 1 : N
    oldg(:, i) = getgrad(X(:,i),Y(i),w,lambda);
end
g = sum(oldg, 2);

SUM = sum(H, 3);
Binv = inv(sum(H, 3));

y_axis_igs = zeros(epochs,1);
M = 0; % tune this to change the convergence rate


fprintf("#############Started IGS!#############\n");
tic;
for ep = 1 : conv_iters
    for i = 1 : N
        x = X(:, i);
        % get the variables to be updated
        oldw = ta(:,i); 
        Hi = H(:,:,i);
        Hwold = Hw(:,i);


        % calculate x and s
        w = Binv*(u - g); % x_t
        s = w - oldw;
        old_hess = gethess(X(:, i), oldw, lambda);
        r = sqrt(s'*old_hess*s);

        % calculate yy = \nabla f(w)-\nabla f(oldw)
        grad = getgrad(X(:,i),Y(i),w,lambda);
        yy = grad - oldg(:,i);

        % compute the greedy vector
        a = gethessdiag(x,w,lambda); % a = diagonal elements of the actual Hessian
        DbyA = diag(Hi)./a; % elementwise division of diagonal elements of the approximation and actual Hessian
        [~,gin] = max(DbyA); % gin denotes the index where the max is found
        greedy_vec = zeros(d,1); greedy_vec(gin) = 1;

        % compute the new hessian
        new_hess = gethess(x, w, lambda);
        new_hess_in = new_hess(:, gin);

        % take the BFGS step BFGS(Q, nabla ^ 2 f(x_+), greedy_vec)
        Q = (1 + M * r) * Hi;
        Qu = Q * greedy_vec;
        Hi = Q - (Qu * (Qu)')/Qu(gin) +  (new_hess_in * new_hess_in')/new_hess_in(gin);


        % update the variables to be used in the next iteration
        SUM = SUM + Hi - H(:, :, i);
        Binv = inv(SUM);

        Hiw = Hi*w;
        u = u + Hiw - Hwold;
        g = g + grad - oldg(:,i);

        % maintain old gradients and iterates
        oldg(:,i) = grad;
        ta(:,i) = w;
        H(:,:,i) = Hi;
        Hw(:,i) = Hiw;
    end
    y_axis_igs(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf('(IGS) epoch: %4d, gradient norm: %.1e\n', ep, norm(getgrad(X,Y,w,lambda)));
end
fprintf("#############Finished IGS!#############\n");

%% Implement SLIQN
w = w0;
H = H0;
ta = repmat(w,[1 N]);

Hw = squeeze(sum(H.*repmat(w,[1 d N]),1));
u = sum(H, 3) * w;

oldg = zeros(d,N);
for i = 1 : N
    oldg(:, i) = getgrad(X(:,i),Y(i),w,lambda);
end
g = sum(oldg, 2);

Binv = inv(sum(H, 3));

y_axis_sliqn = zeros(epochs,1);
M = 0; 

fprintf("#############Started SLIQN!#############\n");
tic;
for ep = 1 : conv_iters
    for i = 1 : N
        x = X(:, i);

        oldw = ta(:, i);
        D = H(:, :, i);
        Hwold = Hw(:,i);


        % calculate x and s
        w = Binv * (u - g); 
        s = w - oldw; 
        old_hess = gethess(X(:, i), oldw, lambda);
        r = sqrt(s'*old_hess*s); % might be a bit slow as exact hessian takes O(d^2) time


        % calculate yy = \nabla f(w)-\nabla f(oldw)
        grad = getgrad(X(:,i),Y(i),w,lambda);
        yy = grad - oldg(:, i);


        % update B
        Ds = D * s;
        sys = yy' * s;
        sDs = s'* Ds;
        Hi = D + (yy*yy')/(yy'*s) - (Ds*Ds')/(s'*Ds);
        Q = (1 + 0.5 * M * r) ^ 2 * Hi;

        % compute the greedy vector
        a = gethessdiag(x,w,lambda); % a = diagonal elements of the actual Hessian
        DbyA = diag(Hi)./a; % elementwise division of diagonal elements of the approximation and actual Hessian
        [~,gin] = max(DbyA); % gin denotes the index where the max is found
        greedy_vec = zeros(d,1); greedy_vec(gin) = 1;

        new_hess = gethess(x, w, lambda);
        new_hess_in = new_hess(:, gin);

        % take the second BFGS step BFGS(Q, nabla ^ 2 f(x_+), greedy_vec)
        Qu = Q * greedy_vec;
        Hi = Q - (Qu * (Qu)')/Qu(gin) +  (new_hess_in * new_hess_in')/new_hess_in(gin);

        % update Binv in O(d ^ 2) time
        Binv_y = Binv * yy;
        psi3_inv = Binv - (Binv_y * Binv_y') / (yy' * s + yy' * Binv_y);

        psi3_inv_Bs = psi3_inv * Ds;
        psi2_inv = psi3_inv + (psi3_inv_Bs * psi3_inv_Bs') / (s' * Ds - Ds' * psi3_inv_Bs);

        psi2_inv_Qu = psi2_inv * Qu;
        psi1_inv = psi2_inv + (psi2_inv_Qu * psi2_inv_Qu') / (Qu(gin) - Qu' * psi2_inv_Qu);

        psi1_inv_hessu = psi1_inv * new_hess_in;
        Binv = psi1_inv - (psi1_inv_hessu * psi1_inv_hessu') / (new_hess_in(gin) + new_hess_in' * psi1_inv_hessu);

        Hiw = Hi*w;
        u = u + Hiw - Hwold;
        g = g + grad - oldg(:,i);

        % maintain old gradients and iterates
        oldg(:,i) = grad;
        ta(:,i) = w;
        H(:,:,i) = Hi;
        Hw(:,i) = Hiw;
    end

    y_axis_sliqn(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf('(SLIQN) epoch: %4d, gradient norm: %.1e\n', ep, norm(getgrad(X,Y,w,lambda)));
end
fprintf("#############Finished SLIQN!#############\n");

%% Implement SN
w = w0;
H = H0;
ta = repmat(w,[1 N]);

Hw = squeeze(sum(H.*repmat(w,[1 d N]),1));
u = sum(H, 3) * w;

oldg = zeros(d,N);
for i = 1 : N
    oldg(:, i) = getgrad(X(:,i),Y(i),w,lambda);
end
g = sum(oldg, 2);

SUM = sum(H, 3);
Binv = inv(SUM);

tau = N; % tau represents the batch size 
y_axis_sn = zeros(epochs,1);


fprintf("#############Started SN!#############\n");
tic;
for ep = 1 : conv_iters
     w = Binv * (u - g);

     % chose a subset of size tau uniformly at random
     batch = randperm(N, tau);

     for j = 1 : length(batch)
         idx = batch(j);
         Hidx = gethess(X(:, idx), w, lambda);
         grad =  getgrad(X(:, idx), Y(idx), w, lambda); 
         Hidxw = Hidx * w;

         SUM = SUM - H(:, :, idx) + Hidx;
         g = g - oldg(:, idx) + grad;
         u = u - Hw(:, idx) + Hidxw;

         oldg(:, idx) = grad;
         ta(:, idx) = w;
         H(:, :, idx) = Hidx;
         Hw(:, idx) = Hidxw;
     end
    Binv = inv(SUM);
    y_axis_sn(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf('(SN) epoch: %4d, gradient norm: %.1e\n', ep, norm(getgrad(X, Y, w, lambda)));
end
fprintf("#############Finished SN!#############\n");

%% Implement NIM
w = w0;
H = H0;
ta = repmat(w,[1 N]);

Hw = squeeze(sum(H.*repmat(w,[1 d N]),1));
u = sum(H, 3) * w;

oldg = zeros(d,N);
for i = 1 : N
    oldg(:, i) = getgrad(X(:,i),Y(i),w,lambda);
end
g = sum(oldg, 2);

SUM = sum(H, 3);
Binv = inv(sum(H, 3));

y_axis_nim = zeros(epochs,1);

fprintf("#############Started NIM!#############\n");
tic;
for ep = 1 : conv_iters
    for i = 1 : N
        x = X(:, i);
        % get the variables to be updated
        oldw = ta(:,i); % pick up the i-th coordinate, z_i 
        Hwold = Hw(:,i); % Bz


        % calculate x and s
        w = Binv * (u - g); % x_t

        Hi = gethess(x, w, lambda);
        grad = getgrad(x, Y(i), w, lambda);

        % update the variables to be used in the next iteration
        SUM = SUM + Hi - H(:, :, i);
        Binv = inv(SUM); 

        Hiw = Hi*w;
        u = u + Hiw - Hwold;
        g = g + grad - oldg(:,i);

        % maintain old gradients and iterates
        oldg(:,i) = grad;
        ta(:,i) = w;
        H(:,:,i) = Hi;
        Hw(:,i) = Hiw;
    end

    y_axis_nim(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf('(NIM) epoch: %4d, gradient norm: %.1e\n', ep, norm(getgrad(X, Y, w, lambda)));
end
fprintf("#############Finished NIM!#############\n");

%% Plots
x_axis = 1:conv_iters;
semilogy(x_axis, y_axis_iqn(1:conv_iters), '-or', 'Linewidth', 2);

hold on
semilogy(x_axis, y_axis_igs(1:conv_iters), '-*b', 'LineWidth', 2);
semilogy(x_axis, y_axis_sliqn(1:conv_iters), '-xg', 'LineWidth', 2);
semilogy(1:conv_iters, y_axis_sn(1:conv_iters), '-squarem', 'LineWidth', 2);
semilogy(x_axis, y_axis_nim(1:conv_iters), '-diamondk', 'LineWidth', 2);
l = legend({'IQN', 'IGS', 'SLIQN', 'SN', 'NIM'});
set(l, 'Interpreter', 'latex', 'fontsize', 20, 'Location', 'southwest')
% title('Plot of Normalized error vs No of effective passes for svmguide3 dataset', 'Interpreter', 'latex')
ylabel('Normalized error $\frac{||\textbf{\emph x} ^ {t} - \textbf{\emph x} ^ \star||}{||\textbf{\emph x} ^ 0 - \textbf{\emph x} ^ \star||}$', 'Interpreter','latex', 'fontsize', 20, 'Rotation', 90)
xlabel('{No of effective passes}', 'Interpreter', 'latex')
xlim([1, 4])
ax = gca;
ax.FontSize = 20;
set(gcf,'position',[0,0,600,400])
hold off
savefig('../Matlab_plots/german_numer_p_21.fig')


%% Useful functions
function grad = getgrad(x,y,w,lambda)
    % return the gradient of logistic loss + regularizer at w
    % data is x labels are y (scalar)
    p = 2.1;
    swx  = 1./(1+exp(-x'*w));
    grad = -x*(y-swx)/length(y) + p * lambda * w * (norm(w) ^ (p - 2));
end

function hessd = gethessdiag(x,w,lambda)
    % return the diagonal elements of the Hessian at w
    % returned as a vector
    p = 2.1;
    swx = 1./(1+exp(-x'*w));
    sw = x*swx*(1-swx);
    NORM = norm(w);
    hessd = p * lambda * (NORM ^ (p - 4)) * (NORM ^ 2 + (p - 2) * (w .^ 2)) + x .* sw;
end

function Binv = rank2update(Binv,x,x0,y,y0)
    % performs rank 2 update B + xx'/x0 - yy'/y0
    % first apply + xx'/x0
    Bx = Binv*x;
    Binv = Binv - Bx*Bx'/(x0+Bx'*x);
    % now apply - yy'/y0
    By = Binv*y;
    Binv = Binv + By*By'/(y0-By'*y);    
end

function hess = gethess(x,w,lambda)
    % return the logistic loss Hessian at w
    % data is x
    p = 2.1;
    swx = 1./(1+exp(-x'*w));
    NORM = norm(w);
    hess = p * lambda * NORM ^ (p - 2) * eye(length(w)) + lambda * p * (p - 2) * NORM ^ (p - 4) * (w * w')  + x * x'* swx * (1 - swx);
end