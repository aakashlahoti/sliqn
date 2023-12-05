clear all
clear
[labels, features] = libsvmread('../Data/protein.txt');


n = length(labels);
d = length(features(1, :));
disp(n); disp(d);

X = full(features);
Y = zeros(n, 1);

for i = 1 : n
    if labels(i) == 2
        Y(i) = 1;
    end
end



N = 5000;
idx = randperm(n, N);
lambda = 1 / N;
X = X(idx,:)';
Y = Y(idx);     % y are 0-1 encoded

N_subset = 100;
epochs = 100;
gstop = 1e-8;

%% Initialization
w0 = 0.5 * randn(d, 1); % Initialize x0


%% Implement IQN (Memory efficient version)
w = w0;
ta = repmat(w, [1 N]); % tracks all z values in a (d, n) array

Hw = zeros(d, N);   % maintains the product Hw in a (d, n) array


oldg = zeros(d, N); % tracks the gradient for each of the n functions
for i = 1 : N
    oldg(:, i) = getgrad(X(:, i), Y(i), w, lambda); % nabla f_i(x)
end


H_window = zeros(d, d, N_subset);
Hess_sum = zeros(d, d);
Hess_i = zeros(d, d);


for i = 1 : N
    Hess_i = gethess(X(:, i), w, lambda);
    Hess_sum = Hess_i + Hess_sum;
    Hw(:, i) = Hess_i * w;
end

u = sum(Hw, 2);
g = sum(oldg, 2);
Binv = inv(Hess_sum); % running variable that keeps a track of (sum_i H_i) ^ {-1}

flag = 0; % used as a flag to check the stopping condition gnormiqn < gstop
fprintf("####################Started M-IQN#################\n")

store_w = zeros(d, epochs);
y_axis_iqn = zeros(epochs, 1);

for ep = 1 : epochs
    for rep = 1 : N / (N_subset)
        % For the first epoch, we do not need to load anything
        if ep > 1
            prev_file_name = "../data_store/protein/epochs/" + "Epoch_" + string(ep - 1) + "_rep_" + string(rep);
            load (prev_file_name, "H_window");
        end
        for j = 1 : N_subset        
            % recall old data
            i = (rep - 1) * N_subset + j;

            % For the first epoch H_window has not been initialized, we do
            % it manually by setting it to the exact value (no loading)
            if ep == 1
                % clear H_window
                Hess_i = gethess(X(:, i), w0, lambda);
                % fprintf("Current at ep: %d rep: %d index: %d norm_Hess_i: %0.4e\n", ep, rep, i, norm(Hess_i, 'fro'));
                H_window(:, :, j) = Hess_i;
            end
            oldw = ta(:,i);
            D = H_window(:, :, j);
            Hwold = Hw(:, i); 


            % calculate x and s
            w = Binv * (u - g);
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
            H_window(:, :, j) = Hi;
            Hw(:,i) = Hiw;
        end
        file_name = "../data_store/protein/epochs/" + "Epoch_" + string(ep) + "_rep_" + string(rep);
        save(file_name, 'H_window');
        % clear H_window;
    end
    grad_norm = norm(getgrad(X, Y, w, lambda)); % gradient of the overall function at current point
    fprintf("(M-IQN) ep: %d, grad_norm: %0.4e\n", ep, grad_norm);
    store_w(:, ep) = w;
    if grad_norm < gstop % check stopping condition
        conv_iters = ep;
        fprintf("Convergence attained after %d epochs with grad_norm: %0.4e\n", ep, grad_norm);
        flag = 1;
        x_star = w; % minima stored in x_star
        break;
    end
    if flag == 1
        break;
    end
end

fprintf("####################Finished M-IQN!####################\n");

for ep = 1 : conv_iters
    w_ep = store_w(:, ep);
    y_axis_iqn(ep) = norm(w_ep - x_star) / norm(w0 - x_star);
end
%% Implement Greedy IGS
w = w0;
ta = repmat(w, [1 N]); % tracks all z values in a (d, n) array

Hw = zeros(d, N);   % maintains the product Hw in a (d, n) array


oldg = zeros(d, N); % tracks the gradient for each of the n functions
for i = 1 : N
    oldg(:, i) = getgrad(X(:, i), Y(i), w, lambda); % nabla f_i(x)
end


H_window = zeros(d, d, N_subset);
Hess_sum = zeros(d, d);
Hess_i = zeros(d, d);
for i = 1 : N
    Hess_i = gethess(X(:, i), w, lambda);
    Hess_sum = Hess_i + Hess_sum;
    Hw(:, i) = Hess_i * w;
end

u = sum(Hw, 2);
g = sum(oldg, 2);
Binv = inv(Hess_sum); % running variable that keeps a track of (sum_i H_i) ^ {-1}


y_axis_greedy = zeros(conv_iters, 1);
M_greedy = 0;

fprintf("####################Started IGS!#################\n");
for ep = 1 : conv_iters
    for rep = 1 : N / (N_subset)
        % For the first epoch, we do not need to load anything
        if ep > 1
            prev_file_name = "../data_store/protein/epochs/" + "Epoch_" + string(ep - 1) + "_rep_" + string(rep);
            load (prev_file_name, "H_window");
        end
        for j = 1 : N_subset        
            % recall old data
            i = (rep - 1) * N_subset + j;

            % For the first epoch H_window has not been initialized, we do
            % it manually by setting it to the exact value (no loading)
            if ep == 1
                % clear H_window
                Hess_i = gethess(X(:, i), w0, lambda);
                % fprintf("Current at ep: %d rep: %d index: %d norm_Hess_i: %0.4e\n", ep, rep, i, norm(Hess_i, 'fro'));
                H_window(:, :, j) = Hess_i;
            end
            x = X(:, i);
            % get the variables to be updated
            oldw = ta(:, i);
            Hi = H_window(:, :, j); 
            Hwold = Hw(:, i);
    
    
            % calculate x and s
            w = Binv * (u - g); % x_t (in the draft)
            s = w - oldw; % variable variation
            old_hess = gethess(X(:, i), oldw, lambda);
            r = sqrt(s' * old_hess * s); % might be a bit slow as exact hessian takes O(d^2) time
    
            % calculate yy = \nabla f(w)-\nabla f(oldw)
            grad = getgrad(X(:, i), Y(i), w, lambda); % gradient of f_i at z_(i_t)(t)
            yy = grad - oldg(:, i); % gradient variation
    
            % compute the greedy vector
            a = gethessdiag(x, w, lambda); % a = diagonal elements of the actual Hessian
            DbyA = diag(Hi)./a; % elementwise division of diagonal elements of the approximation and actual Hessian
            [~,gin] = max(DbyA); % gin denotes the index where the max is found
            greedy_vec = zeros(d, 1); greedy_vec(gin) = 1;
    
            new_hess = gethess(x, w, lambda);
            new_hess_in = new_hess(:, gin);
    
            % take the BFGS step BFGS(Q, nabla ^ 2 f(x_+), greedy_vec)
            Q = (1 + M_greedy * r) * Hi;
            Qu = Q * greedy_vec;
            Hi = Q - (Qu * (Qu)')/Qu(gin) +  (new_hess_in * new_hess_in')/new_hess_in(gin);
    
            psi2_inv_Qu = Binv * Qu;
            psi1_inv = Binv + (psi2_inv_Qu * psi2_inv_Qu') / (Qu(gin) - Qu' * psi2_inv_Qu);
    
            psi1_inv_hessu = psi1_inv * new_hess_in;
            Binv = psi1_inv - (psi1_inv_hessu * psi1_inv_hessu') / (new_hess_in(gin) + new_hess_in' * psi1_inv_hessu);
    
            Hiw = Hi * w;
            u = u + Hiw - Hwold;
            g = g + grad - oldg(:, i);
    
            % maintain old gradients and iterates
            oldg(:, i) = grad;
            ta(:, i) = w;
            H_window(:, :, j) = Hi;
            Hw(:, i) = Hiw;
        end
        file_name = "../data_store/protein/epochs/" + "Epoch_" + string(ep) + "_rep_" + string(rep);
        save(file_name, 'H_window');
        % clear H_window;
    end
    grad_norm = norm(getgrad(X, Y, w, lambda)); % gradient of the overall function at current point
    y_axis_greedy(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf("(IGS) ep: %d, grad_norm: %0.4e\n", ep, grad_norm);
end

fprintf("####################Finished IGS!####################\n");
%% Implement SL-IQN (no correction needed)
w = w0;
ta = repmat(w, [1 N]); % tracks all z values in a (d, n) array

Hw = zeros(d, N);   % maintains the product Hw in a (d, n) array


oldg = zeros(d, N); % tracks the gradient for each of the n functions
for i = 1 : N
    oldg(:, i) = getgrad(X(:, i), Y(i), w, lambda); % nabla f_i(x)
end


H_window = zeros(d, d, N_subset);
Hess_sum = zeros(d, d);
Hess_i = zeros(d, d);
for i = 1 : N
    Hess_i = gethess(X(:, i), w, lambda);
    Hess_sum = Hess_i + Hess_sum;
    Hw(:, i) = Hess_i * w;
end

u = sum(Hw, 2);
g = sum(oldg, 2);
Binv = inv(Hess_sum); % running variable that keeps a track of (sum_i H_i) ^ {-1}

fprintf("####################Started SLIQN!#################\n");

y_axis_sliqn = zeros(conv_iters, 1);
M_sharp = 0;
for ep = 1 : conv_iters
    for rep = 1 : N / (N_subset)
        % For the first epoch, we do not need to load anything
        if ep > 1
            prev_file_name = "../data_store/protein/epochs/" + "Epoch_" + string(ep - 1) + "_rep_" + string(rep);
            load (prev_file_name, "H_window");
        end
        for j = 1 : N_subset        
            % recall old data
            i = (rep - 1) * N_subset + j;

            % For the first epoch H_window has not been initialized, we do
            % it manually by setting it to the exact value (no loading)
            if ep == 1
                % clear H_window
                Hess_i = gethess(X(:, i), w0, lambda);
                % fprintf("Current at ep: %d rep: %d index: %d norm_Hess_i: %0.4e\n", ep, rep, i, norm(Hess_i, 'fro'));
                H_window(:, :, j) = Hess_i;
            end
                x = X(:, i);

                oldw = ta(:, i);
                D = H_window(:, :, j);
                Hwold = Hw(:, i);


                % calculate x and s
                w = Binv * (u - g); 
                s = w - oldw; 
                old_hess = gethess(X(:, i), oldw, lambda);
                r = sqrt(s' * old_hess * s); % might be a bit slow as exact hessian takes O(d^2) time


                % calculate yy = \nabla f(w)-\nabla f(oldw)
                grad = getgrad(X(: ,i), Y(i), w, lambda);
                yy = grad - oldg(:, i);


                % update B
                Ds = D * s;
                sys = yy' * s;
                sDs = s' * Ds;
                Hi = D + (yy * yy')/(yy' *s) - (Ds * Ds')/(s' *Ds);
                Q = (1 + 0.5 * M_sharp * r) ^ 2 * Hi;

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

                % refer to paper for the exact Sherman-Morrison updates
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
                g = g + grad - oldg(:, i);

                % maintain old gradients and iterates
                oldg(:, i) = grad;
                ta(:, i) = w;
                H_window(:, :, j) = Hi;
                Hw(:, i) = Hiw;
        end
        file_name = "../data_store/protein/epochs/" + "Epoch_" + string(ep) + "_rep_" + string(rep);
        save(file_name, 'H_window');
        % clear H_window;
    end
    grad_norm = norm(getgrad(X, Y, w, lambda)); % gradient of the overall function at current point
    y_axis_sliqn(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf("(SLIQN) ep: %d, grad_norm: %0.4e\n", ep, grad_norm);
end

fprintf("####################Finished SLIQN!####################\n");

%% Implement NIM
w = w0;

ta = repmat(w, [1 N]);

g = zeros(d, 1); % tracks the gradient for each of the n functions
u = zeros(d, 1);
Hess_sum = zeros(d, d);

for i = 1 : N
    Hess_i = gethess(X(:, i), w, lambda);
    Hess_sum = Hess_i + Hess_sum;
    u = u + Hess_i * w;
    g = g + getgrad(X(:, i), Y(i), w, lambda); % nabla f_i(x)
end

Binv = inv(Hess_sum);
y_axis_NIM_eff_space = zeros(epochs, 1);


fprintf("####################Started NIM!#################\n");
for ep = 1 : conv_iters
    for i = 1 : N
        x = X(:, i);
        % get the variables to be updated
        oldw = ta(:, i); % pick up the i-th coordinate, z_i 
        Hold = gethess(x, oldw, lambda);
        oldg = getgrad(x, Y(i), oldw, lambda);
        Hwold = Hold * oldw;

        % calculate x and s
        w = Binv * (u - g); % x_t

        Hi = gethess(x, w, lambda);
        grad = getgrad(x, Y(i), w, lambda);

        % update the variables to be used in the next iteration
        Hess_sum = Hess_sum + Hi - Hold;
        Binv = inv(Hess_sum); % Takes O(d^3) time


        Hiw = Hi * w;
        u = u + Hiw - Hwold;
        g = g + grad - oldg;
        ta(:, i) = w;
    end

    y_axis_NIM_eff_space(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf('(NIM Space_eff) epoch: %4d, g norm: %.1e\n', ep, norm(g));
end
fprintf("####################Finished NIM!####################\n");

save("../data_store/protein/all_vars/protein_all_vars_before_SN")

%% Implement SN

w = w0;
g = zeros(d, 1); % tracks the gradient for each of the n functions
u = zeros(d, 1);
Hess_sum = zeros(d, d);

for i = 1 : N
    Hess_i = gethess(X(:, i), w, lambda);
    Hess_sum = Hess_i + Hess_sum;
    u = u + Hess_i * w;
    g = g + getgrad(X(:, i), Y(i), w, lambda); % nabla f_i(x)
end

Binv = inv(Hess_sum);
y_axis_stochastic_newton_eff_space = zeros(epochs, 1);
Hold = zeros(d, d);

fprintf("####################Started SN!####################\n");
for ep = 1 : conv_iters
        oldw = w;
        w = Binv * (u - g);
        for i = 1 : N
             x = X(:, i);
             
             Hold = gethess(x, oldw, lambda);
             Hi = gethess(x, w, lambda);
             Hess_sum = Hess_sum - Hold + Hi;
            
             
             oldg = getgrad(x, Y(i), oldw, lambda);
             grad = getgrad(x, Y(i), w, lambda);
             g = g - oldg + grad;
             
             
             Hwold = Hold * oldw;
             Hiw = Hi * w;
             u = u - Hwold + Hiw;
        end
    Binv = inv(Hess_sum);
    y_axis_stochastic_newton_eff_space(ep) = norm(w - x_star) / norm(w0 - x_star);
    fprintf('(SN Memory Eff) epoch: %4d, g norm: %.1e\n', ep, norm(g));
end

fprintf("####################Finished SN!####################\n");

save("../data_store/protein/all_vars/protein_all_vars_till_SN")
%% Plots
x_axis = 1:conv_iters;
semilogy(x_axis, y_axis_iqn(1:conv_iters), '-or', 'Linewidth', 2);

hold on
semilogy(x_axis, y_axis_greedy(1:conv_iters), '-*b', 'LineWidth', 2);
semilogy(x_axis, y_axis_sliqn(1:conv_iters), '-xg', 'LineWidth', 2);
semilogy(1:conv_iters, y_axis_stochastic_newton_eff_space(1:conv_iters), '-squarem', 'LineWidth', 2);
semilogy(x_axis, y_axis_NIM_eff_space(1:conv_iters), '-diamondk', 'LineWidth', 2);
l = legend({'IQN', 'IGS', 'SLIQN', 'SN', 'NIM'});
set(l, 'Interpreter', 'latex', 'fontsize', 20, 'Location', 'southwest')
% title('Plot of Normalized error vs No of effective passes for connect-4 dataset', 'Interpreter', 'latex')
ylabel('Normalized error $\frac{||\textbf{\emph x} ^ {t} - \textbf{\emph x} ^ \star||}{||\textbf{\emph x} ^ 0 - \textbf{\emph x} ^ \star||}$', 'Interpreter','latex', 'fontsize', 20, 'Rotation', 90)
xlabel('{No of effective passes}', 'Interpreter', 'latex')
ax = gca;
ax.FontSize = 20;
set(gcf,'position',[0,0,600,400])
hold off

save_file_name = "../Matlab_plots/save_load_protein_p_21_" + string(N) + "_N_subset_" + string(N_subset);
savefig(save_file_name)

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
