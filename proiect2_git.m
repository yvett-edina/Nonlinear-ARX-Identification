clc

figure;
subplot(2,2,[1,3]);
plot(id);
subplot(2,2,[2,4]);
plot(val);

%%
clc

max_degree = 4;
max_na = 3;
max_nb = max_na;
nk = 1;

u_id = id.InputData;
y_id = id.OutputData;
N_id = length(y_id);

u_val = val.InputData;
y_val = val.OutputData;
N_val = length(y_val);

MSE_id_all = zeros(4, 3);
MSE_val_all = zeros(4, 3);
MSE_sim_all = zeros(4, 3);

for m = 1 : max_degree
    for na = 1 : max_na

        nb = na;
        n = na + nb;

        %==========MATRIX OF EXPONENTIALS==========

        allExpMatrix = [];

        exponents = zeros(1, n);
        i = 1;

        while true
            if(sum(exponents) <= m)
                allExpMatrix(i, :) = exponents;
                i = i + 1;
            end

            exponents(1) = exponents(1) + 1;
            for z = 1 : n
                if(exponents(z) > m) % carry over if current pos > max degree
                    exponents(z) = 0; % reset current pos

                    if(z < n)
                        exponents(z + 1) = exponents(z + 1) + 1; % next pos++
                    else
                        break; % stop if last pos is exceeded
                    end
                else
                    break; % stop carry over if no further increments needed
                end
            end

            % check if entire loop should terminate
            if(all(exponents == 0))
                break;
            end
        end

        %=================Y_PRED_ID================

        phi_id = [];
        order1Elem_id = zeros(N_id, n);

        for k = 1 : N_id
            row = [];

            for i = 1 : na
                if((k - i) <= 0)
                    row = [row, 0];
                else
                    row = [row, -y_id(k - i)];
                end
            end

            for j = 1 : nb
                if((k - nk - j) <= 0)
                    row = [row, 0];
                else
                    row = [row, u_id(k - nk - j)];
                end
            end

            order1Elem_id(k, :) = row;
        end

        for e = 1 : length(allExpMatrix)
            term = ones(N_id, 1);
            for w = 1 : n
                term = term .* (order1Elem_id(:, w) .^ allExpMatrix(e, w));
            end

            phi_id = [phi_id, term];

        end

        theta = phi_id \ y_id;

        %=================Y_PRED_VAL===============

        phi_val = [];
        order1Elem_val = zeros(N_val, n);

        for k = 1 : N_val
            row = [];

            for i = 1 : na
                if((k - i) <= 0)
                    row = [row, 0];
                else
                    row = [row, -y_val(k - i)];
                end
            end

            for j = 1 : nb
                if((k - nk - j) <= 0)
                    row = [row, 0];
                else
                    row = [row, u_val(k - nk - j)];
                end
            end

            order1Elem_val(k, :) = row;
        end

        for e = 1 : length(allExpMatrix)
            term = ones(N_val, 1);
            for w = 1 : n
                term = term .* (order1Elem_val(:, w) .^ allExpMatrix(e, w));
            end
            
            phi_val = [phi_val, term];
        end

        y_pred_id = phi_id * theta;
        y_pred_val = phi_val * theta;

        if((m == 4) && (na == 3))
            figure;
            plot(y_id, Linewidth=1);
            hold on;
            plot(y_pred_id, '--', Linewidth=1);
            title('Y-PRED_I_D - optimum after id');
            legend('y_i_d', 'y-pred_i_d')
        end

        if((m == 1) && (na == 3))
            figure;
            plot(y_id, Linewidth=1);
            hold on;
            plot(y_pred_id, '--', Linewidth=1);
            title('Y-PRED_I_D - optimum after val');
            legend('y_i_d', 'y-pred_i_d')
        end

        if((m == 1) && (na == 3))
            figure;
            plot(y_val, Linewidth=1);
            hold on;
            plot(y_pred_val, '--', Linewidth=1);
            title('Y-PRED_V_A_L');
            legend('y_v_a_l', 'y-pred_v_a_l')
        end

        %===================Y_SIM==================

        y_sim = zeros(N_val, 1);
        order1Elem_sim = zeros(N_val, n);

        for k = 2 : N_val
            row = [];

            for i = 1 : na
                if((k - i) <= 0)
                    row = [row, 0];
                else
                    row = [row, -y_sim(k - i)];
                end
            end

            for j = 1 : nb
                if((k - nk - j) <= 0)
                    row = [row, 0];
                else
                    row = [row, u_val(k - nk - j)];
                end
            end

            order1Elem_sim(k, :) = row;

            for e = 1 : length(allExpMatrix)
                term = 1;

                for i = 1 : n
                    term = term * (order1Elem_sim(k, i) ^ allExpMatrix(e, i));
                end

                y_sim(k) = y_sim(k) + theta(e) * term;
            end
        end

        if((m == 2) && (na == 2))
            figure;
            plot(y_val, Linewidth=1);
            hold on;
            plot(y_sim, '--', Linewidth=1);
            title('Y_S_I_M');
            legend('y_v_a_l', 'y_s_i_m')
        end

        MSE_id = sum(1/length(y_id) * (y_id - y_pred_id).^2);
        MSE_val = sum(1/length(y_val) * (y_val - y_pred_val).^2);
        MSE_sim = sum(1/length(y_val) * (y_val - y_sim).^2);

        MSE_id_all(m,na) = MSE_id;
        MSE_val_all(m,na) = MSE_val;
        MSE_sim_all(m,na) = MSE_sim;
    end
end

minID = min(MSE_id_all(:))
minVAL = min(MSE_val_all(:))
minSIM = min(MSE_sim_all(:))