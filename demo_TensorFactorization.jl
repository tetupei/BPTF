using LinearAlgebra
using Distributions
using ProgressMeter
using PDMats

#R: NxMxK
#U: DxN
#V: DxM
#T: DxK

#alpha: size(W_tilda)[1]xsize(W_tilda)[2]xL

# check the type and dimenstion of input, output for each function and write test case

function demo()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0  # It should be NaN, but use 0 here becausde 0 * NaN = NaN.

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0::Matrix{Float64} = hcat([0.04])

    U,V,T,alpha = gibbs_sampling(R_obs, mask, D, mu_0, beta_0, W_0, nu_0, rho_0, nu_tilde_0, W_tilde_0, 100)
    R_inferred =  inferenced_R(U,V,T,alpha)
    print("R: ", R_obs, "\n")
    print("R_obs: ", R_obs, "\n")
    print("R_inferred: ", R_inferred, "\n")
end

function gibbs_sampling(R, mask, D, mu_0, beta_0, W_0, nu_0, rho_0, nu_tilde_0, W_tilde_0, iter)

    N,M,K = size(R)

    # initilize
    U1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T1 = init_T(W_0, nu_0, rho_0, beta_0, K)

    # Container
    U = zeros(D, N, iter+1)
    V = zeros(D, M, iter+1)
    T = zeros(D, K, iter+1)
    alpha = zeros(size(W_tilde_0)[1],size(W_tilde_0)[2], iter)

    U[:,:,1] = U1
    V[:,:,1] = V1
    T[:,:,1] = T1

    @showprogress for l in 1 : iter
        alpha[:,:,l] = sample_alpha(nu_tilde_0, W_tilde_0, mask, R, U[:,:,l], V[:,:,l], T[:,:,l])
        mu_U, Lambda_U = sample_theta(U[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_V, Lambda_V = sample_theta(V[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_T, Lambda_T = sample_theta_T(T[:,:,l], beta_0, rho_0, W_0, nu_0)

        # Update U
        Q = zeros(M, K, D)
        for k in 1 : K
            for j in 1 : M
                Q[j,k,:] = V[:,j,l] .* T[:,k,l]
            end
        end
        for i in 1 : N
            U[:,i,l+1] = sample_Ui(i, mu_U, Lambda_U, alpha[:,:,l], R, Q, mask)
        end

        # Update V
        P = zeros(N, K, D)
        for k in 1 : K
            for i in 1 : N
                P[i,k,:] = U[:,i,l] .* T[:,k,l]
            end
        end
        for j in 1 : M
            V[:,j,l+1] = sample_Ui(j, mu_V, Lambda_V, alpha[:,:,l], R, P, mask)
        end

        # Update T
        X = zeros(N, M, D)
        for i in 1 : N
            for j in 1 : M
                X[i,j,:] = U[:,i,l] .* V[:,j,l]
            end
        end
        T[:,1,l+1] = sample_T1(mu_T, Lambda_T, alpha[:,:,l], X, mask, T[:,2,l])
        for k in 2 : K-1
            T[:,k,l+1] = sample_Tk_mt2(k, mu_T, Lambda_T, alpha[:,:,l], X, mask, R, T[:,k-1,l], T[:,k+1,l])
        end
        T[:,K,l+1] = sample_TK(K, mu_T, Lambda_T, alpha[:,:,l], X, mask, R, T[:,K-1,l])

    end

    return U,V,T,alpha
end

function test_gibbs_sampling()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0  # It should be NaN, but use 0 here becausde 0 * NaN = NaN.

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0::Matrix{Float64} = hcat([0.04])

    U,V,T,alpha = gibbs_sampling(R_obs, mask, D, mu_0, beta_0, W_0, nu_0, rho_0, nu_tilde_0, W_tilde_0, 100)
    print(U,V,T,alpha)
end

function inferenced_R(U, V, T, alpha)
    N, M, K = size(U)[2], size(V)[2], size(T)[2]
    _,_,L = size(alpha)
    R_pred = zeros(N,M,K)
    for k in 1 : K
        for i in 1 : N
            for j in 1 : M
                acc = 0.
                for l in 1 : L
                    acc += rand(Normal(multi_dot(U[:,i,l],V[:,j,l],T[:,k,l]), alpha[:,:,l][1][1]))
                end
                R_pred[i,j,k] = acc / L
            end
        end
    end
    return R_pred
end

function test_inferenced_R()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0  # It should be NaN, but use 0 here becausde 0 * NaN = NaN.

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0::Matrix{Float64} = hcat([0.04])

    U,V,T,alpha = gibbs_sampling(R_obs, mask, D, mu_0, beta_0, W_0, nu_0, rho_0, nu_tilde_0, W_tilde_0, 100)
    R_pred = inferenced_R(U, V, T, alpha)
    print(typeof(R_pred), R_pred)
end

function sample_TK(K, mu_, Lambda_, alpha, X, mask, R, T_K_prev)
    N, M, _ = size(X)
    acc_Lambda::Matrix{Float64} = zeros(size(Lambda_))
    for i in 1 : N
        for j in 1 : M
            acc_Lambda += alpha[1,1] * mask[i,j,K] * (X[i,j,:] * X[i,j,:]')
        end
    end
    Lambda = Lambda_ + acc_Lambda

    acc_mu = zeros(size(mu_))
    for i in 1 : N
        for j in 1 : M
            acc_mu += alpha[1,1] * mask[i,j,K] * R[i,j,K] * X[i,j,:]
        end
    end
    mu = inv(Lambda) * (Lambda_ * T_K_prev + acc_mu)
    return rand(MvNormal(mu, PDMats.PDMat(Symmetric(inv(Lambda)))))
end

function test_sample_TK()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0::Matrix{Float64} = hcat([0.04])

    N,M,K = size(R)

    # initilize
    U1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T1 = init_T(W_0, nu_0, rho_0, beta_0, K)

    L = 1
    # Container
    U = zeros(D, N, L+1)
    V = zeros(D, M, L+1)
    T = zeros(D, K, L+1)
    alpha = zeros(size(W_tilde_0)[1],size(W_tilde_0)[2], L)

    U[:,:,1] = U1
    V[:,:,1] = V1
    T[:,:,1] = T1

    p = Progress(L)
    for l in 1 : L
        alpha[:,:,l] = sample_alpha(nu_tilde_0, W_tilde_0, mask, R, U[:,:,l], V[:,:,l], T[:,:,l])
        mu_U, Lambda_U = sample_theta(U[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_V, Lambda_V = sample_theta(V[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_T, Lambda_T = sample_theta_T(T[:,:,l], beta_0, rho_0, W_0, nu_0)

        # Update U
        Q = zeros(M, K, D)
        for k in 1 : K
            for j in 1 : M
                Q[j,k,:] = V[:,j,l] .* T[:,k,l]
            end
        end
        for i in 1 : N
            U[:,i,l+1] = sample_Ui(i, mu_U, Lambda_U, alpha[:,:,l], R_obs, Q, mask)
        end

        # Update V
        P = zeros(N, K, D)
        for k in 1 : K
            for i in 1 : N
                P[i,k,:] = U[:,i,l] .* T[:,k,l]
            end
        end
        for j in 1 : M
            V[:,j,l+1] = sample_Ui(j, mu_V, Lambda_V, alpha[:,:,l], R_obs, P, mask)
        end

        # Update T
        X = zeros(N, M, D)
        for i in 1 : N
            for j in 1 : M
                X[i,j,:] = U[:,i,l] .* V[:,j,l]
            end
        end
        T[:,1,l+1] = sample_T1(mu_T, Lambda_T, alpha[:,:,l], X, mask, T[:,2,l])
        for k in 2 : K-1
            T[:,k,l+1] = sample_Tk_mt2(k, mu_T, Lambda_T, alpha[:,:,l], X, mask, R_obs, T[:,k-1,l], T[:,k+1,l])
        end
        T[:,K,l+1] = sample_TK(K, mu_T, Lambda_T, alpha[:,:,l], X, mask, R_obs, T[:,K-1,l])
    end
    print(T, typeof(T))
end

function sample_Tk_mt2(k, mu_, Lambda_, alpha, X, mask, R, T_prev, T_next)
    N, M, _ = size(X)
    acc_Lambda::Matrix{Float64} = zeros(size(Lambda_))
    for i in 1 : N
        for j in 1 : M
            acc_Lambda += alpha[1,1] * mask[i,j,k] * (X[i,j,:] * X[i,j,:]')
        end
    end
    Lambda = 2Lambda_ + acc_Lambda

    acc_mu = zeros(size(mu_))
    for i in 1 : N
        for j in 1 : M
            acc_mu += alpha[1,1] * mask[i,j,k] * R[i,j,k] * X[i,j,:]
        end
    end
    mu = inv(Lambda) * (Lambda_ * (T_prev + T_next) + acc_mu)
    return rand(MvNormal(mu, PDMats.PDMat(Symmetric(inv(Lambda)))))
end

function test_sampleTk_mt2()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0::Matrix{Float64} = hcat([0.04])

    N,M,K = size(R)

    # initilize
    U1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T1 = init_T(W_0, nu_0, rho_0, beta_0, K)

    L = 1
    # Container
    U = zeros(D, N, L+1)
    V = zeros(D, M, L+1)
    T = zeros(D, K, L+1)
    alpha = zeros(size(W_tilde_0)[1],size(W_tilde_0)[2], L)

    U[:,:,1] = U1
    V[:,:,1] = V1
    T[:,:,1] = T1

    p = Progress(L)
    for l in 1 : L
        alpha[:,:,l] = sample_alpha(nu_tilde_0, W_tilde_0, mask, R, U[:,:,l], V[:,:,l], T[:,:,l])
        mu_U, Lambda_U = sample_theta(U[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_V, Lambda_V = sample_theta(V[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_T, Lambda_T = sample_theta_T(T[:,:,l], beta_0, rho_0, W_0, nu_0)

        # Update U
        Q = zeros(M, K, D)
        for k in 1 : K
            for j in 1 : M
                Q[j,k,:] = V[:,j,l] .* T[:,k,l]
            end
        end
        for i in 1 : N
            U[:,i,l+1] = sample_Ui(i, mu_U, Lambda_U, alpha[:,:,l], R_obs, Q, mask)
        end

        # Update V
        P = zeros(N, K, D)
        for k in 1 : K
            for i in 1 : N
                P[i,k,:] = U[:,i,l] .* T[:,k,l]
            end
        end
        for j in 1 : M
            V[:,j,l+1] = sample_Ui(j, mu_V, Lambda_V, alpha[:,:,l], R_obs, P, mask)
        end

        # Update T
        X = zeros(N, M, D)
        for i in 1 : N
            for j in 1 : M
                X[i,j,:] = U[:,i,l] .* V[:,j,l]
            end
        end
        T[:,1,l+1] = sample_T1(mu_T, Lambda_T, alpha[:,:,l], X, mask, T[:,2,l])
        for k in 2 : K-1
            T[:,k,l+1] = sample_Tk_mt2(k, mu_T, Lambda_T, alpha[:,:,l], X, mask, R_obs, T[:,k-1,l], T[:,k+1,l])
        end
    end
    print(T, typeof(T))
end

function sample_T1(mu_, Lambda_, alpha, X, mask, T_2)
    N, M, _ = size(X)
    acc_Lambda = zeros(size(Lambda_))
    for i in 1 : N
        for j in 1 : M
            acc_Lambda += alpha[1,1] * mask[i,j,1] * (X[i,j,:] * X[i,j,:]')
        end
    end
    Lambda = 2Lambda_ + acc_Lambda

    mu = (T_2 + mu_) ./ 2
    return rand(MvNormal(mu, PDMats.PDMat(Symmetric(inv(Lambda)))))
end

function test_sample_T1()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0::Matrix{Float64} = hcat([0.04])

    N,M,K = size(R)

    # initilize
    U1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T1 = init_T(W_0, nu_0, rho_0, beta_0, K)

    L = 1
    # Container
    U = zeros(D, N, L+1)
    V = zeros(D, M, L+1)
    T = zeros(D, K, L+1)
    alpha = zeros(size(W_tilde_0)[1],size(W_tilde_0)[2], L)

    U[:,:,1] = U1
    V[:,:,1] = V1
    T[:,:,1] = T1

    p = Progress(L)
    for l in 1 : L
        alpha[:,:,l] = sample_alpha(nu_tilde_0, W_tilde_0, mask, R, U[:,:,l], V[:,:,l], T[:,:,l])
        mu_U, Lambda_U = sample_theta(U[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_V, Lambda_V = sample_theta(V[:,:,l], beta_0, mu_0, W_0, nu_0)
        mu_T, Lambda_T = sample_theta_T(T[:,:,l], beta_0, rho_0, W_0, nu_0)

        # Update U
        Q = zeros(M, K, D)
        for k in 1 : K
            for j in 1 : M
                Q[j,k,:] = V[:,j,l] .* T[:,k,l]
            end
        end
        for i in 1 : N
            U[:,i,l+1] = sample_Ui(i, mu_U, Lambda_U, alpha[:,:,l], R_obs, Q, mask)
        end

        # Update V
        P = zeros(N, K, D)
        for k in 1 : K
            for i in 1 : N
                P[i,k,:] = U[:,i,l] .* T[:,k,l]
            end
        end
        for j in 1 : M
            V[:,j,l+1] = sample_Ui(j, mu_V, Lambda_V, alpha[:,:,l], R_obs, P, mask)
        end

        # Update T
        X = zeros(N, M, D)
        for i in 1 : N
            for j in 1 : M
                X[i,j,:] = U[:,i,l] .* V[:,j,l]
            end
        end
        T[:,1,l+1] = sample_T1(mu_T, Lambda_T, alpha[:,:,l], X, mask, T[:,2,l])
    end
    print(T,typeof(T))
end

"""
# Arguments
- X(Q(=V'*T): MxK, P(=U'*T): NxK)

# Note
Q_jk = V_j element_wise product T_k
"""
function sample_Ui(i, mu_, Lambda_, alpha, R, Q, mask)
    M, K, _ = size(Q)
    acc_Lambda = zeros(size(Lambda_))
    for k in 1 : K
        for j in 1 : M
            acc_Lambda += alpha[1,1] * mask[i,j,k] * (Q[j,k,:] * Q[j,k,:]')
        end
    end
    Lambda = Lambda_ + acc_Lambda

    acc_mu = zeros(size(mu_))
    for k in 1 : K
        for j in 1 : M
            acc_mu += alpha[1,1] * mask[i, j, k] * R[i, j, k] * Q[j,k,:]
        end
    end
    mu = inv(Lambda)*(Lambda_ * mu_ + acc_mu)
    return rand(MvNormal(mu, PDMats.PDMat(Symmetric(inv(Lambda)))))
end

function test_sample_Ui()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0::Matrix{Float64} = hcat([0.04])

    N,M,K = size(R)

    # initilize
    U1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T1 = init_T(W_0, nu_0, rho_0, beta_0, K)

    L = 1
    # Container
    U = zeros(D, N, L+1)
    V = zeros(D, M, L+1)
    T = zeros(D, K, L+1)
    alpha = zeros(size(W_tilde_0)[1],size(W_tilde_0)[2], L)

    U[:,:,1] = U1
    V[:,:,1] = V1
    T[:,:,1] = T1

    p = Progress(L)
    for l in 1 : L
        alpha[:,:,l] = sample_alpha(nu_tilde_0, W_tilde_0, mask, R, U[:,:,l], V[:,:,l], T[:,:,l])
        mu_U, Lambda_U = sample_theta(U[:,:,l], beta_0, mu_0, W_0, nu_0)

        # Update U
        Q = zeros(M, K, D)
        for k in 1 : K
            for j in 1 : M
                Q[j,k,:] = V[:,j,l] .* T[:,k,l]
            end
        end
        for i in 1 : N
            U[:,i,l+1] = sample_Ui(i, mu_U, Lambda_U, alpha[:,:,l], R_obs, Q, mask)
        end
    end
    print(typeof(U), U)
end

function sample_theta_T(T, beta_, rho_, W_, nu_)
    _, K = size(T)
    mu = (beta_*rho_ + T[:,1]) / (beta_ + 1)
    beta = beta_ + 1
    nu = nu_ + K
    acc = zeros(size(W_))
    for k in 2 : K
        tmp = T[:, k] - T[:, k-1]
        acc += (tmp * tmp')
    end
    W = inv(inv(W_) + acc + (beta_/(1+beta)) * ((T[:,1] - rho_) * (T[:,1] - rho_)'))

    Lambda_sample = rand(Wishart(nu, PDMats.PDMat(Symmetric(W))))
    mu_sample = rand(MvNormal(mu, PDMats.PDMat(Symmetric(inv(beta * Lambda_sample)))))
    return mu_sample, Lambda_sample
end

function test_sample_theta_T()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    N,M,K = size(R)

    # initilize
    T1 = init_T(W_0, nu_0, rho_0, beta_0, K)

    L = 1
    # Container
    T = zeros(D, K, L+1)

    T[:,:,1] = T1

    p = Progress(L)
    for l in 1 : L
        mu_T, Lambda_T = sample_theta_T(T[:,:,l], beta_0, rho_0, W_0, nu_0)
    end
end

function sample_theta(U, beta_, mu_, W_, nu_)
    _, N = size(U)
    U_bar = dropdims(sum(U, dims=2) ./ N, dims=2)
    S_bar::Matrix{Float64} = zeros(size(W_)[1], size(W_)[2])
    for i in 1 : N
        tmp = U[:,i] - U_bar
        S_bar += (tmp * tmp')/N
    end
    mu = (beta_*mu_ + N*U_bar) ./ (beta_ + N)
    beta = beta_ + N
    nu = nu_ + N
    W = inv(inv(W_) + N*S_bar + beta_ * N * ((mu_ - U_bar) * (mu_ - U_bar)') / (beta_ + N))

    Lambda_sample = rand(Wishart(nu, PDMats.PDMat(Symmetric(W))))
    mu_sample = rand(MvNormal(mu, PDMats.PDMat(Symmetric(inv(beta * Lambda_sample)))))
    return mu_sample, Lambda_sample
end

function test_sample_theta()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist


    N,M,K = size(R)

    # initilize
    U1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)

    L = 1
    # Container
    U = zeros(D, N, L+1)
    V = zeros(D, M, L+1)
    T = zeros(D, K, L+1)
    alpha = zeros(size(W_tilde_0)[1],size(W_tilde_0)[2], L)

    U[:,:,1] = U1
    V[:,:,1] = V1
    T[:,:,1] = T1

    p = Progress(L)
    for l in 1 : L
        mu_U, Lambda_U = sample_theta(U[:,:,l], beta_0, mu_0, W_0, nu_0)
        print(mu_U, Lambda_U)
    end
end


function multi_dot(X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64})
    L = length(X)
    acc::Float64 = 0.
    for i in 1 : L
        acc += X[i] * Y[i] * Z[i]
    end
    return acc
end

function test_multi_dot()
    @assert 2 == multi_dot([1.,1.], [1.,1.], [1.,1.])
end

"""
# Arguments
- W_tilde: DxD

# Return value
alpha: DxD
"""
function sample_alpha(nu_tilde, W_tilde, mask, R, U, V, T)
    N, M, K = size(R)
    nu = nu_tilde + sum(mask)
    acc = hcat([0.])
    for k in 1 : K
        for i in 1 : N
            for j in 1 : M
                acc[1,1] += mask[i,j,k] * (R[i,j,k] - multi_dot(U[:,i],V[:,j],T[:,k]))^2
            end
        end
    end
    W = inv(inv(W_tilde) + acc)
    return rand(Wishart(nu, PDMats.PDMat(Symmetric(W))))
end

function test_sample_alpha()
    R = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[mask] .= 0

    # hyper parameter for dimension of U,V and T
    D = 2

    # hyper parameter for U,V,T prior
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R prior
    rho_0 = ones(D)

    # hyper parameter for alpha prior
    nu_tilde_0 = 1
    W_tilde_0 = hcat([0.04]) # 1x1 matrix


    N,M,K = size(R)

    # initilize
    U1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V1,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T1 = init_T(W_0, nu_0, rho_0, beta_0, K)

   print(sample_alpha(nu_tilde_0, W_tilde_0, mask, R_obs, U1, V1, T1))
end

function creat_dummy_data(N::Int64, M::Int64, K::Int64, threshold::Float64)
    R = rand(N,M,K)
    mask = R .< threshold
    R[mask] .= 0
    R[.!mask] .= 1
    return R
end

function test_create_dummy_data()
    X = creat_dummy_data(2,2,4,0.5)
    print(X)
end

function missing_mask(X::Array{Float64, 3}, missing_rate::Float64)
    mask = rand(size(X)[1], size(X)[2], size(X)[3]) .< missing_rate
    return mask
end

function test_missing_mask()
    X = creat_dummy_data(2,2,4,0.5)
    mask = missing_mask(X, 0.7)
end

"""
Initilize U and V matrix
# Arguments
- W_0: DxD
- nu_0: Scalar
- mu_0: Dx1
- beta_0: Scalar
- N: scalar

# Return value
init_matrix: DxN
Lambda: DxD
mu: Dx1
"""
function init_model_params(W_0, nu_0, mu_0, beta_0, N)
    Lambda = rand(Wishart(nu_0, W_0))
    mu = rand(MvNormal(mu_0, PDMats.PDMat(Symmetric(inv(beta_0 * Lambda)))))
    init_matrix = rand(MvNormal(mu, PDMats.PDMat(Symmetric(inv(Lambda)))), N)
    return init_matrix, Lambda, mu
end

function test_init_model_params()
    D = 2
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist
    N = 4
    matrix, Lambda, mu = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    print(matrix)
    print(Lambda)
    print(mu)
    @assert (D, N) == size(matrix)
    @assert (D, D) == size(Lambda)
    @assert (D, ) == size(mu)
end

"""
Initilize T matrix
# Arguments
- W_0: DxD
- nu_0: Scalar
- rho_0: Dx1
- beta_0: Scalar
- K: scalar

# Return value
T: DxK
"""
function init_T(W_0, nu_0, rho_0, beta_0, K)
    D = size(W_0)[1]
    T = zeros(D, K)
    T_1, Lambda_T, mu_T = init_model_params(W_0, nu_0, rho_0, beta_0, 1)
    T[:,1] = T_1
    for i in 2 : K
        T[:, i] = rand(MvNormal(T[:, i-1], PDMats.PDMat(Symmetric(inv(Lambda_T)))))
    end
    return T
end

function test_init_T()
    D = 2
    rho_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist
    K = 4
    matrix = init_T(W_0, nu_0, rho_0, beta_0, K)
    print(matrix)
    @assert (D, K) == size(matrix)
end

#test_init_model_params()
#test_init_T()
#test_create_dummy_data()
#test_missing_mask()
#test_multi_dot()
#test_sample_alpha()
#test_sample_theta() #-> [1.36413, 2.11034][2.20263 -1.40809; -1.40809 1.09076]
#test_sample_theta_T() #-> [4.85556, 0.93177][0.374475 -0.760314; -0.760314 7.5789]
#test_sample_Ui()
#test_sample_T1()
#test_sampleTk_mt2()
#test_sample_TK()
#test_gibbs_sampling()
#test_inferenced_R()
demo()
