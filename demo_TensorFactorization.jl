using LinearAlgebra
using Distributions

#R: NxMxK
#U: DxN
#V: DxM
#T: DxK

# check the type and dimenstion of input, output for each function and write test case

function test()
    R = creat_dummy_data(2,2,4,0.5)
    missing_mask = missing_mask(R, 0.7)
    R_obs = deepcopy(R)
    R_obs[missing_mask] = NaN

    N = size(R)[1]
    M = size(R)[2]
    K = size(R)[3]

    D = 2

    # hyper parameter for U,V,T
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist

    # hyper parameter for R
    rho_0 = ones(D)

    # hyper parameter for α
    mu~_0 = 1
    W~_0::Matrix{Float64} = 0.04

    # initilize
    U,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T = init_T(W_0, nu_0, rho_0, beta_0, K)
end

function sample_alpha(nu~, I, W~, R, U, V, K)
    nu* = nu~ + sum(I)
    acc::Matrix{Float64} = 0
    for k in 1 : K
        for i in 1 : N
            for j in 1 : M
                acc[1,1] += I[i,j,k](R[i,j,k] - (U[:,i] ⋅ V[:,j] ⋅ T[:,k]))
            end
        end
    end
    W*::Matrix{Float64} = inv(inv(W~) + acc)
    return rand(Wishart(nu*, W*))
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
    mu = rand(MvNormal(mu_0, inv(beta_0 * Lambda)))
    init_matrix = rand(MvNormal(mu, inv(Lambda)), N)
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
        T[:, i] = rand(MvNormal(T[:, i-1], inv(Lambda_T)))
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
test_missing_mask()
