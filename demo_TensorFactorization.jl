using LinearAlgebra
using Distributions

#R: NxMxK
#U: DxN
#V: DxM
#T: DxK

# check the type and dimenstion of input, output for each function and write test case

function test()
    R = rand(2,2,10)
    N = size(R)[1]
    M = size(R)[2]
    K = size(R)[3]

    D = 2
    mu_0 = zeros(D)
    beta_0 = 1
    W_0 = Matrix{Float64}(I, D, D)  # parameter for Wishart dist, DxD matrix
    nu_0 = D  # parameter for Wishart dist
    rho_0 = ones(D)

    # initilize
    U,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, N)
    V,_,_ = init_model_params(W_0, nu_0, mu_0, beta_0, M)
    T = init_T(W_0, nu_0, rho_0, beta_0, K)
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

test_init_model_params()
test_init_T()
