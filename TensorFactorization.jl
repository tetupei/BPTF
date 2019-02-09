
#R: NxMxK
#U: DxN
#V: DxM
#T: DxK

struct BPTFModel
    mu_0::Float64
    beta_0::Float64
    W_0::Matrix{Float64}
    nu_0::Float64
end

function init()

end
