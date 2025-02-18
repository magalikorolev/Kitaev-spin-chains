using ITensors, ITensorMPS
using Plots
using LaTeXStrings
using GLM
using DataFrames
using LsqFit

J1 = 1
J2 = 1
N = 2*2*50


# Get Hamiltonian and sites
function kitaev_spin_chain(L,j1,j2)
    sites = siteinds("S=1/2", L)
    h = OpSum()
    for i in 1:2:L-2
        add!(h, 4*j1, "Sx", i, "Sx", i+1)
        add!(h, 4*j2, "Sy", i+1, "Sy", i+2)
    end
    return MPO(h, sites), sites
end


function two_pt_cor(L, sites)
    opOL = OpSum()
    for j in 1:2:L-1
        add!(opOL, 4, "Sx", j, "Sx", j+1)
    end
    return MPO(opOL,sites)
end

function four_pt_cor(L, sites)
    opOL = OpSum()
    for j in 1:2:L-1
        for i in 1:2:L-1
            add!(opOL, 16, "Sx", j, "Sx", j+1, "Sx", i, "Sx", i+1)
        end        
    end
    return MPO(opOL,sites)
end


# Perform DMRG to find the ground state and calculate fluctuations

# Create the Hamiltonian and the sites and get the GS
H, sites = kitaev_spin_chain(N,J1,J2)
psi = randomMPS(sites)
sweeps = Sweeps(35)
maxdim!(sweeps, 10, 20, 40, 80, 100, 180, 250, 300, 500, 800)
cutoff!(sweeps, 1e-10)
energy, psi0 = dmrg(H, psi, sweeps)

function calculate_fluctuations(L)

    # Calculate <Q>
    Q = two_pt_cor(L, sites)
    avgQ = inner(psi0', Q, psi0)

    # Calculate <Q^2>
    Q2 = four_pt_cor(L, sites)
    avgQ2 = inner(psi0', Q2, psi0)

    # Calculate fluctuations
    fluctuations = avgQ2 - avgQ^2 

    println("system size =", L)

    return fluctuations
end 

# Calculate the average value of O for different system sizes L
System_sizes = 2:2:N  # Change this range to explore more values of L
fluct = [calculate_fluctuations(x) for x in System_sizes]
fluct_re = real(fluct)

plot(System_sizes,fluct_re, xlabel="L",ylabel=L"F_{Q_{VB}}(L)",label=L" <(\Sigma_{j=2m-1}\sigma^x_j\sigma^x_{j+1})^2> - <\Sigma_{j=2m-1}\sigma^x_j\sigma^x_{j+1}>^2",title=L"Kitaev \; spin \; chain \; Hamiltonian, J_1 = J_2")

p2 = [1.0, 1.0, 1.0]
function fitF(x,k)
    return k[1] .* x .+ k[2] .* log.(x) .+ k[3]
end

fit_F = curve_fit(fitF,System_sizes,fluct_re,p2)
fit_y_F = fitF(System_sizes,coef(fit_F))
a,b,c = coef(fit_F)
a_round = round(a,digits=3)
b_round = round(b,digits=3)
c_round = round(c,digits=3)

plot!(System_sizes,fit_y_F,linestyle=:dash,label="$a_round x + $b_round log(x) + $c_round")
