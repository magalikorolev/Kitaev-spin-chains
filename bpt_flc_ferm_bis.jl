using ITensors, ITensorMPS
using Plots
using LaTeXStrings
using GLM
using DataFrames
using LsqFit
using Statistics


J1 = 1
J2 = 1
N = 2*2*50


# Get Hamiltonian and sites
function kitaev_spin_chain(L,j1,j2)
    sites = siteinds("Fermion", L)
    h = OpSum()
    for i in 1:L-1
        add!(h, -2*j1, "Cdag", i, "C", i)
        add!(h, j1, "Id", i)
        add!(h, -j2, "Cdag", i, "C", i+1)
        add!(h, -j2, "Cdag", i+1, "C", i)
        add!(h, -j2, "Cdag", i, "Cdag", i+1)
        add!(h, -j2, "C", i+1, "C", i)
    end
    return MPO(h, sites), sites
end


function two_pt_cor(L, sites)
    opOL = OpSum()
    for j in 1:L
        add!(opOL, 1, "Cdag", j, "C", j)
        add!(opOL, -1/2, "Id", j)
    end
    return MPO(opOL,sites)
end

function four_pt_cor(L, sites)
    opOL = OpSum()
    for j in 1:L
        for i in 1:L
            add!(opOL, 1, "Cdag", j, "C", j, "Cdag", i, "C", i)
            add!(opOL, -1/2, "Cdag", j, "C", j)
            add!(opOL, -1/2, "Cdag", i, "C", i)
            add!(opOL, 1/4, "Id", j, "Id", i)
        end        
    end
    return MPO(opOL,sites)
end


# Perform DMRG to find the ground state and calculate fluctuations

# Create the Hamiltonian and the sites and get the GS
H, sites = kitaev_spin_chain(N,J1,J2)
psi = randomMPS(sites)
sweeps = Sweeps(50)
maxdim!(sweeps, 10, 20, 40, 80, 90, 100, 170, 200, 250, 300, 400, 500, 800)
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
fluct_re = 4*real(fluct)

plot(System_sizes,fluct_re, xlabel="L",ylabel=L"4*F_{Q}(L)",label=L"4 <(\Sigma_{m} c_m^\dag c_m - 1/2)^2> - <\Sigma_{m}c_m^\dag c_m - 1/2>^2",title=L"p-wave \; superconductor \; Hamiltonian, J_1 = J_2")

function fit_function(x,k)
    return k[1] .* x .+ k[2] .* log.(x) .+ k[3]
end

p0 = [1.0, 1.0, 1.0]

fit_log = curve_fit(fit_function,System_sizes,fluct_re,p0)
a,b,c = coef(fit_log)
a_round = round(a,digits=4)
b_round = round(b,digits=4)
c_round = round(c,digits=4)
fit_y = fit_function(System_sizes,coef(fit_log))

plot!(System_sizes,fit_y,linestyle=:dash,label="$a_round x + $b_round log(x) + $c_round")

