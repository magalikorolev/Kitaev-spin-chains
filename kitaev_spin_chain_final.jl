using ITensors, ITensorMPS
using Plots
using DataFrames


#arg1 = parse(Float64, ARGS[1])

# Parameters of the system
#J1_list = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.6 , 0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4 , 1.5, 1.75 , 2 , 2.5, 5, 7.5, 10, 25, 50, 75, 100]
J1_list = 10 .^ range(-2, stop=2, length=100)  # x-axis values
J2 = 1
N = 2*2*50

# Get Hamiltonian and sites
function kitaev_spin_chain(L, j1)
    sites = siteinds("S=1/2", L)
    h = OpSum()
    for i in 1:2:N-2
        add!(h, j1, "Sx", i, "Sx", i+1)
        add!(h, J2, "Sy", i+1, "Sy", i+2)
    end
    return MPO(h, sites), sites
end


function four_pt_cor(j, sites)
    opOR = OpSum()
    add!(opOR, 1, "Sx", j, "Sz", j+1, "Sz", j+2, "Sx", j+3)
    return MPO(opOR,sites)
end

function two_pt_cor(j, sites)
    opOL = OpSum()
    add!(opOL, 1, "Sy", j+1, "Sy", j+2)
    return MPO(opOL,sites)
end


S4_list = []
S2_list = []
iter = 1
for J1 in J1_list
    println("Iteration ", iter)

    H, sites = kitaev_spin_chain(N, J1)
    psi = randomMPS(sites)
    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 80, 100, 150)
    cutoff!(sweeps, 1e-10)
    energy, psi0 = dmrg(H, psi, sweeps)

    # Calculate <4ptcor>
    S4 = four_pt_cor(Int(N/2)-1, sites)
    avgS4 = inner(psi0', S4, psi0)

    # Calculate <2ptcor>
    S2 = two_pt_cor(Int(N/2)-1, sites)
    avgS2 = inner(psi0', S2, psi0)

    push!(S4_list, real(avgS4))
    push!(S2_list, real(avgS2))

    iter = iter + 1
end


plot(J1_list, S4_list * 16,xlabel="J1/J2",xscale=:log10,label="<σ^xσ^zσ^zσ^x>")
#plot!(sites_list,real(S4_list) * 8,label="<σ^xσ^zσ^zσ^x>")
plot!(J1_list, S2_list * 4,xscale=:log,label="<σ^yσ^y>")
#plot!(sites_list,real(S2_list) * 4,label="<σ^yσ^y>")
hline!([-1], linestyle=:dash, color=:black,label=false)
hline!([-2/(pi)], linestyle=:dash, color=:black,label=false)
hline!([2/(3*pi)], linestyle=:dash,color=:black, label=false)
vline!([1],linestyle=:dash, color=:black,label=false)



#-------------------------------------------------------------------------------

d2pt_dx = diff((S2_list * 4)) ./ diff(J1_list)
d4pt_dx = diff((S4_list * 16)) ./ diff(J1_list)

# Define x values for the derivative (midpoints of the intervals)
x_mid = (J1_list[1:end-1] .+ J1_list[2:end]) ./ 2

# Plot the derivative
plot(x_mid, d4pt_dx, xscale=:log10,label="d<σ^xσ^zσ^zσ^x>/d(J1/J2)")
plot!(x_mid, d2pt_dx, xscale=:log10,label="d<σ^yσ^y>/d(J1/J2)", xlabel="J1/J2", title="Derivative of the correlation functions")
hline!([0],linestyle=:dash, color=:black,label=false)

