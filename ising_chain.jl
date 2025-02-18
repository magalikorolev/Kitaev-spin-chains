using ITensors, ITensorMPS
using Plots
using DataFrames
using Distributions

#arg1 = parse(Float64, ARGS[1])

# Parameters of the system
J_list = 10 .^ range(-2, stop=2, length=100)
Δ_list = 10 .^ range(-2, stop=2, length=100)
Δ = 1
J = 1
N = 2*2*50

function generate_normal_list(n, a, sigma)
    dist = Normal(a, sigma)  # Define normal distribution with mean `a` and std deviation `sigma`
    rand(dist, n)            # Generate `n` random samples
end

variance = 0.75
J_disorder_list = generate_normal_list(N,J,variance)

# Get Hamiltonian and sites
function ising_spin_chain(L, d, j)
    sites = siteinds("S=1/2", L)
    h = OpSum()
    for i in 1:N-1
        add!(h, J_disorder_list[i]/4, "Sx", i, "Sx", i+1)
        add!(h, generate_normal_list(N,d,variance)[i]/2, "Sz", i)
    end
    return MPO(h, sites), sites
end


function four_pt_cor(j, sites)
    opOR = OpSum()
    add!(opOR, 1/16, "Sx", j, "Sz", j+1, "Sz", j+2, "Sx", j+3)
    return MPO(opOR,sites)
end

function two_pt_cor(j, sites)
    opOL = OpSum()
    add!(opOL, 1/4, "Sy", j, "Sy", j+1)
    return MPO(opOL,sites)
end

function two_pt_cor_z(j, sites)
    opOL = OpSum()
    add!(opOL, 1/4, "Sz", j, "Sz", j+1)
    return MPO(opOL,sites)
end

function two_pt_cor_x(j, sites)
    opOL = OpSum()
    add!(opOL, 1/4, "Sx", j, "Sx", j+1)
    return MPO(opOL,sites)
end


S2_list_y = []
S2_list_z = []
S2_list_x = []

iter = 1
for D in Δ_list
    println("Iteration ", iter)

    H, sites = ising_spin_chain(N, D, J)
    psi = randomMPS(sites)
    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 80, 100, 150)
    cutoff!(sweeps, 1e-10)
    energy, psi0 = dmrg(H, psi, sweeps)

    # Calculate <2ptcor>
    list_S2_y = []
    list_S2_z = []
    list_S2_x = []
    for j in 1:5:N-1
        S2_y = two_pt_cor(j, sites)
        avgS2_y = inner(psi0', S2_y, psi0)
        push!(list_S2_y,avgS2_y)
        S2_z = two_pt_cor_z(j, sites)
        avgS2_z = inner(psi0', S2_z, psi0)
        push!(list_S2_z,avgS2_z)
        S2_x = two_pt_cor_x(j, sites)
        avgS2_x = inner(psi0', S2_x, psi0)
        push!(list_S2_x,avgS2_x)
    end


    final_S2_y = mean(list_S2_y)
    final_S2_z = mean(list_S2_z)
    final_S2_x = mean(list_S2_x)

    push!(S2_list_y,real(final_S2_y))
    push!(S2_list_z,real(final_S2_z))
    push!(S2_list_x,real(final_S2_x))


    iter = iter + 1
end



plot(Δ_list, S2_list_y * 4, xscale=:log,xlabel="Δ/J",label="<σ^yσ^y>",title="Ising transverse chain bis, σ = $variance×μ")
plot!(Δ_list, S2_list_z * 4,xscale=:log,label="<σ^zσ^z>")
#plot!(Δ_list,  S2_list_x * 4 ,xscale=:log,label="-<σ^xσ^x> ")




#plot!(sites_list,real(S2_list) * 4,label="<σ^yσ^y>")
hline!([-1], linestyle=:dash, color=:black,label=false)
hline!([2/(pi)], linestyle=:dash, color=:black,label=false)
hline!([2/(3*pi)], linestyle=:dash,color=:black, label=false)
vline!([0.5],linestyle=:dash, color=:black,label=false)



#-------------------------------------------------------------------------------

d2ypt_dx = diff((S2_list_y * 4)) ./ diff(Δ_list)
d2zpt_dx = diff((S2_list_z * 4)) ./ diff(Δ_list)
d2xpt_dx = diff((S2_list_x * 4)) ./ diff(Δ_list)
d2xbispt_dx = diff((1 .- S2_list_x * 4)) ./ diff(Δ_list)

# Define x values for the derivative (midpoints of the intervals)
x_mid = (Δ_list[1:end-1] .+ Δ_list[2:end]) ./ 2

# Plot the derivative
plot(x_mid, -d2ypt_dx, xscale=:log10,ylims=[-0.5,3],label="d<σ^yσ^y>/d(Δ/J)", xlabel="Δ/J", title="Derivative of the correlation functions")
plot!(x_mid,- d2zpt_dx, xscale=:log10,label="d<σ^zσ^z>/d(Δ/J)")
#plot!(x_mid, d2xpt_dx, xscale=:log10,label="d<σ^xσ^x>/d(Δ/J)")
#plot!(x_mid, - d2xbispt_dx, xscale=:log10,label="d(1-<σ^xσ^x>)/d(Δ/J)")
vline!([0.5],linestyle=:dash, color=:black,label=false)
hline!([0],linestyle=:dash, color=:black,label=false)

