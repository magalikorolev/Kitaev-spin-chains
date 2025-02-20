using ITensors, ITensorMPS
using Plots
using DataFrames
using FFTW


#arg1 = parse(Float64, ARGS[1])

# Parameters of the system
Δ_list = 10 .^ range(-2, stop=2, length=100)
J = 1
L = 2*2*50

# Get Hamiltonian and sites
function ising_spin_chain(l, d)
    sites = siteinds("S=1/2", l)
    h = OpSum()
    for i in 1:l-1
        add!(h, J*4, "Sz", i, "Sz", i+1)
        add!(h, d*2, "Sx", i)
    end
    return MPO(h, sites), sites
end



function sigm_x(j, sites)
    opOR = OpSum()
    add!(opOR, 2, "Sx", j)
    return MPO(opOR,sites)
end

function sigm_y(j, sites)
    opOL = OpSum()
    add!(opOL, 2, "Sy", j)
    return MPO(opOL,sites)
end

function sigm_z(j, sites)
    opOL = OpSum()
    add!(opOL, 2, "Sz", j)
    return MPO(opOL,sites)
end

function calculate_avg_sigm(loc_site)

    SX = sigm_x(loc_site, sites)
    avgSX = inner(psi0', SX, psi0)
    SY = sigm_y(loc_site, sites)
    avgSY = inner(psi0', SY, psi0)
    SZ = sigm_z(loc_site, sites)
    avgSZ = inner(psi0', SZ, psi0)

    return avgSX, avgSY, avgSZ
end 

Sx_0 = []
Sx_pi = []
Sy_0 = []
Sy_pi = []
Sz_0 = []
Sz_pi = []
iter = 1
for D in Δ_list
    println("Iteration ", iter)

    H, sites = ising_spin_chain(L, D)
    psi = randomMPS(sites)
    sweeps = Sweeps(20)
    maxdim!(sweeps, 10, 20, 80, 100, 150)
    cutoff!(sweeps, 1e-10)
    energy, psi0 = dmrg(H, psi, sweeps)

    Sx_pos = []
    Sy_pos = []
    Sz_pos = []
    
    for s in 1:length(sites)
        a,b,c = calculate_avg_sigm(s)
        push!(Sx_pos,real(a))
        push!(Sy_pos,real(b))
        push!(Sz_pos,real(c))
    end

    N = length(sites)
    sx_k = fftshift(fft(Float64.(Sx_pos))) / N 
    sy_k = fftshift(fft(Float64.(Sy_pos))) / N  
    sz_k = fftshift(fft(Float64.(Sz_pos))) / N  
    k_vals = fftshift(fftfreq(N, 1)) * 2π

    push!(Sx_0,sx_k[1])
    push!(Sx_pi,sx_k[Int(N/2)+1])
    push!(Sy_0,sy_k[1])
    push!(Sy_pi,sy_k[Int(N/2)+1])
    push!(Sz_0,sz_k[1])
    push!(Sz_pi,sz_k[Int(N/2)+1])

    iter = iter + 1
end

plot(Δ_list, abs.(real(Sx_0)), xscale=:log, label="|<σ_x(k=0)>|",xlabel="Δ/J")
plot!(Δ_list, real(Sx_pi), label="<σ_x(k=π)>")
plot!(Δ_list, real(Sy_0), label="<σ_y(k=0)>")
plot!(Δ_list, real(Sy_pi), label="<σ_y(k=π)>")
plot!(Δ_list, real(Sz_0), label="<σ_z(k=0)>")
plot!(Δ_list, real(Sz_pi), label="<σ_z(k=π)>")



plot(Δ_list, (abs.(real(Sz_0)) .- real(Sz_pi)) ./ 2, label="(<σ_z(k=0)> - <σ_z(k=π)>)/2",xlabel="Δ/J",xscale=:log,legend=:right)
plot!(Δ_list, ((real(Sx_0)) .- real(Sx_pi)) ./ 2, label="(|<σ_x(k=0)>| - <σ_x(k=π)>)/2")
plot!(Δ_list, (real(Sy_0) .- real(Sy_pi)) ./ 2, label="(<σ_y(k=0)> - <σ_y(k=π)>)/2")