using ITensors, ITensorMPS
using Plots
using DataFrames
using FFTW


#arg1 = parse(Float64, ARGS[1])

# Parameters of the system
Δ = 100
J = 1
N = 2*2*50

# Get Hamiltonian and sites
function ising_spin_chain(L)
    sites = siteinds("S=1/2", L)
    h = OpSum()
    for i in 1:N-1
        add!(h, J, "Sx", i, "Sx", i+1)
        add!(h, Δ, "Sz", i)
    end
    return MPO(h, sites), sites
end


function two_pt_cor_x(j, sites)
    opOR = OpSum()
    add!(opOR, 1, "Sx", j, "Sx", j+1)
    return MPO(opOR,sites)
end

function two_pt_cor_y(j, sites)
    opOL = OpSum()
    add!(opOL, 1, "Sy", j, "Sy", j+1)
    return MPO(opOL,sites)
end

function two_pt_cor_z(j, sites)
    opOL = OpSum()
    add!(opOL, 1, "Sz", j, "Sz", j+1)
    return MPO(opOL,sites)
end



function sigm_x(j, sites)
    opOR = OpSum()
    add!(opOR, 1, "Sx", j)
    return MPO(opOR,sites)
end

function sigm_y(j, sites)
    opOL = OpSum()
    add!(opOL, 1, "Sy", j)
    return MPO(opOL,sites)
end

function sigm_z(j, sites)
    opOL = OpSum()
    add!(opOL, 1, "Sz", j)
    return MPO(opOL,sites)
end

# Perform DMRG to find the ground state and calculate fluctuations
H, sites = ising_spin_chain(N)
psi = randomMPS(sites)
sweeps = Sweeps(20)
maxdim!(sweeps, 10, 20, 80, 100, 150)
cutoff!(sweeps, 1e-10)
energy, psi0 = dmrg(H, psi, sweeps)


function calculate_correl(loc_site)

    # Calculate <4ptcor>
    S2x = two_pt_cor_x(loc_site, sites)
    avgS2x = inner(psi0', S2x, psi0)

    # Calculate <2ptcor>
    S2 = two_pt_cor(loc_site, sites)
    avgS2 = inner(psi0', S2, psi0)

    S2z = two_pt_cor_z(loc_site, sites)
    avgS2z = inner(psi0', S2z, psi0)

    SX = sigm_x(loc_site, sites)
    avgSX = inner(psi0', SX, psi0)
    SY = sigm_y(loc_site, sites)
    avgSY = inner(psi0', SY, psi0)
    SZ = sigm_z(loc_site, sites)
    avgSZ = inner(psi0', SZ, psi0)

    println("position = ", loc_site)

    return avgS2x, avgS2, avgS2z, avgSX, avgSY, avgSZ
end 

site_list = 1:N-2
S2x_list = []
S2_list = []
S2z_list = []
SX_list = []
SY_list = []
SZ_list = []
for s in site_list
    a,b,c,d,e,f = calculate_correl(s)
    push!(S2x_list,real(a))
    push!(S2_list,real(b))
    push!(S2z_list,real(c))
    push!(SX_list,real(d))
    push!(SY_list,real(e))
    push!(SZ_list,real(f))
end


plot(site_list,S2x_list*4,xlabel="j",label="<σ^xσ^x>")
plot!(site_list,S2_list*4,label="<σ^yσ^y>")
plot!(site_list,S2z_list*4,label="<σ^zσ^z>")

plot(site_list,SX_list,label="<s^x>")
plot!(site_list,SY_list,label="<s^y>")
plot!(site_list,SZ_list,label="<s^z>")




N = length(site_list)
sx_k = fftshift(fft(Float64.(SX_list))) / N 
sy_k = fftshift(fft(Float64.(SY_list))) / N  
sz_k = fftshift(fft(Float64.(SZ_list))) / N  
k_vals = fftshift(fftfreq(N, 1)) * 2π  # Convert to physical wavevector


plot(k_vals,real(sx_k))
plot!(k_vals,real(sy_k))
plot!(k_vals,real(sz_k))

