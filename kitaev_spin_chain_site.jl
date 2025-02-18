using ITensors, ITensorMPS
using Plots
using DataFrames


#arg1 = parse(Float64, ARGS[1])

# Parameters of the system
J1 = 0.01
J2 = 1
N = 2*2*50

# Get Hamiltonian and sites
function kitaev_spin_chain(L)
    sites = siteinds("S=1/2", L)
    h = OpSum()
    for i in 1:2:N-2
        add!(h, J1, "Sx", i, "Sx", i+1)
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


# Perform DMRG to find the ground state and calculate fluctuations
H, sites = kitaev_spin_chain(N)
psi = randomMPS(sites)
sweeps = Sweeps(20)
maxdim!(sweeps, 10, 20, 80, 100, 150)
cutoff!(sweeps, 1e-10)
energy, psi0 = dmrg(H, psi, sweeps)


function calculate_correl(loc_site)

    # Calculate <4ptcor>
    S4 = four_pt_cor(loc_site, sites)
    avgS4 = inner(psi0', S4, psi0)

    # Calculate <2ptcor>
    S2 = two_pt_cor(loc_site, sites)
    avgS2 = inner(psi0', S2, psi0)

    println("position = ", loc_site)

    return avgS4, avgS2
end 

site_list = 1:2:N-3
S4_list = []
S2_list = []
for s in site_list
    a,b = calculate_correl(s)
    push!(S4_list,real(a))
    push!(S2_list,real(b))
end


plot(site_list,S4_list,xlabel="j",label="<S_j^xS_j+1^zS_j+2^zS_j+3^x>(J1/J2=0.01)")
#plot!(sites_list,real(S4_list) * 8,label="<σ^xσ^zσ^zσ^x>")
plot!(site_list,S2_list,label="<S_j+1^yS_j+2^y>(J1/J2=0.01)")
#plot!(sites_list,real(S2_list) * 4,label="<σ^yσ^y>")




#open("results_chiral/FINAL_flc_chiral_U_$arg1.txt", "w") do file
#    for i in 1:length(System_sizes)
#        println(file, "$(System_sizes[i]),$(fluct_re[i])")
#    end
#end 
