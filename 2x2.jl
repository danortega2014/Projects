using POMDPs,POMDPModelTools

Base.@kwdef struct mypomdp <: POMDP{Int64, Int64, Int64} # POMDP{State, Action, Observation}
    discount_factor::Float64 = 0.95 # discount
end


POMDPs.states(pomdp::mypomdp) = [1,2]
POMDPs.stateindex(pomdp::mypomdp, s::Int64) = s 


POMDPs.actions(::mypomdp) = [1,2]
POMDPs.actionindex(pomdp::mypomdp, a::Int64) = a 



function POMDPs.transition(pomdp::mypomdp, s::Int64, a::Int64)
    return SparseCat([1, 2], [.40, .60])
end 


POMDPs.observations(::mypomdp) = [1,2,3,4]
POMDPs.obsindex(::mypomdp, o::Int) = o


function POMDPs.observation(pomdp::mypomdp, a::Int64, sp::Int64)
    if a == 1  &&  sp == 2
        return SparseCat([1,2,3,4], [.10, .70, .10,.10]) # sparse categorical distribution
    elseif a== 1 && sp == 1
        return SparseCat([1,2,3,4], [.70, .10, .10,.10])
    elseif a==2 && sp==1
        return SparseCat([1,2,3,4], [.10, .10,.70,.10])
    else 
        return SparseCat([1,2,3,4], [.10, .10,.10,.70])
    end
end


function POMDPs.reward(pomdp::mypomdp, s::Int64, a::Int64)
    if a == 1 && s == 2
        return -1.0

    elseif a == 1 && s== 1
        return 1.0
    elseif a == 2 && s== 1
        return 0.0
    else
        return -1.0 # a= 2 s= 2
    end
end

POMDPs.initialstate(::mypomdp) = SparseCat([1,2], [.50,.50])
POMDPs.discount(pomdp::mypomdp) = pomdp.discount_factor





m = mypomdp()

using QMDP
solver = QMDPSolver()

policy = POMDPs.solve(solver, m)


using FIB
using POMDPModels

solver = FIBSolver()
policy = solve(solver,m)

rsum = 0.0
using POMDPSimulators
solver = QMDPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")