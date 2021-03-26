using POMDPs,POMDPModelTools



Base.@kwdef struct pomdp3x3 <: POMDP{Int64, Int64, Int64} # POMDP{State, Action, Observation
    discount_factor::Float64 = 0.95 # discount
end


POMDPs.states(pomdp::pomdp3x3) = [1,2,3]
POMDPs.stateindex(pomdp::pomdp3x3, s::Int64) = s 


POMDPs.actions(::pomdp3x3) = [1,2,3]
POMDPs.actionindex(pomdp::pomdp3x3, a::Int64) = a 



function POMDPs.transition(pomdp::pomdp3x3, s::Int64, a::Int64)
    return SparseCat([1, 2, 3], [.10, .30, .60])
end 


POMDPs.observations(::pomdp3x3) = [1,2,3,4,5,6,7,8,9]
POMDPs.obsindex(::pomdp3x3, o::Int) = o


function POMDPs.observation(pomdp::pomdp3x3, a::Int64, sp::Int64)
    if a == 1  &&  sp == 1
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.05,.05,.05,.05,.05,.05,.05,.60]) # sparse categorical distribution
    elseif a== 1 && sp == 2
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.05,.05,.05,.05,.05,.05,.60,.05]) 
    elseif a==1 && sp==3
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.05,.05,.05,.05,.05,.60,.05,.05]) 
    elseif a == 2  &&  sp == 1
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.05,.05,.05,.05,.60,.05,.05,.05])  # sparse categorical distribution
    elseif a== 2 && sp == 2
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.05,.05,.05,.60,.05,.05,.05,.05]) 
    elseif a==2 && sp==3
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.05,.05,.60,.05,.05,.05,.05,.05]) 
    elseif a== 3 && sp == 1
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.05,.60,.05,.05,.05,.05,.05,.05]) 
    elseif a==3 && sp==2
        return SparseCat([1,2,3,4,5,6,7,8,9], [.05,.60,.05,.05,.05,.05,.05,.05,.05]) 
    else 
        return SparseCat([1,2,3,4,5,6,7,8,9], [.60,.05,.05,.05,.05,.05,.05,.05,.05]) 
    end
end


function POMDPs.reward(pomdp::pomdp3x3, s::Int64, a::Int64)
    if a == 1  &&  s == 1
        return 13.0
    elseif a== 1 && s == 2
        return 1.0
    elseif a== 1 && s == 3
        return 7.0 
    elseif a== 2 && s == 1
        return 4.0 
    elseif a== 2 && s == 2
        return 3.0 
    elseif a== 2 && s == 3
        return 6.0 
    elseif a== 3 && s == 1
        return -1.0
    elseif a== 3 && s == 2
        return 2.0
    elseif a== 3 && s == 3
        return 8.0 
    end
end

POMDPs.initialstate(::pomdp3x3) = SparseCat([1,2,3], [.33,.33,.34])
POMDPs.discount(pomdp::pomdp3x3) = pomdp.discount_factor





m = pomdp3x3()

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
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")

