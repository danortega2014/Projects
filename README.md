# Prisoner's Dilemma in a POMDP framework, optimal policy using Q learning algorthim provided in QMDP package. 

using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPSimulators, QMDP

m = QuickPOMDP(
    states = ["Confessed", "Kept Silent"],
    actions = ["Confess", "Keep Silent"],
    observations = ["ShortPrisonSentence", "MedPrisonSentence", "LongPrisonSentence", "Noprisonsentence"],
    initialstate = Uniform(["Confessed", "Kept Silent"]),
    discount = 0.95,

    transition = function (s, a)
        return SparseCat(["Confessed", "Kept Silent"],  [0.60, 0.40])
    end,

    observation = function (s, a, sp)
        if a == "Confess"  &&  sp == "Confessed"
            return SparseCat(["ShortPrisonSentence", "MedPrisonSentence", "LongPrisonSentence", "Noprisonsentence"], [.10, .70, .10,.10]) # sparse categorical distribution
        elseif a=="Confess" && sp == "Kept Silent"
            return SparseCat(["ShortPrisonSentence", "MedPrisonSentence", "LongPrisonSentence", "Noprisonsentence"], [.70, .10, .10,.10])
        elseif a=="Keep Silent" && sp=="Confessed"
            return SparseCat(["ShortPrisonSentence", "MedPrisonSentence", "LongPrisonSentence", "Noprisonsentence"], [.10, .10,.70,.10])
        else SparseCat(["ShortPrisonSentence", "MedPrisonSentence", "LongPrisonSentence", "Noprisonsentence"], [.10, .10,.10,.70])
        end
    end,

    reward = function (s, a)
        if a == "Confess" && s == "Confessed"
            return -1.0
        elseif a == "Keep Silent" && s== "Kept Silent"
            return 10.0
        elseif a== "Keep Silent" && s=="Confessed"
            return -10.0
        else
            return 0.0 # a= confess s=kept silent
        end
    end
)



solver = QMDPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=1000)
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")
