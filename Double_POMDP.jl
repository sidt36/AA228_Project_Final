using POMDPs, QuickPOMDPs, POMDPTools
using LinearAlgebra
using DiscreteValueIteration
using POMDPSimulators
using Plots
using ElectronDisplay
using Plots
#using QMDP
using ARDESPOT
using Statistics
using Random
using StatsBase

ElectronDisplay.CONFIG.single_window = true
plot([15],[15],xlims = (0,31),ylims = (0,31),seriestype=:scatter,ms = 8,legend = false,color = :green)



function observation_fn(sp)

  n = length(sp)/2
  states = [sp]
  probabs = [0.8]

  p = 0.2/(4*n)


  for i = 1:2*n
    tup1 = []
    tup2 = []
    count = 1
    for e in sp
      if (count == i)
      append!(tup1,min(e+1,30))
      append!(tup2,max(e-1,1))
      else
      append!(tup1,e)
      append!(tup2,e)
      end
      count+=1
    end

    t1 = Tuple(e for e in tup1)
    t2 = Tuple(e for e in tup2)

    append!(states,[t1])
    append!(probabs,[p])

    append!(states,[t2])
    append!(probabs,[p])

  end
  return SparseCat(states,probabs)

end


function initialize(num_states=1,num_aircrafts = 3)
  s= [tuple(rand(1:30, 1, 2*num_aircrafts)...)]
  for i = 1:num_states-1
    s = vcat(s,tuple(rand(1:30, 1, 2*num_aircrafts)...))
  end
  return s
end 

function out_of_bounds(s)
  return sum(count(s.>30) + count(s.<1))
end


function reward(s,goal = Vector([15,15]))
  Points = Dict()
  reward = 0
  #print(size(s))
  n = Int(length(s)/2)
  s_vector=[i for i=s]
  for i =  1:n
    Points[i] = s_vector[2*i-1:2*i]
    d = norm(goal -Points[i])
    spread = 7
    if (d==0)
      reward+=100
    elseif  (d<10)
      reward+= 100*exp(-d/spread)
    end  

  end
  
  reward -= out_of_bounds(s_vector)*150

  for i = 0:n-1
    for j = 1:n-i-1
      if (all(Points[i+1].==goal) || all(Points[j+1].==goal))
        continue
      end
      dist = norm(Points[i+1]-Points[j+1])

      if (dist>4)
      reward += 0 
      else
        reward += 0 -100*exp(-dist/4)
      end
    end    
  end     
  return reward

end


function next_states(si,theta)
  return si + [Int(sign(if abs(cos(deg2rad(theta)))<0.01; 0 else; cos(deg2rad(theta)) end)) Int(sign(if abs(sin(deg2rad(theta)))<0.01; 0 else; sin(deg2rad(theta)) end))]
end


function Transition(s,a)
  goal = [15 15]
  Points = Dict()
  actions = Dict()
  direction_list = []
  probability_list = []
  s_vector=[i for i=s]
  T = Dict()
  n = Int(size(s,1)/2)
  for i = 1:n
    Points[i] = s_vector[2*i-1:2*i]
    ai = a[i]
    theta = Int((ai-1)/8*360)
    theta_l = Int((ai-1)/8*360) + 45
    theta_r = Int((ai-1)/8*360) - 45
    append!(probability_list, [[0.8,0.1,0.1]])
    append!(direction_list, [[theta,theta_l,theta_r]])
  end
  direction_combos = collect(reduce(vcat,(Iterators.product(direction_list...))))
  probability_list = collect(reduce(vcat,(Iterators.product(probability_list...))))

  t = Dict()

  for (idx,ds) in enumerate(direction_combos)
    p = 1
    tup = []
    for i = 0:n-1
      if (all(Points[i+1]'==goal))
        t[i+1] = goal
        p = p*probability_list[idx][i+1]
      else
        t[i+1] = next_states(Points[i+1]',ds[i+1])
        p = p*probability_list[idx][i+1]
        if (out_of_bounds(t[i+1])>0)
          t[i+1] = [-1 -1]
        end  
      end
      append!(tup,t[i+1][1,1])
      append!(tup,t[i+1][1,2])
    end  
    if tuple(tup) in keys(T)
      #print('dup')
      T[tuple(tup)] +=p
    else
      T[tuple(tup)]  =  p
    end 
  end
  norm =1.0/sum(values(T))
  for key in keys(T)
    T[key] = T[key]*norm
  end  
  return T

  

end
#println(Transition([15 15 15 15],[1 1]))

function state_space(n=3)
    state_list = [collect(1:30) for p = 1:2*n]
    states = collect(reduce(vcat,((Iterators.product(state_list...)))))
    push!(states,(-1,-1))
    return states
end
function statespace(n=1)
  #print(n)
  if n==1
      list=vec([(i,j) for i=1:30,j=1:30])
      push!(list,(-1,-1))
      return list
  end
  low_list=statespace(n-1)
  one_list=statespace(1)
  list=vec([(i...,j...) for i=low_list,j=one_list])
  return list
end

function action_space(n=3)
    if (n>1)
        action_list = [collect(1:8) for p = 1:n]
        actions = collect(reduce(vcat,((Iterators.product(action_list...)))))
    else
        actions = [i for i=1:8]
    end
    return actions
end


#solver = QMDPSolver()
#policy = solve(solver, m)

# rsum = 0.0
# for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
#     println("s: $s, b: $([s=>pdf(b,s) for s in states(m)]), a: $a, o: $o")
#     global rsum += r
# end
# println("Undiscounted reward was $rsum.")
ns = 1
m1 = QuickPOMDP(
    states = statespace(ns) ,
    actions = action_space(ns) ,
    initialstate = Uniform(state_space(ns)),
    #initialstate=Deterministic((25,28)),
    observations = statespace(ns),
    discount = 0.8,
    #isterminal = s -> any([e == -1 for e in s]),
    transition = function (s, a)
       transition_dict = Transition(s,a)
       transition_tuples=[]
       for key in keys(transition_dict)
        push!(transition_tuples,Tuple(Tuple(key[1])))
       end
       #state_list = state_space(ns) 
       #probabs = []
      return SparseCat(transition_tuples,collect(values(transition_dict)))
    end,



    reward = function (s, a)
        return reward(s)
    end,

    render = function (step)
        cx = [step.s[i] for i = 1:2:2*ns]
        cy = [step.s[i] for i = 2:2:2*ns]
        color = [:blue, :red,:green]
        return plot!(cx,cy,xlims = (0,31),ylims = (0,31),seriestype=:scatter,ms = 4*pdf(step.b, step.s),legend = false, color = color[1:ns])
      end,
  

    observation = function (a,sp)
      return observation_fn(sp)
    end
    
)
ns=2
m2 = QuickPOMDP(
    states = statespace(ns) ,
    actions = action_space(ns) ,
    #initialstate = Uniform(state_space(ns)),
    initialstate=Uniform([(13,13,22,21),(13,14,21,22)]),
    #initialstate=Deterministic((9,9,18,17)),
    discount = 0.8,
    observations = statespace(ns),
    isterminal = s -> any([e == -1 for e in s]),
    transition = function (s, a)
       transition_dict = Transition(s,a)
       transition_tuples=[]
       for key in keys(transition_dict)
        push!(transition_tuples,Tuple(Tuple(key[1])))
       end
       #state_list = state_space(ns) 
       #probabs = []
      return SparseCat(transition_tuples,collect(values(transition_dict)))
    end,

    #= observation = function (s, a, sp)
        state_list = state_space(ns)
        probab = []
        for s1 in state_list
            if (all(s1 == sp))
                append!(probabs, 1)
            else
                append!(probabs, 0)
            end
             
        end    
        return SparseCat(state_list, probabs)
    end, =#

    reward = function (s, a)
        return reward(s)
    end,
    render = function (step)
        cx = [step.s[i] for i = 1:2:2*ns]
        cy = [step.s[i] for i = 2:2:2*ns]
        color = [:blue, :red,:green]
        return plot!(cx,cy,xlims = (0,31),ylims = (0,31),seriestype=:scatter,ms = 4*pdf(step.b, step.s),legend = false, color = color[1:ns])
      end,

    observation = function (a,sp)
      return observation_fn(sp)
    end
    

)

#println(observation_fn((1,2,3,4),1,(1,2,3,4)))

#solvervi = QMDPSolver(verbose=true)
#policy = solve(solvervi, m1)
#observation_fn((3,4,5,6))

# function vi_estimate(mdp,s,depth)
#   n=length(s)/2
#   value_estimate=0
#   for i=1:n
#     value_estimate+=value(policy, (s[2*i-1],s[2*i]))
#   end
#   return value_estimate
# end
#adaoposolver = DESPOTSolver(bounds=(-1000, 1000))
#solvervi = QMDPSolver(verbose=true)
#adapolicy=solve(solvervi,m2)
#  rsum = 0.0
#  for (s,b,a,o,r) in stepthrough(m2, adapolicy, "s,b,a,o,r", max_steps=35)
#      println("s: $s, a: $a, o: $o")
#      global rsum += r
#  end
#  rsum
#println("Undiscounted reward was $rsum.")
#solver_A = (n_iterations=1000, depth=30, exploration_constant=0.1,estimate_value=vi_estimate) # initializes the Solver type
#planner_mcts = solve(solver_mcts, m2)
# #reward((1,2,1,2))
#ds = DisplaySimulator()
#simulate(ds,m2,adapolicy)
# planner = solve(MCTS_Greedy_Solver, m2)

# print("running simulation")
# hr = HistoryRecorder(max_steps=20)
# roller=RolloutSimulator(max_steps=20)
# h = simulate(hr, m2, planner)
#print(test((25,28,2,4),(2,3)))
#collect(eachstep(h, "s,a"))

adaoposolver = DESPOTSolver(bounds=(-500, 300))
policy=solve(adaoposolver,m2)

rnd = solve(RandomSolver(MersenneTwister(10)), m2)
hr = HistoryRecorder(max_steps=150)
roller=RolloutSimulator(max_steps=20)

h = simulate(roller, m2, policy)


println("The discounted reward from one simulation is")
@show h
println("________________________________")


q = [] # vector of the simulations to be run
push!(q, Sim(m2, policy, max_steps=30, rng=MersenneTwister(4), metadata=Dict(:policy=>"DESPOT Policy")))
push!(q, Sim(m2, rnd, max_steps=30, rng=MersenneTwister(4), metadata=Dict(:policy=>"Random")))

println("Running Monte Carlo Simulation")

data = run_parallel(q,proc_warn=false)
println(data)
println("_________________________")

ds = DisplaySimulator(max_steps = 25)
simulate(ds,m2,policy)
