using POMDPs, QuickPOMDPs, POMDPTools
using LinearAlgebra
using DiscreteValueIteration
using POMDPSimulators
using Plots
using ElectronDisplay
using Plots
using Random


ElectronDisplay.CONFIG.single_window = true
plot([15],[15],xlims = (0,31),ylims = (0,31),seriestype=:scatter,ms = 6,label=["runway"])


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
    spread = 3
    if (d==0)
      reward+=100
    elseif  (d<8)
      reward+= 100*exp(-d/spread)
    end  

  end
  
  reward -= out_of_bounds(s_vector)*200

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
m1 = QuickMDP(
    states = statespace(ns) ,
    actions = action_space(ns) ,
    initialstate = Uniform(state_space(ns)),
    #initialstate=Deterministic((25,28)),
    discount = 0.8,
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


    reward = function (s, a)
        return reward(s)
    end,
    render = function (step)
      cx = step.s[1:2:ns*2]
      cy = step.s[2:2:ns*2]

      cx = [e for e in cx]
      cy = [e for e in cy]
      color = [:blue, :red,:green]

      

      return plot!(cx,cy,xlims = (0,31),ylims = (0,31),seriestype=:scatter,ms = 2,legend = false,color = color[1:ns])
    end
)


solvervi = ValueIterationSolver(max_iterations=3000, belres=1e-2, verbose=true)

policy = solve(solvervi, m1)


rnd = solve(RandomSolver(MersenneTwister(7)), m1)

hr = HistoryRecorder(max_steps=150)

roller=RolloutSimulator(max_steps=20)

h = simulate(roller, m1, policy)


println("The discounted reward from one simulation is")
@show h
println("________________________________")


q = [] # vector of the simulations to be run
push!(q, Sim(m1, policy, max_steps=30, rng=MersenneTwister(4), metadata=Dict(:policy=>"Value Iteration Policy")))
push!(q, Sim(m1, rnd, max_steps=30, rng=MersenneTwister(4), metadata=Dict(:policy=>"Random")))

println("Running Monte Carlo Simulation")

data = run_parallel(q,proc_warn=false)
println(data)
println("_________________________")

ds = DisplaySimulator(max_steps = 25)
simulate(ds,m1,policy)




