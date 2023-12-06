using POMDPs, QuickPOMDPs, POMDPTools
using LinearAlgebra
using DiscreteValueIteration
using BeliefUpdaters

n = 2


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
  
  reward -= out_of_bounds(s_vector)*100

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
      if (all(Points[i+1]==goal))
        t[i+1] = goal
        p = p*probability_list[idx][i]
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

#Adjust for n>1
function state_space(n=3)
    state_list = [collect(1:30) for p = 1:2*n]
    states = collect(reduce(vcat,((Iterators.product(state_list...)))))
    push!(states,(-1,-1))
    return states
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
println("Setting up MDP")
m = QuickMDP(
    states = state_space(ns) ,
    actions = action_space(ns) ,
    #initialstate = Uniform(state_space(ns)),
    #Change according to ns
    initialstate=Deterministic((25,28)),
    discount = 0.95,
    transition = function (s, a)
       transition_dict = Transition(s,a)
       transition_tuples=[]
       for key in keys(transition_dict)
        push!(transition_tuples,Tuple(Tuple(key[1])))
       end
       #state_list = state_space(ns) 
       #probabs = []
      return SparseCat(transition_tuples,values(transition_dict))
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
    isterminal = s -> s[1] == 15 && s[2] == 15,
    render = function (step)
      cx = step.s[1]
      cy = step.s[2]
      aircraft = (context(), circle(cx, cy, 0.1), fill("blue"))
      runway = (context(), line([(14.5,15),(15.5,15)]), stroke("black"))
      #goal = (context(), star(0.5, 1.0, -0.035, 5), fill("gold"), stroke("black"))
      bg = (context(), rectangle(), fill("white"))
      #ctx = context(0.7, 0.05, 0.6, 0.9, mirror=Mirror(0, 0, 0.5))
      return compose(context(), (aircraft, runway), bg)
  end

)
println("Solver Starting")
solver = ValueIterationSolver(max_iterations=10, belres=1e-3, verbose= true)
policy = solve(solver, m)
print("running simulation")
# filter = DiscreteUpdater(m);
# hr = HistoryRecorder(max_steps=5)
# h = simulate(hr, m, policy,filter)

short = RolloutSimulator(max_steps=5)
short_dr = simulate(short, m, policy)

#print(Transition([17,19],1))

