using QuickPOMDPs
using POMDPModelTools: Deterministic

mountaincar = QuickMDP(
    states=function state_function()
        state_list=[(-1,-1)]
        for i=range(1,30)
            for j=range(1,30)
                push!(state_list,(i,j))
            end
        end
        return state_list
    end
    actions = [1,2,3,4,5,6,7,8],
    initialstate = Deterministic((25, -18)),
    discount = 0.95,
    isterminal = s -> s[1] > 0.5
    transition=fu
)