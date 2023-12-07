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
statespace(3)