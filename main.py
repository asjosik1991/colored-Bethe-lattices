import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from functools import partial

def h_function(hops,h,s):
    q=len(hops)-1
    print(q)
    main_term=0
    for i in range(q+1):
        main_term+=np.sqrt(h**2+4*hops[i]**2*s**2)
        print(main_term)
    return (main_term-2)/(q-1)   

def colored_tree_func(hops,g,s):
    q=len(hops)-1
    print(q)
    main_term=0
    for i in range(q+1):
        main_term+=np.sqrt(1+4*hops[i]**2*g**2)
        print(main_term)
    return (main_term-(q-1))/(2*s)

def colored_tree_func_newton(hops,s,g):
    q=len(hops)-1
    print(q)
    main_term=0
    for i in range(q+1):
        main_term+=np.sqrt(1+4*hops[i]**2*g**2)
        print(main_term)
    return main_term-2*s*g-(q-1)

def AT(x_n,x_n1,x_n2):
    return (x_n*x_n2-x_n1*x_n1)/(x_n+x_n2-2*x_n1)

def recursion(func, hops,s):
    #print("s",s)
    g_n=1j*0.5
    g_n1=func(hops,g_n,s)
    g_n2=func(hops,g_n1,s)
    g=AT(g_n,g_n1,g_n2)
    
    error=0.1
    step=0
    while abs(error)>10**(-4):
        g_n=g_n1
        g_n1=g_n2
        g_n2=func(hops,g_n1,s)
        next_g=AT(g_n,g_n1,g_n2)
        error=next_g-g
        #print("step", step, "g", g, "g_next", next_g, "error", abs(error))
        g=next_g
        step+=1
    return g

def main():
    ss=np.arange(-3,3,0.1)+1j*10**(-4)
    #print(ss)
    hops=[1,1,1]
    gs=np.zeros(len(ss))
    for i in range(len(ss)):
        #gs[i]=np.imag(recursion(colored_tree_func, hops,ss[i]))
        #gs[i]=np.imag(recursion(h_function, hops,ss[i]))
        gs[i]=np.imag(newton(partial(colored_tree_func_newton, hops=hops,s=ss[i]), 1+1j*0.1))

    
    #print(gs)
    ss_real=np.real(ss)
    plt.plot(ss_real,gs)

main()