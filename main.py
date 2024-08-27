import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from functools import partial

def bethe_dos(q,s):
    if abs(s)<2*np.sqrt(q):
        #print("check", abs(s), 2*np.sqrt(q))
        return (q+1)/(2*np.pi)*np.sqrt(4*q-s**2)/((q+1)**2-s**2)
    else:
        #print("check2", abs(s), 2*np.sqrt(q))
        return 0

def CT__derivative(hops,s):
    def func(g):
        q=len(hops)-1
        main_term=0
        for i in range(q+1):
            main_term+=(4*hops[i]**2*g)/np.sqrt(1+4*hops[i]**2*g**2)
        return main_term-2*s 
    return func

def CT_newton(hops,s):
    def func(g):
        q=len(hops)-1
        main_term=0
        for i in range(q+1):
            main_term+=np.sqrt(1+4*hops[i]**2*g**2)
        return main_term-(q-1)-2*s*g  
    return func


def AT(x_n,x_n1,x_n2):
    return (x_n*x_n2-x_n1*x_n1)/(x_n+x_n2-2*x_n1)
        
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

#apply fixed point method and Aitken's delta-squared process
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
    ss=np.arange(-4,4,0.1)-1j*10**(-1)
    #print(ss)
    hops=[1,1,1]
    gs=np.zeros(len(ss))
    bs=np.zeros(len(ss))
    for i in range(len(ss)):
        print("step", i, "energy", ss[i])
        #gs[i]=np.imag(recursion(colored_tree_func, hops,ss[i]))
        #gs[i]=np.imag(recursion(h_function, hops,ss[i]))
        gs[i]=np.imag(newton(CT_newton(hops,ss[i]), 0.1, fprime=CT__derivative(hops,ss[i]), tol=10**(-4)))
        bs[i]=bethe_dos(len(hops)-1,np.real(ss[i]))
    #print(gs)
    ss_real=np.real(ss)
    plt.plot(ss_real,gs/np.pi, label="numerical")
    plt.plot(ss_real, bs, label="exact")
    plt.legend()
    plt.show()

main()