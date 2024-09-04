import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, root
from functools import partial


"One-variable functional equations for Greens functions"

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

def quadratic_function_bethe(q,s):
    def func(g):
        return (q+1)**2*(1+4*g**2)-((q-1)+2*s*g)**2
    return func  

def quartic_function_bethe(a,b,s):
    def func(g):
        return 64*(1+4*a**2*g**2)*(1+4*b**2*g**2)-( (2+2*s*g)**2 -4*((1+4*a**2*g**2)+(1+4*b**2*g**2)) )**2
    return func  

"TEST FUNCTIONS"
def AT(x_n,x_n1,x_n2):
    return (x_n*x_n2-x_n1*x_n1)/(x_n+x_n2-2*x_n1)

def homotopy_from_bethe(a, n_steps):
    a_hom=np.zeros((n_steps, len(a)))
    for i in range(n_steps):
        a_hom[i]=1+i*(a[i]-1)/n_steps
    return a_hom

def gs_vector_function(s, a):
    def func(g0):
        #print("g0", g0)
        N=len(a)
        g=g0[:N]+1j*g0[N:]
        #print("g",g)
        g_sum=np.sum(g)
        f_array=np.zeros(N, dtype=complex)
        for i in range(N):
            f_array[i]=g[i]*(s-g_sum+g[i])-1
            #print(f_array)
        output=np.zeros(2*N)
        output[:N]=np.real(f_array)
        output[N:]=np.imag(f_array)

        #print("result", output)
        return output
    
    return func

     
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
    ss=np.arange(-6,6,0.1)-1j*10**(-5)
    #print(ss)
    hops=[1,1,1]
    N=len(hops)
    gs=np.zeros(len(ss))
    bs=np.zeros(len(ss))
    a=1
    b=1
    g0=[0,0,0,1,1,1]

    for i in range(len(ss)):
        print("step", i, "energy", ss[i])
        #various approaches
        #gs[i]=np.imag(recursion(colored_tree_func, hops,ss[i]))
        #gs[i]=np.imag(recursion(h_function, hops,ss[i]))
        #gs[i]=np.imag(newton(quadratic_function_bethe(2,ss[i]), 0.1, tol=10**(-4)))
        #gs[i]=np.imag(newton(quartic_function_bethe(a,b,ss[i]), 0.1+0.1*1j, tol=10**(-4)))
        #gs[i]=np.imag(newton(CT_newton(hops,ss[i]), 0.1, fprime=CT__derivative(hops,ss[i]), tol=10**(-4),disp=False))
        
        #newton method for the system of equations
        bs[i]=bethe_dos(2,np.real(ss[i]))
        sol=root(gs_vector_function(ss[i], hops),g0, method="hybr").x
        re_part=np.sum(sol[:N])
        im_part=np.sum(sol[N:])
        print(re_part, im_part)

        gs[i]=im_part/(np.pi*((np.real(ss[i])-re_part)**2+im_part**2))
        
    #print(gs)
    ss_real=np.real(ss)
    plt.plot(ss_real,gs, label="numerical")
    plt.plot(ss_real, bs, label="exact")
    plt.legend()
    plt.show()

main()