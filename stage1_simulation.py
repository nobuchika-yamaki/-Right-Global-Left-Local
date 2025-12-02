
# stage1_simulation.py

import numpy as np
from scipy.ndimage import gaussian_filter

def generate_environment():
    grid = np.random.normal(0,1,(20,20))
    global_bg = gaussian_filter(grid,1.0,mode='reflect')
    env = global_bg.copy()
    x = np.random.randint(1,18)
    y = np.random.randint(1,18)
    anomaly = np.random.uniform(1.5,2.0)
    env[x-1:x+2,y-1:y+2] += anomaly
    return env

def filter_grid(grid, a):
    return gaussian_filter(grid,a,mode='reflect')

def mismatch(grid, filtered):
    return np.abs(grid-filtered).sum()

def decision(mR,mL,lam):
    return (mR-mL) + lam*(abs(mR)-abs(mL))

def reaction_time(D):
    return 1/(abs(D)+1e-6)

def evaluate(ind, trials=8):
    aL,aR,lam = ind
    rts = []
    for _ in range(trials):
        env = generate_environment()
        fL = filter_grid(env,aL)
        fR = filter_grid(env,aR)
        mL = mismatch(env,fL)
        mR = mismatch(env,fR)
        D = decision(mR,mL,lam)
        rts.append(reaction_time(D))
    return -np.mean(rts)

def run_stage1(generations=20, pop_size=20):
    pop = np.column_stack([
        np.random.uniform(1.4,1.6,pop_size),
        np.random.uniform(1.4,1.6,pop_size),
        np.random.uniform(0.2,0.4,pop_size)
    ])
    for g in range(generations):
        fitness = np.array([evaluate(ind) for ind in pop])
        idx = np.argsort(fitness)[-pop_size//2:]
        parents = pop[idx]
        new = []
        for _ in range(pop_size):
            p1,p2 = parents[np.random.choice(len(parents),2,replace=True)]
            child = (p1+p2)/2
            child += np.random.normal(0,0.05,3)
            child[0] = max(child[0],0.1)
            child[1] = max(child[1],0.1)
            new.append(child)
        pop = np.array(new)
    fitness = np.array([evaluate(ind,20) for ind in pop])
    best = pop[np.argmax(fitness)]
    return best, fitness.max()

if __name__=="__main__":
    best, fit = run_stage1()
    print("Best:",best,"Fitness:",fit)
