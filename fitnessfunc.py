import numpy as np
import scipy.stats
import random
import statistics
from cec17_functions import cec17_test_func
from typing import Callable, Union, Dict, Any, List, Tuple
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool
import os


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

class highrmse():
    
    def evaluate(self, individual):
        
        phen=individual
        print(phen)
        seeds=[1521, 5082, 1271, 4503, 3835, 4753, 4081, 5143, 8388, 2679, 4584, 3560, 5653, 8796, 6304, 5516, 9419, 8549, 1076, 7263, 1212, 3026, 9229, 2558, 7551]
        sumsqerrors=[]
        results1 = []
        fno=[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
        pool1 = MyPool(29)
        for fn in range(len(fno)): 
            results1.append(pool1.apply_async(execute_run1, args=(seeds,phen,fno[fn])))
        for result in results1:
            sumsqerrors.append(result.get())
        rmserror=np.sqrt(sum(sumsqerrors)/145.0)
        pool1.close()
        return(rmserror,{})

def execute_run1(seeds,phen,fno):
        results2 = []
        sqerrors=[]
        pool2 = Pool(processes=5)
        for run in range(5):
            # Execute a single evolutionary run.
            results2.append(pool2.apply_async(execute_run2, args=(seeds[run],phen,fno)))

        for result in results2:
            sqerrors.append(result.get()**2)
        sumsqerrors=sum(sqerrors)
        pool2.close()
        return(sumsqerrors)        

        
def execute_run2(runseed,phen,fno):
        bo=[[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100]]
        if fno>=3:
           findex=fno-2
        elif fno==1:
           findex=0
        D=30
        
        opt=[100,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]
        subsiz=0.0
        subset=0
        if phen.count(':')==6:
           NP,ng,rew,sizeorsubset,alg,whichbest,mixing=phen.split(':')
           NP=int(NP)
           ng=int(ng)
           if rew=='separate':
              subsiz=float(sizeorsubset)
           else:
              subset=int(sizeorsubset)
           alg=alg.split(';') 
           '''print('NP',NP)
           print('ng',ng)
           print('rew',rew)
           print('sizeorsubset',sizeorsubset)
           print('subsiz',subsiz)
           print('subset',subset)
           print('alg',alg)
           print('whichbest',whichbest)
           print('mixing',mixing)'''
          
           
           params = get_default_params_high(dim=D,NP=NP,subsiz=subsiz, subset=subset, rew=rew, alg=alg,ng=ng,seed=runseed,whichbest=whichbest,mixing=mixing)  
           params['bounds'] = np.array([bo[findex]] * D )
           params['fno']=fno
           solution, fitness = applyhigh(**params)
           

        else:
           NP,ng,rew,sizeorsubset,strat,adapt,avgposguiding,whichbest,mixing=phen.split(':')
           NP=int(NP)
           ng=int(ng)
           
           if rew=='separate':
              subsiz=float(sizeorsubset)
           else:
              subset=int(sizeorsubset)
           strat=strat.split(';') 
           mut=[strat[i] for i in range(len(strat)) if i%2==0]
           cross=[strat[i] for i in range(len(strat)) if i%2!=0]
           '''print('NP',NP)
           print('ng',ng)
           print('rew',rew)
           print('sizeorsubset',sizeorsubset)
           print('subsiz',subsiz)
           print('subset',subset)
           print('mut',mut)
           print('cross',cross)
           print('adapt',adapt)
           print('whichbest',whichbest)
           print('mixing',mixing)
           print('averageposguiding',avgposguiding)'''
           
           params = get_default_params_low(dim=D,NP=NP,subsiz=subsiz, subset=subset, rew=rew, mut=mut,cross=cross,adapt=adapt,ng=ng,seed=runseed,whichbest=whichbest,mixing=mixing,avgposguiding=avgposguiding) 
           params['bounds'] = np.array([bo[findex]] * D )
           params['fno']=fno
           solution, fitness = applylow(**params)
       
        return(abs(opt[findex]-fitness))
        
  
def get_default_params_high(dim: int,NP: int,rew: str, subsiz:float, subset:int, alg: list,  ng: int, seed: int, whichbest: str, mixing: str):
    
    return {'max_evals': 10000 * dim, 'individual_size': dim, 'callback': None, 'population_size': NP, 'rew': rew,'subsiz':subsiz, 'subset':subset, 'alg':alg,  'ng': ng,'seed': seed, 'whichbest':whichbest, 'mixing':mixing}

def get_default_params_low(dim: int,NP: int,rew: str, subsiz:float, subset:int, adapt: str, mut: list, cross:list, ng: int, seed: int, whichbest: str, mixing: str,avgposguiding:str):
    
    return {'max_evals': 10000 * dim, 'individual_size': dim, 'callback': None, 'population_size': NP, 'rew': rew, 'subsiz':subsiz, 'subset':subset, 'adapt':adapt, 'mut':mut, 'cross':cross, 'ng': ng,'seed': seed,'whichbest':whichbest, 'mixing':mixing,'avgposguiding':avgposguiding}
  
def applyhigh(population_size: int, individual_size: int, bounds: np.ndarray, rew: str, subsiz:float, subset:int,  callback: Callable[[Dict], Any], ng: int, max_evals: int, seed: Union[int, None],alg:list, whichbest: str, mixing: str,fno:int) -> [np.ndarray, int]:

# 0. Check external parameters
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

  
    if type(ng) is not int:
        raise ValueError("ng must be a positive integer number.")
    
    np.random.seed(seed)
    ns=len(alg)
    num_evals = 0
    #params of jde
    f_jde=[[] for i in range(ns)]
    cr_jde=[[] for i in range(ns)]

    #params of jade
    p_jade=0.05
    c_jade=0.1
    u_cr_jade = [0.5 for i in range(ns)] 
    u_f_jade = [0.5 for i in range(ns)] 
    archive_size_jade=0
    archive_jade=[]
    archive_cur_len_jade=[0 for i in range(ns)]

    #params of code
    f_ar_code = np.array([1.0,1.0,0.8])
    cr_ar_code = np.array([0.1,0.9,0.2])

    #params of sade
    cr_m_sade = [np.array([0.5,0.5,0.5,0.5]) for i in range(ns)]
    p_k_sade=[np.array([0.25,0.25,0.25,0.25]) for i in range(ns)]
    LP_sade=50
    num_suc_sade=[np.zeros((LP_sade,4)) for i in range(ns)]
    num_fail_sade=[np.zeros((LP_sade,4)) for i in range(ns)]
    cr_mem_sade=[[[] for i in range(4)] for i in range(ns)]
    a_sade=[[0,0,0,0] for i in range(ns)]
    mut_sade=['rand1ind','randtobest2ind','rand2ind','currenttorand1ind']

    #params of epsde
    f_epsde=[[] for i in range(ns)]
    cr_epsde=[[] for i in range(ns)]
    mut_epsde=[[] for i in range(ns)]
    sucf_epsde=[[] for i in range(ns)]
    succr_epsde=[[] for i in range(ns)]
    sucstrat_epsde=[[] for i in range(ns)]
    indexesepsde=[[] for i in range(ns)]
    oldsize_epsde=[0 for i in range(ns)]

    #params of shade
    memory_size_shade=100
    m_cr_shade = [np.ones(memory_size_shade) * 0.5 for i in range(ns)]
    m_f_shade = [np.ones(memory_size_shade) * 0.5 for i in range(ns)]
    all_indexes_shade = list(range(memory_size_shade))
    archive_shade = []
    archive_size_shade=0
    archive_cur_len_shade=[0 for i in range(ns)]
    k_shade=[0 for i in range(ns)]

    current_generation = 0
    checkfes=10000
    funerr=[] 
    
    f_var = np.zeros(ns)
 
    
    
    
    big_population = init_population(population_size, individual_size, bounds)
    #print('big',big_population)
    fitness=apply_fitness(big_population,fno)
    num_evals += population_size
    pop_size=[]
    
    if rew=='separate':
       for i in range(ns):
           pop_size.append(int(subsiz*population_size))
       pop_size.append(population_size-sum(pop_size))
    elif rew=='rankbased':
       ranksum=sum(range(ns+1))
       for i in range(ns):
          pop_size.append(int((1/ns)*population_size))
       if sum(pop_size)!=population_size:
          r=np.random.randint(ns)
          pop_size[r]=pop_size[r]+population_size-sum(pop_size)       
       temp_pop_size=pop_size
       ascorderindices=range(ns)  
    else:
       for i in range(ns):
          pop_size.append(int((1/ns)*population_size))
       if sum(pop_size)!=population_size:
          r=np.random.randint(ns)
          pop_size[r]=pop_size[r]+population_size-sum(pop_size)     
       temp_pop_size=pop_size
       ascorderindices=range(ns) 
    
    
    #print(big_population)
    #print(fitness)
    if mixing=='completenet':
          pops,fitnesses=split(big_population,'random',pop_size,ns,fitness)
    else:
          pops,fitnesses=split(big_population,mixing,pop_size,ns,fitness)
    #print(pops)
    #print(fitnesses)
    
    #print('pops',pops)
    archive_size_jade=pop_size[0]
    archive_jade=[np.zeros((archive_size_jade,individual_size)) for i in range(ns)]
    archive_size_shade=pop_size[0]
    archive_shade=[np.zeros((archive_size_shade,individual_size)) for i in range(ns)]  
    if rew=='separate':
        chosen = np.random.randint(0, ns)
        #print('chosen',chosen)
    
        newpop = np.concatenate((pops[chosen], pops[ns]))
        pops[chosen] = newpop
        newfit=np.concatenate((fitnesses[chosen],fitnesses[ns]))
        fitnesses[chosen]=newfit
        #print('pops',pops)
        #print(pops[chosen])
        pop_size = list(map(len, pops))
        #print(pop_size)
    
   
    #print(pop_size)
    #print(pops)
    
    while num_evals <= max_evals:
        current_generation += 1
        #print(fitnesses)
        for j in range(ns):
            if alg[j]=='jde':
                 pops[j],fitnesses[j],num_evals,f_jde,cr_jde,f_var=globals()[alg[j]](pops[j],fitnesses[j],bounds,num_evals,j,f_jde,cr_jde,f_var,whichbest,fno) 
            elif alg[j]=='jade':
                 pops[j],fitnesses[j],num_evals, u_cr_jade, u_f_jade, archive_jade, archive_cur_len_jade,f_var=globals()[alg[j]](pops[j],fitnesses[j], bounds, num_evals, j, p_jade, c_jade, u_cr_jade, u_f_jade, archive_size_jade,archive_jade,archive_cur_len_jade,f_var,whichbest,fno) 
            elif alg[j]=='code':
                 pops[j],fitnesses[j],num_evals,f_var=globals()[alg[j]](pops[j],fitnesses[j],bounds,num_evals,j,f_ar_code, cr_ar_code,f_var,whichbest,fno) 
            elif alg[j]=='sade':
                 pops[j],fitnesses[j],num_evals,cr_m_sade, p_k_sade, num_suc_sade, num_fail_sade, cr_mem_sade, a_sade, f_var=globals()[alg[j]](pops[j],fitnesses[j],bounds,num_evals,j,cr_m_sade, p_k_sade, LP_sade, num_suc_sade, num_fail_sade, cr_mem_sade, a_sade, mut_sade,current_generation,f_var,whichbest,fno) 
            elif alg[j]=='epsde':
                 pops[j],fitnesses[j],num_evals,f_epsde, cr_epsde, mut_epsde, sucf_epsde, succr_epsde, sucstrat_epsde, indexesepsde, oldsize_epsde, f_var=globals()[alg[j]](pops[j],fitnesses[j],bounds,num_evals,j,f_epsde, cr_epsde, mut_epsde, sucf_epsde, succr_epsde, sucstrat_epsde, indexesepsde, oldsize_epsde, current_generation, f_var,whichbest,fno) 
            elif alg[j]=='shade':
                 pops[j],fitnesses[j],num_evals,m_cr_shade, m_f_shade, archive_shade,  archive_cur_len_shade, k_shade, f_var=globals()[alg[j]](pops[j],fitnesses[j],bounds,num_evals,j,memory_size_shade, m_cr_shade, m_f_shade, all_indexes_shade, archive_shade, archive_size_shade, archive_cur_len_shade, k_shade, f_var,whichbest,fno) 

            
            #print(pops,fitnesses)   
        #print(fitnesses)             
        #print(f_var)
        #if current_generation==5:
            #return
        
        
        if current_generation % ng == 0:
            k = [f_var[i] / (len(pops[i]) * ng) for i in range(ns)]
            #print('k',k)
            if rew=='separate':
                chosen = np.argmax(k)
                #print('chosen',chosen)
            elif rew=='rankbased':
                ascorderindices=list(np.argsort(k))
                ascorderindices.reverse()
                #print(ascorderindices)
                temp_pop_size=[]
                for i in range(subset):
                    temp_pop_size.append(int(((ns-i)/ranksum)*population_size))
                rem_size=population_size-sum(temp_pop_size)
                for i in range(ns-subset):
                    temp_pop_size.append(int((1/(ns-subset))*rem_size))
                if sum(temp_pop_size)!=population_size:
                    r=np.random.randint(ns)
                    temp_pop_size[r]=temp_pop_size[r]+population_size-sum(temp_pop_size) 
            else:
                if sum(k)!=0.0:
                    fractions=[k[i]/sum(k) for i in range(ns)]
                else:
                    fractions=[0.0 for i in range(ns)]
                
                #print(fractions)
                ascorderindices=list(np.argsort(fractions))
                ascorderindices.reverse()
                temp_pop_size=[]
                for i in range(subset):
                    temp_pop_size.append(int(fractions[ascorderindices[i]]*population_size))
                rem_size=population_size-sum(temp_pop_size)
                for i in range(ns-subset):
                    temp_pop_size.append(int((1/(ns-subset))*rem_size))
                if sum(temp_pop_size)!=population_size:
                    r=np.random.randint(ns)
                    temp_pop_size[r]=temp_pop_size[r]+population_size-sum(temp_pop_size)
                #print(temp_pop_size) 
                for i in range(ns):
                    if temp_pop_size[i]<7:
                       temp_pop_size[i]=7
                       if sum(temp_pop_size)>population_size:
                          temp_pop_size[temp_pop_size.index(max(temp_pop_size))]-=sum(temp_pop_size)-population_size
                       
            for i in range(ns):
                f_var[i]=0.0
        
        
        if rew=='separate':    
                pop_size=[]
                for i in range(ns):
                   pop_size.append(int(subsiz*population_size))
                pop_size.append(population_size-sum(pop_size))
        else:
                for i in range(ns):
                   pop_size[ascorderindices[i]]=temp_pop_size[i]
        #print(pop_size)
        #print('before topology',pops)
        #print(fitnesses)
        if mixing!='completenet':
            population=pops[0]
            fitness=fitnesses[0]
            for i in range(1,ns):
                 population=np.concatenate((population,pops[i]))
                 fitness = np.concatenate((fitness, fitnesses[i]))
            #print('Combinedpop',population)
            #print('Combinedfit',fitness)
   
            pops,fitnesses=split(population,mixing,pop_size, ns,fitness)
     
            if rew=='separate':
               newpop = np.concatenate((pops[chosen], pops[ns]))
               pops[chosen] = newpop
               newfit=np.concatenate((fitnesses[chosen],fitnesses[ns]))
               fitnesses[chosen]=newfit
               pop_size = list(map(len, pops))
        else:
            if rew=='separate':
               pop_size[chosen]+=pop_size[ns]
               #print(pop_size)
            pops,fitnesses,num_evals=topology(pops,fitnesses,ns,pop_size,bounds,num_evals,fno)
            #print('after topology')
        #print(pops)
        #print(fitnesses)             
        #print(pops[chosen])
        fitness=fitnesses[0]
        for i in range(1,ns):
            fitness = np.concatenate((fitness, fitnesses[i]))
        if callback is not None:
            callback(**(locals()))
        

    population=pops[0]
    fitness=fitnesses[0]
    for i in range(1,ns):
        population=np.concatenate((population,pops[i]))
        fitness = np.concatenate((fitness, fitnesses[i]))
    best = np.argmin(fitness)
    return population[best], fitness[best]
  
def split(population:np.ndarray,division:str, pop_size:list,ns:int,fitness:list):
    if division=='random':
        indexes = np.arange(0, len(population), 1, np.int)
        np.random.shuffle(indexes)
        splitindexes=[]
        j=0
        for i in pop_size:
           j=j+i
           splitindexes.append(j)
        indexes = np.split(indexes, splitindexes[:-1])
        pops = []
        fitnesses=[[] for i in range(len(pop_size))]
        for j in range(len(pop_size)):
                pops.append(population[indexes[j]])
                for i in indexes[j]:
                   fitnesses[j].append(fitness[i])        
        fitnesses=np.array([np.array(xi) for xi in fitnesses])
    elif division=='sortshuffle':
        indexes=fitness.argsort()
        population=population[indexes]
        fitness=fitness[indexes]
        pops=[]
        fitnesses=[[] for i in range(len(pop_size))]
        indexes=[[] for i in range(len(pop_size))]
        
        i=0
        population_size=len(population)
        
      
        while i != (population_size):
             for j in range(len(pop_size)):
                 if len(indexes[j])!=pop_size[j]:
                      indexes[j].append(i)
                      i=i+1
                      if i==(population_size):
                           break
        
        indexes=np.array([np.array(xi) for xi in indexes])
        for j in range(len(pop_size)):
                pops.append(population[indexes[j]])
                for i in indexes[j]:
                   fitnesses[j].append(fitness[i])
       
        fitnesses=np.array([np.array(xi) for xi in fitnesses])
    elif division=='distancecluster':
        population_size=len(population)
        pops=[]
        fitnesses=[[] for i in range(len(pop_size))]
        indexes=[[] for i in range(len(pop_size))]
        availableindexes=list(range(population_size))
        #print(availableindexes)
        for i in range(len(pop_size)):
            r=np.random.choice(availableindexes)
            #print(r)
            indexes[i].append(r)
            #print(indexes)
            availableindexes.remove(r)
            #print(availableindexes)
            distances=[]
            destind=[]
            for j in availableindexes:
                distances.append(dist(population[r],population[j]))
                destind.append(j)
            #print(distances)
            #print(destind)
            distances=np.array(distances)
            destind=np.array(destind)
            inddist=distances.argsort()
            destind=destind[inddist]
            #print(inddist)
            #print(destind)
            nearest=destind[:pop_size[i]-1]
            for z in nearest:
                indexes[i].append(z)
                availableindexes.remove(z)
            #print(indexes)
            #print(availableindexes)
        indexes=np.array([np.array(xi) for xi in indexes])
        
        for j in range(len(pop_size)):
           pops.append(population[indexes[j]])
           for i in indexes[j]:
               fitnesses[j].append(fitness[i])
        fitnesses=np.array([np.array(xi) for xi in fitnesses])
    return(pops,fitnesses)  

def dist(p1,p2):
   s=0
   for i in range(len(p1)):
         s=s+(p1[i]-p2[i])**2
   return(np.sqrt(s))

    
def topology(pops,fitnesses,ns,new_pop_size,bounds,num_evals,fno): 
    
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
                             
    
    cur_pop_size=[len(pops[i]) for i in range(ns)]
    index1=[[] for i in range(ns)]                              
    index2=[[] for i in range(ns)]
    temp=[]
    tempf=[]          
    for i in range(ns):
      index1[i] = list(np.argpartition(fitnesses[i], 1)[:1]) 
    for i in range(ns):
      temp.append(pops[i][index1[i]]) 
      tempf.append(fitnesses[i][index1[i]]) 
    temp=np.array(temp)  
    tempf=np.array(tempf)                            
    for i in range(ns):
      index2[i]=list(np.argpartition(fitnesses[i],-(ns-1))[-(ns-1):])
    #print(index1,index2)  
    #print(temp) 
    #print(tempf)          
    
                                      
    for i in range(ns):
        k=0
        for j in range(ns):
            if i!=j:      
                            
                foundat=np.where(np.all(temp[j] == pops[i], axis=1))    
                if len(foundat[0])!=0: 
                     #print('pops',i)                                      
                     #print(foundat[0][0])  
                     chosen = np.random.rand(len(pops[0][0]))
                     for z in range(len(chosen)):
                         pops[i][foundat[0][0]][z]=(maximum[z]-minimum[z])*chosen[z]+minimum[z]
                     calcarray=np.zeros((1,len(pops[i][foundat[0][0]])))
                     calcarray[0]=pops[i][foundat[0][0]]
                     fitnesses[i][foundat[0][0]]=apply_fitness(calcarray,fno)
                     num_evals=num_evals+1
                pops[i][index2[i][k]]=temp[j]
                fitnesses[i][index2[i][k]]=tempf[j]
                k=k+1
    #print(pops)
    #print(fitnesses)
    mempool=[]
    fitpool=[]
    
    for i in range(ns):    
      if cur_pop_size[i]>new_pop_size[i]:
           set1=set(index1[i])
           set2=set(index2[i])
           elite=set1.union(set2)
           allind=set(list(range(cur_pop_size[i])))
           nonelite=list(allind.difference(elite))
           # randomly select cur-new indiv and add to mem pool
           # delete those selected ones here itself 
           if len(nonelite)<(cur_pop_size[i]-new_pop_size[i]):
              nonelite=list(allind)
           randsel=np.random.choice(nonelite,cur_pop_size[i]-new_pop_size[i],replace=False)
           tflist=list(fitnesses)
           tplist=list(pops)
           #print(randsel)
           #print(i)
           randsel_desorder=list(np.argsort(randsel))
           randsel_desorder.reverse()
           #print(randsel_desorder)
           for j in randsel_desorder:
               mempool.append(pops[i][randsel[j]])
               fitpool.append(fitnesses[i][randsel[j]])
               #pops[i]=np.delete(pops[i],randsel[j],0)
               #fitnesses[i]=np.delete(fitnesses[i],randsel[j],0)
               tplist[i]=np.delete(tplist[i],randsel[j],0)
               tflist[i]=np.delete(tflist[i],randsel[j],0)
           pops=np.array(tplist)
           fitnesses=np.array(tflist)
    #print(mempool)
    #print(fitpool)     
    #print(pops)
    avail=list(range(len(mempool)))   
    #print(avail)      
    for i in range(ns):    
      if cur_pop_size[i]<new_pop_size[i]:        
           t=list(pops[i])
           s=list(fitnesses[i])  
           
           randsel=np.random.choice(avail,new_pop_size[i]-cur_pop_size[i],replace=False)
           #print(randsel)           
           for j in range(len(randsel)):   
                   t.append(mempool[randsel[j]])    
                   s.append(fitpool[randsel[j]])
                   avail.remove(randsel[j])                              
           #print(avail)
           listpops=list(pops)
           listfitnesses=list(fitnesses)
           listpops[i]=np.array(t)
           listfitnesses[i]=np.array(s)
           pops=np.array(listpops)
           fitnesses=np.array(listfitnesses)
           
    return(pops,fitnesses,num_evals)


def jde(population,fitnesses,bounds,num_evals,algindex,f_jde,cr_jde,f_var,whichbest,fno):
        
        size1=len(population)
        if len(f_jde[algindex])<size1:
            for i in range(len(f_jde[algindex]),size1):
                f_jde[algindex].append(0.5)
                cr_jde[algindex].append(0.9)
        
        for i in range(size1):
            rnd1=np.random.uniform(0,1)
            rnd2=np.random.uniform(0,1)
            rnd3=np.random.uniform(0,1)
            rnd4=np.random.uniform(0,1)
            if rnd2<0.1:
                f_jde[algindex][i]=0.1+rnd1*0.9
            if rnd4<0.1:
                cr_jde[algindex][i]=rnd3  
        #print(population)
        #print(f_jde,cr_jde)
        
        mutated=rand1jde(population,bounds,np.array(f_jde[algindex][:size1]).reshape(len(f_jde[algindex][:size1]), 1))
       
        crossed=bin(population, mutated, np.array(cr_jde[algindex][:size1]).reshape(len(cr_jde[algindex][:size1]), 1))

        #crossed=[[np.double(i[j]) for j in range(len(population[0]))] for i in crossed]
        c_fitness= apply_fitness(crossed,fno)
        
        winners=(c_fitness < fitnesses)
        
        #print(population)
        #print(crossed)
        #print(c_fitness)   
        population,nsucmut = selection(population, crossed, fitnesses, c_fitness)
        
        #print('new',flist)
        #print('new',population)
        num_evals += size1
        if whichbest=='fitness':
            f_var[algindex] += np.sum(fitnesses[winners] - c_fitness[winners])
        else:
            f_var[algindex] += nsucmut
        fitnesses[winners] = c_fitness[winners]
        return population,fitnesses,num_evals,f_jde,cr_jde,f_var
    
def jade(population,fitnesses,bounds,num_evals,algindex,p_jade, c_jade, u_cr_jade, u_f_jade,  archive_size_jade, archive_jade, archive_cur_len_jade,f_var,whichbest,fno):

        
        #print(u_cr,u_f)
        size1=len(population)
        f = np.empty(size1)
        cr = np.empty(size1)
        #print(population)
        #print(fitnesses)
        #print(bounds)
        #print(u_cr,u_f,c,p,archive_size,archive,archive_cur_len)
        for i in range(size1):
           temp=scipy.stats.cauchy.rvs(loc=u_f_jade[algindex], scale=0.1)
           if temp<=0:
              while(temp<=0):
                 temp=scipy.stats.cauchy.rvs(loc=u_f_jade[algindex], scale=0.1)
           if temp>=1:
              temp=1
           f[i]=temp
        cr = np.random.normal(u_cr_jade[algindex], 0.1, size1)
        cr = np.clip(cr, 0, 1)
        #print(f)
        #print(cr)
        mutated=currenttopbestjade(population, archive_jade[algindex], archive_cur_len_jade[algindex], fitnesses, f.reshape(len(f), 1), np.ones(size1) * p_jade, bounds)
        #print(mutated)
        
        crossed=bin(population, mutated, cr.reshape(len(cr), 1))
        #print(crossed)
        
        c_fitness= apply_fitness(crossed,fno)
        #print(c_fitness)
        winners=(c_fitness < fitnesses)
        #print(population)
        #print(winners)
        for i in population[winners]:   
                if(archive_cur_len_jade[algindex]>=archive_size_jade):
                    ind=np.random.randint(0,archive_size_jade)     
                    archive_jade[algindex][ind]=i
                else:
                    archive_jade[algindex][archive_cur_len_jade[algindex]]=i
                    archive_cur_len_jade[algindex]+=1
        #print(archive)
        
        #print(c_fitness1,c_fitness2,c_fitness3)
        #print(winners1,winners2,winners3)
        population,nsucmut = selection(population, crossed, fitnesses, c_fitness)
        
        
        if sum(winners) != 0 and np.sum(f[winners]) != 0:
            u_cr_jade[algindex] = (1 - c_jade) * u_cr_jade[algindex] + c_jade * np.mean(cr[winners])
            u_f_jade[algindex] = (1 - c_jade) * u_f_jade[algindex] + c_jade * (np.sum(f[winners] ** 2) / np.sum(f[winners]))
        #print(fitnesses)
        #print(c_fitness) 
        if whichbest=='fitness':  
             f_var[algindex] += np.sum(fitnesses[winners] - c_fitness[winners])
        else:
             f_var[algindex] += nsucmut
        fitnesses[winners] = c_fitness[winners]
        #print(fitnesses)
        #print(u_cr,u_f)
        num_evals=num_evals+size1
        return(population,fitnesses,num_evals, u_cr_jade, u_f_jade, archive_jade, archive_cur_len_jade,f_var)


def code(population,fitnesses,bounds,num_evals,algindex,f_ar_code, cr_ar_code,f_var,whichbest,fno):
     
        #print(f_ar,cr_ar)
        size1=len(population)
        ind=np.random.randint(0,3,(3,size1))
        f1=np.array([f_ar_code[i] for i in ind[0]])
        f2=np.array([f_ar_code[i] for i in ind[1]])
        f3=np.array([f_ar_code[i] for i in ind[2]])
        cr1=np.array([cr_ar_code[i] for i in ind[0]])
        cr2=np.array([cr_ar_code[i] for i in ind[1]])
        k=np.random.rand(size1)
        rnd1=np.random.rand(size1)
        rnd2=np.random.rand(size1)   
     
        mutated1=rand1code(population, f1.reshape(len(f1), 1),  bounds)
        #print(mutated1)
        
        crossed1=bin(population, mutated1, cr1.reshape(len(cr1), 1))
        #print(crossed1)
        
        mutated2=rand2code(population, k.reshape(len(k),1), f2.reshape(len(f2), 1),  bounds)
        #print(mutated2)
        crossed2=bin(population, mutated2, cr2.reshape(len(cr2), 1))
        #print(crossed2)
        
        mutated3=currenttorand1code(population, rnd1.reshape(len(rnd1),1), f3.reshape(len(f3), 1),  bounds)
        #print(mutated3)
        crossed3=arithmetic(population, mutated3, rnd2.reshape(len(rnd2), 1))
        #print(crossed3)
        
        c_fitness1= apply_fitness(crossed1,fno)
        c_fitness2= apply_fitness(crossed2,fno)
        c_fitness3= apply_fitness(crossed3,fno)

        crossed=np.ndarray((size1,len(population[0])))
        c_fitness=np.zeros(size1)
        for i in range(size1):
            mini=min(c_fitness1[i],c_fitness2[i],c_fitness3[i])
            if c_fitness1[i]==mini:
               crossed[i]=crossed1[i]
            elif c_fitness2[i]==mini:
               crossed[i]=crossed2[i]
            else:
               crossed[i]=crossed3[i]
            c_fitness[i]=mini
        #print('winners',crossed)
        #print('winfit',c_fitness)
        
        winners=(c_fitness <= fitnesses)
        #print(fitnesses)
        #print(winners)
        
        population,nsucmut = selection(population, crossed, fitnesses, c_fitness)
        #print(population)
        if whichbest=='fitness':
             f_var[algindex] += np.sum(fitnesses[winners] - c_fitness[winners])
        else:
             f_var[algindex] += nsucmut
        fitnesses[winners] = c_fitness[winners]
        #print(fitnesses)
        num_evals=num_evals+3*(size1)
        return(population,fitnesses,num_evals, f_var)

def sade(population,fitnesses,bounds,num_evals,algindex,cr_m_sade, p_k_sade, LP_sade, num_suc_sade, num_fail_sade, cr_mem_sade, a_sade, mut_sade,current_generation,f_var,whichbest,fno):
         
        size1=len(population)
        f=np.empty(size1)
        cr=np.zeros((4,size1))   
        if current_generation > LP_sade:
           S=np.divide(np.sum(num_suc_sade[algindex],axis=0),np.add(np.sum(num_suc_sade[algindex],axis=0),np.sum(num_fail_sade[algindex],axis=0)))+0.01
           #print(S)
           for i in range(4):
              p_k_sade[algindex][i]=S[i]/sum(S)
           #print(p_k)
           
           num_suc_sade[algindex]=np.delete(num_suc_sade[algindex],(0),axis=0)
           num_fail_sade[algindex]=np.delete(num_fail_sade[algindex],(0),axis=0) 
           num_suc_sade[algindex]=np.append(num_suc_sade[algindex],[[0.0,0.0,0.0,0.0]],axis=0) 
           num_fail_sade[algindex]=np.append(num_fail_sade[algindex],[[0.0,0.0,0.0,0.0]],axis=0)
           #print(ns,nf)
           
        for i in range(4):
           a_sade[algindex][i]=sum(p_k_sade[algindex][0:i+1])
        n=i=0
        strategy=[0 for j in range(size1)]
        r=np.random.uniform(0,1.0/size1)
        while n<size1:
            while r<=a_sade[algindex][i]:
                strategy[n]=i
                r=r+1.0/size1
                n=n+1
            i=i+1
            if i==4:
              break
        #print('strategy',strategy)
        f=np.random.normal(0.5,0.3,size1)
        #print('f',f)
        if current_generation > LP_sade:
            for k in range(4):
               if len(cr_mem_sade[algindex][k])!=0:
                   cr_m_sade[algindex][k]=statistics.median(cr_mem_sade[algindex][k])
            cr_mem_sade[algindex]=[[] for i in range(4)] 
            #print('New cr_m',cr_m)
            #print('new cr_mem',cr_mem)   
            
        for k in range(4):
           for i in range(size1):
             temp=np.random.normal(cr_m_sade[algindex][k], 0.1)
             while(temp<0 or temp>1):
                 temp=np.random.normal(cr_m_sade[algindex][k], 0.1)
             cr[k][i]=temp
        #print('cr',cr)
        
        mutated=np.zeros((size1,len(population[0])))
        crossed=np.zeros((size1,len(population[0])))
        #print(f)
        #print(cr)
        
        for i in range(size1):
           k=strategy[i]
           mutated[i]=globals()[mut_sade[k]](population, fitnesses,i, f[i], bounds)
           if k != 3:
              crossed[i]=binind(population[i], mutated[i], cr[k][i])
           else:
              crossed[i]=mutated[i]
        #print(mutated)
        #print(crossed)
        
        c_fitness= apply_fitness(crossed,fno)
        #print(c_fitness)
        num_evals += size1
        winners=(c_fitness <= fitnesses)
        #print(winners)
       
        population,indexes,nsucmut = selectionsade(population, crossed, fitnesses, c_fitness)
        if whichbest=='fitness':
            f_var[algindex] += np.sum(fitnesses[winners] - c_fitness[winners])
        else:
            f_var[algindex] += nsucmut
        fitnesses[winners] = c_fitness[winners]
        #print(population)
        #print(fitnesses)
        #print(indexes)
        for i in indexes:
            if current_generation<=LP_sade:
                 num_suc_sade[algindex][current_generation-1][strategy[i]]+=1
            else:
                 num_suc_sade[algindex][LP_sade-1][strategy[i]]+=1
            cr_mem_sade[algindex][strategy[i]].append(cr[strategy[i]][i])   
                
        for i in range(size1):
            if i not in indexes:
                if current_generation<=LP_sade:
                    num_fail_sade[algindex][current_generation-1][strategy[i]]+=1
                else:
                    num_fail_sade[algindex][LP_sade-1][strategy[i]]+=1    
        return(population,fitnesses,num_evals,cr_m_sade, p_k_sade, num_suc_sade, num_fail_sade, cr_mem_sade, a_sade,f_var)


def epsde(population,fitnesses,bounds,num_evals,algindex,f_epsde, cr_epsde, mut_epsde, sucf_epsde, succr_epsde, sucstrat_epsde, indexesepsde, oldsize_epsde, current_generation, f_var,whichbest,fno):
        
        size1=len(population)
        if len(f_epsde[algindex])<size1:
            for i in range(len(f_epsde[algindex]),size1):
                f_epsde[algindex].append(np.random.choice([0.4,0.5,0.6,0.7,0.8,0.9]))
                cr_epsde[algindex].append(np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
                mut_epsde[algindex].append(np.random.choice(['best2ind','rand1ind','currenttorand1ind'])) 
        
        
        if current_generation>1:
          for i in range(oldsize_epsde[algindex]):
            if i not in indexesepsde[algindex]:
                    if np.random.rand()<0.5 or len(sucf_epsde[algindex])==0:
                       f_epsde[algindex][i]=np.random.choice([0.4,0.5,0.6,0.7,0.8,0.9])
                       cr_epsde[algindex][i]=np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                       mut_epsde[algindex][i]=np.random.choice(['best2ind','rand1ind','currenttorand1ind'])
                    else:  
                       ind=np.random.choice([numbers for numbers in range(len(sucf_epsde[algindex]))])
                       f_epsde[algindex][i]=sucf_epsde[algindex][ind]
                       cr_epsde[algindex][i]=succr_epsde[algindex][ind]
                       mut_epsde[algindex][i]=sucstrat_epsde[algindex][ind]
        #print(len(population))
        #print(population)
        #print('f_epsde',f_epsde)
        #print('cr_epsde',cr_epsde)
        #print('mut_epsde',mut_epsde)
        sucf_epsde[algindex]=[]
        succr_epsde[algindex]=[]
        sucstrat_epsde[algindex]=[]
        mutated=np.zeros((size1,len(population[0])))
        crossed=np.zeros((size1,len(population[0])))
        for i in range(size1):
           mutated[i]=globals()[mut_epsde[algindex][i]](population, fitnesses,i, f_epsde[algindex][i], bounds)
           if mut_epsde[algindex][i]!='currenttorand1ind':
               crossed[i]=binind(population[i], mutated[i], cr_epsde[algindex][i])
           else:
               crossed[i]=mutated[i]

        c_fitness= apply_fitness(crossed,fno)
        
        winners=(c_fitness < fitnesses)
        winepsde=winners
        #print(winners)
        #print(f_epsde)
        for item in range(len(winners)):
           if winners[item]==True:
              sucf_epsde[algindex].append(f_epsde[algindex][item])
              succr_epsde[algindex].append(cr_epsde[algindex][item])
              sucstrat_epsde[algindex].append(mut_epsde[algindex][item])
        #print(sucf)
        '''if len(sucf_epsde)==0:
           f_epsde=[]
           cr_epsde=[]
           mut_epsde=[]'''
        #print(population)
        #print(crossed)
        #print(c_fitness)   
        population,indexesepsde[algindex],nsucmut = selectionsade(population, crossed, fitnesses, c_fitness)
        
        #print('new',flist)
        #print('new',population)
        num_evals += size1
        if whichbest=='fitness':
             f_var[algindex] += np.sum(fitnesses[winners] - c_fitness[winners])
        else:
             f_var[algindex] += nsucmut
        fitnesses[winners] = c_fitness[winners]
        oldsize_epsde[algindex]=size1
        return population,fitnesses,num_evals,f_epsde, cr_epsde, mut_epsde, sucf_epsde, succr_epsde, sucstrat_epsde, indexesepsde, oldsize_epsde, f_var

def shade(population,fitnesses,bounds,num_evals,algindex,memory_size_shade, m_cr_shade, m_f_shade, all_indexes_shade, archive_shade, archive_size_shade, archive_cur_len_shade, k_shade, f_var,whichbest,fno):
       
        size1=len(population)
        randnums = np.random.choice(all_indexes_shade, size1)
        cr = np.random.normal(m_cr_shade[algindex][randnums], 0.1, size1)
        cr = np.clip(cr, 0, 1)
        f = scipy.stats.cauchy.rvs(loc=m_f_shade[algindex][randnums], scale=0.1, size=size1)
        f[f > 1] = 1
        while sum(f <= 0) != 0:
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=m_f_shade[algindex][randnums[f<=0]], scale=0.1, size=sum(f <= 0))
        p = np.random.uniform(low=2/size1, high=0.2, size=size1)

        mutated=currenttopbestjade(population, archive_shade[algindex], archive_cur_len_shade[algindex], fitnesses, f.reshape(len(f), 1), p, bounds)
        #print(mutated)
        
        crossed=bin(population, mutated, cr.reshape(len(cr), 1))
        #print(crossed)
        
        c_fitness= apply_fitness(crossed,fno)
        #print(c_fitness)
        winners=(c_fitness < fitnesses)
        #print(population)
        #print(winners)
        for i in population[winners]:   
                if(archive_cur_len_shade[algindex]>=archive_size_shade):
                    ind=np.random.randint(0,archive_size_shade)     
                    archive_shade[algindex][ind]=i
                else:
                    archive_shade[algindex][archive_cur_len_shade[algindex]]=i
                    archive_cur_len_shade[algindex]+=1
        #print(archive)
        
        #print(c_fitness1,c_fitness2,c_fitness3)
        #print(winners1,winners2,winners3)
        population,indexes,nsucmut = selectionsade(population, crossed, fitnesses, c_fitness)
        
        if sum(winners) > 0:
            #if max(cr) != 0:
            weights = np.abs(fitnesses[winners] - c_fitness[winners])
            if np.sum(weights) != 0:
                  weights /= np.sum(weights)
            m_cr_shade[algindex][k_shade[algindex]] = np.sum(weights * cr[winners])

            if np.sum(weights*f[winners]) !=0:
                    m_f_shade[algindex][k_shade[algindex]]=np.sum(weights*f[winners] ** 2) / np.sum(weights*f[winners])
            

            k_shade[algindex] += 1
            if k_shade[algindex] == memory_size_shade:
                k_shade[algindex] = 0
        
        num_evals += size1
        if whichbest=='fitness':
             f_var[algindex] += np.sum(fitnesses[winners] - c_fitness[winners])
        else:
             f_var[algindex] += nsucmut
        fitnesses[winners] = c_fitness[winners]
        return population, fitnesses,num_evals, m_cr_shade, m_f_shade, archive_shade, archive_cur_len_shade, k_shade, f_var


def random_indexes(n, size, ignore=[]):
    indexes = [pos for pos in range(size) if pos not in ignore]
    assert len(indexes) >= n
    np.random.shuffle(indexes)
    if n == 1:
        return indexes[0]
    else:
        return indexes[:n]

def init_population(population_size: int, individual_size: int,
                    bounds: Union[np.ndarray, list]) -> np.ndarray:
    population=np.ndarray((population_size,individual_size))
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    chosen = np.random.rand(*population.shape)
    for i in range(population_size):
       for j in range(individual_size):
          population[i][j]=(maximum[j]-minimum[j])*chosen[i][j]+minimum[j]
    return population   
    
def apply_fitness(population: np.ndarray,fno:int) -> np.ndarray:
    nx=30
    mx=1
    func_num=fno
    fit=[0]
    flist=[]
    for individual in population:
         cec17_test_func(individual,fit,nx,mx,func_num)
         flist.append(fit[0])
    return np.array(flist)

def rand1jde(population:np.ndarray,bounds: np.ndarray, f: Union[int, float])-> np.ndarray:

    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           parents.append(a)  
    parents=np.array(parents)
    #print(parents)
    mutated = f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += population[parents[:, 0]]
    

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=minimum[j]
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=maximum[j]
    #print(mutated)
    return mutated   
   
    
def currenttopbestjade(population: np.ndarray,
                              archive:np.ndarray,
                              archive_cur_len:int,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              bounds: np.ndarray) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))
    
    p_best = np.array(p_best)
    
    # 2. We choose two random parents
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0]+archive_cur_len, ignore=[i, r1])
           a=[]
           a.append(r1)
           a.append(r2)
           parents.append(a)  
    parents=np.array(parents)
    
    arcpool=np.concatenate((population,archive),axis=0)
    #print('arcpool',arcpool.shape[0])
    #print('pop',population.shape[0])
    mutated = population + f * (population[p_best] - population)
    #print(parents[:,1])
    mutated += f * (population[parents[:, 0]] - arcpool[parents[:, 1]])
    
   

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated




def rand1code(population: np.ndarray,
                    f: Union[int, float],
                    bounds: np.ndarray) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           parents.append(a)  
    parents=np.array(parents)
    #print(parents)
    # 2. Apply the formula to each set of parents
    mutated = f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += population[parents[:, 0]]

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=min(maximum[j], (2*minimum[j]-mutated[i][j]))
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=max(minimum[j],(2*maximum[j]-mutated[i][j]))
    return mutated   



def rand2code(population: np.ndarray, k: Union[int,float],
                    f: Union[int, float],
                    bounds: np.ndarray) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 5:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           r4 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3])
           r5 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3,r4])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           a.append(r4)
           a.append(r5)
           parents.append(a)  
    parents=np.array(parents)
    #print(parents)
    # 2. Apply the formula to each set of parents
    mutated = k * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += population[parents[:, 0]]
    mutated += f * (population[parents[:, 3]] - population[parents[:, 4]])

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=min(maximum[j], (2*minimum[j]-mutated[i][j]))
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=max(minimum[j],(2*maximum[j]-mutated[i][j]))
    return mutated  


def currenttorand1code(population:  np.ndarray, rnd: Union[int,float], f: List[float], bounds: np.ndarray) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           parents.append(a)  
    parents=np.array(parents)
    #print(parents)
    # 2. Apply the formula to each set of parents
    mutated = population+ rnd * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=min(maximum[j], (2*minimum[j]-mutated[i][j]))
           if mutated[i][j]>maximum[0]:
                mutated[i][j]=max(minimum[j],(2*maximum[j]-mutated[i][j]))
    return mutated




def rand1ind(population: np.ndarray,
                    population_fitness: np.ndarray, i:int,
                    f: Union[int, float],
                    bounds: np.ndarray) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population[i]
    
    # 1. For each number, obtain 3 random integers that are not the number
        
    r1 = random_indexes(1, population.shape[0], ignore=[i])
    r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
    r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
    
    #print('rand1r1r2r3',r1,r2,r3)
    # 2. Apply the formula to each set of parents
    mutated = f * (population[r2] - population[r3])
    mutated += population[r1]
    #print(mutated)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
  
    for j in range(len(mutated)):
           if mutated[j]<minimum[0] or mutated[j]>maximum[0]:
                chosen = np.random.rand(population.shape[1])
                #print(chosen)
                for x in range(len(mutated)):
                   mutated[x]=(maximum[x]-minimum[x])*chosen[x]+minimum[x]
                return mutated
    return mutated

def rand2ind(population: np.ndarray,
                    population_fitness: np.ndarray, i:int,
                    f: Union[int, float],
                    bounds: np.ndarray) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 5:
        return population[i]
    
    # 1. For each number, obtain 3 random integers that are not the number
    
    r1 = random_indexes(1, population.shape[0], ignore=[i])
    r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
    r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
    r4 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3])
    r5 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3,r4])
           
    #print('rand2r1r2r3r4r5',r1,r2,r3,r4,r5)
    # 2. Apply the formula to each set of parents
    mutated = f * (population[r2] - population[r3])
    mutated += population[r1]
    mutated += f * (population[r4] - population[r5])

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for j in range(len(mutated)):
           if mutated[j]<minimum[0] or mutated[j]>maximum[0]:
                chosen = np.random.rand(population.shape[1])
                #print(chosen)
                for x in range(len(mutated)):
                   mutated[x]=(maximum[x]-minimum[x])*chosen[x]+minimum[x]
                return mutated
    return mutated 

def currenttorand1ind(population: np.ndarray,
                              population_fitness: np.ndarray,
                              i:int,
                              f: Union[int, float],
                              bounds: np.ndarray) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population[i]
    
    # 1. For each number, obtain 3 random integers that are not the number
    
    r1 = random_indexes(1, population.shape[0], ignore=[i])
    r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
    r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
    #print('curtorand1r1r2r3',r1,r2,r3)       
    k=np.random.rand()
    #print('k',k)
    # 2. Apply the formula to each set of parents
    mutated = population[i]+ k * (population[r1] - population[i])
    mutated += f * (population[r2] - population[r3])
    #print(mutated)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for j in range(len(mutated)):
           if mutated[j]<minimum[0] or mutated[j]>maximum[0]:
                chosen = np.random.rand(population.shape[1])
                #print(chosen)
                for x in range(len(mutated)):
                   mutated[x]=(maximum[x]-minimum[x])*chosen[x]+minimum[x]
                
                return mutated
    return mutated

def randtobest2ind(population: np.ndarray,
                              population_fitness: np.ndarray,
                              i:int,
                              f: Union[int, float],
                              bounds: np.ndarray) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 5:
        return population[i]
    bestind = np.argmin(population_fitness)
    
    # 1. For each number, obtain 3 random integers that are not the number
    
    r1 = random_indexes(1, population.shape[0], ignore=[i,bestind])
    r2 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1])
    r3 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2])
    r4 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2,r3])
    #print('randtobest2bestr1r2r3r4',bestind,r1,r2,r3,r4)
   
    # 2. Apply the formula to each set of parents
    mutated = population[i]+ f * (population[bestind] - population[i])
    mutated += f * (population[r1] - population[r2])
    mutated += f * (population[r3] - population[r4])
    
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for j in range(len(mutated)):
           if mutated[j]<minimum[0] or mutated[j]>maximum[0]:
                chosen = np.random.rand(population.shape[1])
                #print(chosen)
                for x in range(len(mutated)):
                   mutated[x]=(maximum[x]-minimum[x])*chosen[x]+minimum[x]
                return mutated
    return mutated

def best2ind(population: np.ndarray,
                    population_fitness: np.ndarray,
                    i:int,
                    f: Union[int, float],
                    bounds: np.ndarray) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 4:
        return population[i]
    
    bestind = np.argmin(population_fitness)
    # 1. For each number, obtain 3 random integers that are not the number
    
    r1 = random_indexes(1, population.shape[0], ignore=[i])
    r2 = random_indexes(1, population.shape[0], ignore=[i,r1])
    r3 = random_indexes(1, population.shape[0], ignore=[i,r1,r2])
    r4 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3])
    
    # 2. Apply the formula to each set of parents
    mutated = f * (population[r1] - population[r2])
    mutated += population[bestind]
    mutated += f * (population[r3] - population[r4])

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for j in range(len(mutated)):
           if mutated[j]<minimum[0] or mutated[j]>maximum[0]:
                chosen = np.random.rand(population.shape[1])
                #print(chosen)
                for x in range(len(mutated)):
                   mutated[x]=(maximum[x]-minimum[x])*chosen[x]+minimum[x]
                return mutated
    return mutated   


def exp(population: np.ndarray, mutated: np.ndarray,
                          cr: Union[int, float]) -> np.ndarray:
   
    #if cr is int or float:
     #   cr = np.array([cr] * len(population))
    def __exponential_crossover_1(x: np.ndarray, y: np.ndarray, cr: Union[int, float]) -> np.ndarray:
        z = x.copy()
        n = len(x)
        k = np.random.randint(0, n)
        j = k
        l = 0
        while True:
            z[j] = y[j]
            j = (j + 1) % n
            l += 1      
            if np.random.randn() > cr or l == n:
                return z
    return np.array([__exponential_crossover_1(population[i], mutated[i], cr[i]) for i in range(len(population))])

    
def arithmetic(population: np.ndarray, mutated: np.ndarray,
              rnd: Union[int, float]) -> np.ndarray:
    temp = population+ rnd * (mutated - population)
    return temp

def bin(population: np.ndarray, mutated: np.ndarray,
              cr: Union[int, float]) -> np.ndarray:
    
    chosen = np.random.rand(*population.shape)
    #print('chosen',chosen)
    j_rand = [np.random.randint(0, population.shape[1]) for i in range(population.shape[0])]
    #print('jrand',j_rand)
    
    #chosen[j_rand::population.shape[1]] = 0
    #print('chosen',chosen)
    #print('cr',cr)
    temp=np.where(chosen <= cr, mutated, population)
    #print(temp)
    for i in range(population.shape[0]):
        temp[i][j_rand[i]]=mutated[i][j_rand[i]]
    return temp
    #return np.where(chosen <= cr, mutated, population)

def binind(population: np.ndarray, mutated: np.ndarray,
              cr: Union[int, float]) -> np.ndarray:
   
    chosen = np.random.rand(len(population))
    #print('chosen',chosen)
    j_rand = np.random.randint(0, len(population))
    #print('jrand',j_rand)
    
    temp=np.where(chosen < cr, mutated, population)
    #print(temp)
    
    temp[j_rand]=mutated[j_rand]
    return temp


def selection(population: np.ndarray, new_population: np.ndarray,fitness: np.ndarray, new_fitness: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    indexes = np.where(fitness > new_fitness)[0]
    population[indexes] = new_population[indexes]
    return population,len(indexes)

def selectionsade(population: np.ndarray, new_population: np.ndarray,fitness: np.ndarray, new_fitness: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    indexes = np.where(fitness >= new_fitness)[0]
    betterindexes = np.where(fitness > new_fitness)[0]
    population[indexes] = new_population[indexes]
    return population,indexes,len(betterindexes)  
    
  

def applylow(population_size: int, individual_size: int, bounds: np.ndarray, rew: str, subsiz: float, subset:int, 
       callback: Callable[[Dict], Any], adapt: str,
          ng: int, max_evals: int, seed: Union[int, None],mut:list,cross:list, whichbest:str, mixing:str,avgposguiding:str,fno:int) -> [np.ndarray, int]:

    # 0. Check external parameters
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    
    if type(ng) is not int:
        raise ValueError("ng must be a positive integer number.")

   
    np.random.seed(seed)
    ns=len(mut)
    num_evals = 0
    mp=0.25
    memory_size_empede=6
    k_empede=[0 for i in range(ns)]
    #print(adapt)
    if adapt=='impede' or adapt=='mpede':
       p=0.04
       pb=0.04
       c=0.1
       u_cr = np.ones(ns) * 0.5
       u_f = np.ones(ns) * 0.5
    elif adapt=='empede':
       p=pb=c=0.1
       m_cr= [np.ones(memory_size_empede) * 0.5 for i in range(ns)]
       m_f= [np.ones(memory_size_empede) * 0.5 for i in range(ns)] 
    f_var = np.zeros(ns)
    fes = np.zeros(ns)
    #print(m_cr)
    #print(m_f)
    
    
    big_population = init_population(population_size, individual_size, bounds)
    #print(big_population)
    fitness=apply_fitness(big_population,fno)
    num_evals += population_size
    pop_size=[]
    if rew=='separate':
        for i in range(ns):
           pop_size.append(int(subsiz*population_size))
        pop_size.append(population_size-sum(pop_size))
    elif rew=='rankbased':
        ranksum=sum(range(ns+1))
     
        for i in range(ns):
           pop_size.append(int((1/ns)*population_size))
        if sum(pop_size)!=population_size:
           r=np.random.randint(ns)
           pop_size[r]=pop_size[r]+population_size-sum(pop_size)       
        temp_pop_size=pop_size
        ascorderindices=range(ns)
    else:
       for i in range(ns):
          pop_size.append(int((1/ns)*population_size))
       if sum(pop_size)!=population_size:
          r=np.random.randint(ns)
          pop_size[r]=pop_size[r]+population_size-sum(pop_size)     
       temp_pop_size=pop_size
       ascorderindices=range(ns) 


    if mixing=='completenet':
          pops,fitnesses=split(big_population,'random',pop_size,ns,fitness)
    else:
          pops,fitnesses=split(big_population,mixing,pop_size,ns,fitness)

    archive_size=[pop_size[0] for i in range(ns)]
    archive=[np.zeros((archive_size[i],individual_size)) for i in range(ns)]
    archive_cur_len=[0 for i in range(ns)]
    if rew=='separate':
        chosen = np.random.randint(0, ns)
        #print(chosen)
    
        newpop = np.concatenate((pops[chosen], pops[ns]))
        pops[chosen] = newpop
        newfit=np.concatenate((fitnesses[chosen],fitnesses[ns]))
        fitnesses[chosen]=newfit
        #print(pops)
        #print(pops[chosen])
        pop_size = list(map(len, pops))
        #print(pop_size)
    current_generation = 0
    #print(pop_size)
    masterindex=pop_size.index(max(pop_size))
    pbind = [i for i, x in enumerate(mut) if x == "currenttopbest"]
    
    f = []
    cr = []
    
    for j in range(ns):
        f.append(np.empty(pop_size[j]))
        cr.append(np.empty(pop_size[j]))
    #print(pops)
    #print(fitnesses)
    
    checkfes=10000
    funerr=[]
    population=pops[0]
    fitness=fitnesses[0]
    for i in range(1,ns):
        population=np.concatenate((population,pops[i]))
        fitness = np.concatenate((fitness, fitnesses[i]))
    gbest = population[np.argmin(fitness)]

    while num_evals <= max_evals:
        current_generation += 1

        if adapt=='impede' or adapt=='mpede':
          for j in range(ns):
            f[j] = scipy.stats.cauchy.rvs(loc=u_f[j], scale=0.1, size=len(pops[j]))
            f[j] = np.clip(f[j], 0, 1)

            cr[j] = np.random.normal(u_cr[j], 0.1, len(pops[j]))
            cr[j] = np.clip(cr[j], 0, 1)
        elif adapt=='empede':
            ran=np.random.randint(0,memory_size_empede)
            #print(ran)
            for j in range(ns):
               cr[j] = np.random.normal(m_cr[j][ran], 0.1, len(pops[j]))
               cr[j] = np.clip(cr[j], 0, 1)
               f[j] = scipy.stats.cauchy.rvs(loc=m_f[j][ran], scale=0.1, size=len(pops[j]))
               f[j] = np.clip(f[j], 0, 1)
               
        lbest=[]
        mutated=[]
        for j in range(ns):
            if j!=masterindex:
                lbest.append(pops[j][np.argmin(fitnesses[j])])
        #print(lbest)
        averageposition=np.mean(lbest,axis=0)
        #print(averageposition)

        for i in range(ns):
          if mut[i]=="currenttopbest":
             mutated.append(globals()[mut[i]](pops[i], archive[i], archive_cur_len[i], fitnesses[i], f[i].reshape(len(f[i]), 1), np.ones(len(pops[i])) * p, bounds,masterindex,i,averageposition,avgposguiding))
          elif mut[i]=="pbadtopbest":
             mutated.append(globals()[mut[i]](pops[i], fitnesses[i], f[i].reshape(len(f[i]), 1), np.ones(len(pops[i])) * p, np.ones(len(pops[i])) * pb, bounds,masterindex,i,averageposition,avgposguiding))
          elif mut[i]=="randtompbest1":
             mutated.append(globals()[mut[i]](pops[i], fitnesses[i], f[i].reshape(len(f[i]), 1), np.ones(len(pops[i])) * mp, bounds,masterindex,i,averageposition,avgposguiding))
          elif mut[i]=="pbadtopbestgbest":
             mutated.append(globals()[mut[i]](pops[i], fitnesses[i], f[i].reshape(len(f[i]), 1), np.ones(len(pops[i])) * p, np.ones((len(pops[i]),1)) * gbest,bounds,masterindex,i,averageposition,avgposguiding))   
          else:
             mutated.append( globals()[mut[i]](pops[i], fitnesses[i], f[i].reshape(len(f[i]), 1), bounds,masterindex,i,averageposition,avgposguiding))
        #print('m1',mutated1)
        
        #return
        #print('cr',cr)
        # 2.3 Do the crossover and calculate new fitness
        crossed=[]
        for i in range(ns):
           if cross[i]=="nil":
               crossed.append(mutated[i])
           else:
               crossed.append( globals()[cross[i]](pops[i], mutated[i], cr[i].reshape(len(cr[i]), 1)))
        c_fitness=[]
        for i in range(ns):
           c_fitness.append( apply_fitness(crossed[i],fno))
        
        for j in range(ns):
            num_evals += len(pops[j])
            fes[j] += len(pops[j])
        if "randtompbest1" in mut: 
            mp=((0.05-0.25)/max_evals)*num_evals+0.25
            #print(mp)
        # 2.4 Do the selection and update control parameters
        winners=[]
        for i in range(ns):
           winners.append(c_fitness[i] < fitnesses[i])
        
        for x in pbind:
           for i in pops[x][winners[x]]:   
                if(archive_cur_len[x]>=archive_size[x]):
                    ind=np.random.randint(0,archive_size[x])     
                    archive[x][ind]=i
                else:
                    archive[x][archive_cur_len[x]]=i
                    archive_cur_len[x]+=1
       
        #print(c_fitness1,c_fitness2,c_fitness3)
        #print(winners1,winners2,winners3)
        for i in range(ns):        
            pops[i],nsucmut = selection(pops[i], crossed[i], fitnesses[i], c_fitness[i])
            if whichbest=='fitness':
                f_var[i] += np.sum(fitnesses[i][winners[i]] - c_fitness[i][winners[i]])
            else: 
                f_var[i] += nsucmut
            if adapt=='impede':
               if i in pbind:
                 if sum(winners[i]) != 0:
                    weights = np.abs(fitnesses[i][winners[i]] - c_fitness[i][winners[i]])
                    if np.sum(weights) != 0:
                      weights /= np.sum(weights)
                    if np.sum(weights*cr[i][winners[i]]) !=0 :  
                      u_cr[i]=(1-c) * u_cr[i]+ c* (np.sum(weights*cr[i][winners[i]] ** 2) / np.sum(weights*cr[i][winners[i]]))
                    if np.sum(weights*f[i][winners[i]]) !=0:
                      u_f[i]=(1-c) * u_f[i]+ c*  (np.sum(weights*f[i][winners[i]] ** 2) / np.sum(weights*f[i][winners[i]]))
               else: 
                  if sum(winners[i]) != 0 and np.sum(f[i][winners[i]]) != 0:
                    u_cr[i] = (1 - c) * u_cr[i] + c * np.mean(cr[i][winners[i]])
                    u_f[i] = (1 - c) * u_f[i] + c * (np.sum(f[i][winners[i]] ** 2) / np.sum(f[i][winners[i]]))
            elif adapt=='mpede':
                  if sum(winners[i]) != 0 and np.sum(f[i][winners[i]]) != 0:
                    u_cr[i] = (1 - c) * u_cr[i] + c * np.mean(cr[i][winners[i]])
                    u_f[i] = (1 - c) * u_f[i] + c * (np.sum(f[i][winners[i]] ** 2) / np.sum(f[i][winners[i]]))
            elif adapt=='empede':
                  if sum(winners[i]) != 0:
                    weights = np.abs(fitnesses[i][winners[i]] - c_fitness[i][winners[i]])
                    if np.sum(weights) != 0:
                      weights /= np.sum(weights)
                    if np.sum(weights*cr[i][winners[i]]) !=0 :  
                      m_cr[i][k_empede[i]]= np.sum(weights*cr[i][winners[i]] ** 2) / np.sum(weights*cr[i][winners[i]])
                    if np.sum(weights*f[i][winners[i]]) !=0:
                      m_f[i][k_empede[i]]=np.sum(weights*f[i][winners[i]] ** 2) / np.sum(weights*f[i][winners[i]])
                    k_empede[i]=k_empede[i]+1
                    if k_empede[i]==memory_size_empede:
                      k_empede[i]=0
            #print(m_cr)
            #print(m_f)
            
            fitnesses[i][winners[i]] = c_fitness[i][winners[i]]
            
        
        
       
        if current_generation % ng == 0:
            k = [f_var[i] / (len(pops[i]) * ng) for i in range(ns)]
            #print('k',k)
            if rew=='separate':
                chosen = np.argmax(k)
                masterindex=chosen
                #print('chosen',chosen)
            elif rew=='rankbased':
                ascorderindices=list(np.argsort(k))
                ascorderindices.reverse()
                #print(ascorderindices)
                masterindex=ascorderindices[0]
                temp_pop_size=[]
                for i in range(subset):
                   temp_pop_size.append(int(((ns-i)/ranksum)*population_size))
                rem_size=population_size-sum(temp_pop_size)
                for i in range(ns-subset):
                   temp_pop_size.append(int((1/(ns-subset))*rem_size))
                if sum(temp_pop_size)!=population_size:
                   r=np.random.randint(ns)
                   temp_pop_size[r]=temp_pop_size[r]+population_size-sum(temp_pop_size) 
            else:
                if sum(k)!=0.0:
                    fractions=[k[i]/sum(k) for i in range(ns)]
                else:
                    fractions=[0.0 for i in range(ns)]
                #print(fractions)
                ascorderindices=list(np.argsort(fractions))
                ascorderindices.reverse()
                masterindex=ascorderindices[0]
                temp_pop_size=[]
                for i in range(subset):
                    temp_pop_size.append(int(fractions[ascorderindices[i]]*population_size))
                rem_size=population_size-sum(temp_pop_size)
                for i in range(ns-subset):
                    temp_pop_size.append(int((1/(ns-subset))*rem_size))
                if sum(temp_pop_size)!=population_size:
                    r=np.random.randint(ns)
                    temp_pop_size[r]=temp_pop_size[r]+population_size-sum(temp_pop_size)
                #print(temp_pop_size) 
                for i in range(ns):
                    if temp_pop_size[i]<7:
                       temp_pop_size[i]=7
                       if sum(temp_pop_size)>population_size:
                          temp_pop_size[temp_pop_size.index(max(temp_pop_size))]-=sum(temp_pop_size)-population_size
                    
            for i in range(ns):
                f_var[i]=0.0

        

        if rew=='separate':
           pop_size=[]
           for i in range(ns):
               pop_size.append(int(subsiz*population_size))
           pop_size.append(population_size-sum(pop_size))
           
        else:
           for i in range(ns):
               pop_size[ascorderindices[i]]=temp_pop_size[i]
        #print(pop_size)
        if mixing!='completenet':
           population=pops[0]
           fitness=fitnesses[0]
           for i in range(1,ns):
             population=np.concatenate((population,pops[i]))
             fitness = np.concatenate((fitness, fitnesses[i]))
           pops,fitnesses=split(population,mixing,pop_size, ns,fitness)
           #print(indexes)
           if rew=='separate':
               newpop = np.concatenate((pops[chosen], pops[ns]))
               pops[chosen] = newpop
               newfit=np.concatenate((fitnesses[chosen],fitnesses[ns]))
               fitnesses[chosen]=newfit
               pop_size = list(map(len, pops))
        else:
            if rew=='separate':
               pop_size[chosen]+=pop_size[ns]
               #print(pop_size)
            pops,fitnesses,num_evals=topology(pops,fitnesses,ns,pop_size,bounds,num_evals,fno)

        #print(pop_size)
        f=[]
        cr=[]
        
        for j in range(ns):
             f.append(np.empty(len(pops[j])))
             cr.append(np.empty(len(pops[j])))
        population=pops[0]
        fitness=fitnesses[0]
        for i in range(1,ns):
            population=np.concatenate((population,pops[i]))
            fitness = np.concatenate((fitness, fitnesses[i]))        
        gbest = population[np.argmin(fitness)]
        #print(pops)
        #print(fitnesses)
        
        if callback is not None:
           callback(**(locals()))
        

    
    best = np.argmin(fitness)
    return population[best], fitness[best]

  
def currenttopbest(population: np.ndarray,
                              archive:np.ndarray,
                              archive_cur_len:int,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population

    # 1. We find the best parent
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))
    
    p_best = np.array(p_best)
    
    # 2. We choose two random parents
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0]+archive_cur_len, ignore=[i, r1])
           a=[]
           a.append(r1)
           a.append(r2)
           parents.append(a)  
    parents=np.array(parents)
    
    arcpool=np.concatenate((population,archive),axis=0)
    #print('arcpool',arcpool.shape[0])
    #print('pop',population.shape[0])
    mutated = population + f * (population[p_best] - population)
    #print(parents[:,1])
    mutated += f * (population[parents[:, 0]] - arcpool[parents[:, 1]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)

    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated

def pbadtopbest(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              pb: Union[float, np.ndarray, int],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) < 3:
        return population
   
    
    p_best = []
    p_bad = []
    
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))
    for p_i in pb:
        bad_index = np.argsort(population_fitness)[len(population)-max(2, int(round(p_i*len(population)))):]
        p_bad.append(np.random.choice(bad_index))
    p_best = np.array(p_best)
    p_bad=np.array(p_bad)
 
    mutated = population + f * (population[p_best] - population[p_bad])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
      
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated

def rand1(population: np.ndarray,
                    population_fitness: np.ndarray,
                    f: Union[int, float],
                    bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += population[parents[:, 0]]
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated   

def rand2(population: np.ndarray,
                    population_fitness: np.ndarray,
                    f: Union[int, float],
                    bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 5:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           r4 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3])
           r5 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3,r4])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           a.append(r4)
           a.append(r5)
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += population[parents[:, 0]]
    mutated += f * (population[parents[:, 3]] - population[parents[:, 4]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated   

def best1(population: np.ndarray,
                    population_fitness: np.ndarray,
                    f: Union[int, float],
                    bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population
    bestind = np.argmin(population_fitness)
    
    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i,bestind])
           r2 = random_indexes(1, population.shape[0], ignore=[i, bestind,r1])

           a=[]
           a.append(bestind)
           a.append(r1)
           a.append(r2)
           
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += population[parents[:, 0]]
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated   
    
def best2(population: np.ndarray,
                    population_fitness: np.ndarray,
                    f: Union[int, float],
                    bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
    
    # If there's not enough population we return it without mutating
    if len(population) <= 5:
        return population
    
    bestind = np.argmin(population_fitness)
    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i,bestind])
           r2 = random_indexes(1, population.shape[0], ignore=[i, bestind,r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, bestind,r1,r2])
           r4 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2,r3])
           
           a=[]
           a.append(bestind)
           a.append(r1)
           a.append(r2)
           a.append(r3)
           a.append(r4)
           
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += population[parents[:, 0]]
    mutated += f * (population[parents[:, 3]] - population[parents[:, 4]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated   

def currenttorand1(population: np.ndarray,
                              
                              population_fitness: np.ndarray,
                              
                              f: List[float],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           parents.append(a)  
    parents=np.array(parents)
   
    # 2. Apply the formula to each set of parents
    mutated = population+ f * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated

def currenttorand2(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 5:
        return population

    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           r4 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3])
           r5 = random_indexes(1, population.shape[0], ignore=[i, r1,r2,r3,r4])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           a.append(r4)
           a.append(r5)
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = population+ f * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += f * (population[parents[:, 3]] - population[parents[:, 4]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated
def currenttobest1(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 3:
        return population
    
    bestind = np.argmin(population_fitness)
    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i,bestind])
           r2 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1])
           a=[]
           a.append(bestind)
           a.append(r1)
           a.append(r2)
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = population+ f * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated

def currenttobest2(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 5:
        return population
    
    bestind = np.argmin(population_fitness)
    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i,bestind])
           r2 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2])
           r4 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2,r3])
           a=[]
           a.append(bestind)
           a.append(r1)
           a.append(r2)
           a.append(r3)
           a.append(r4)
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = population+ f * (population[parents[:, 0]] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])
    mutated += f * (population[parents[:, 3]] - population[parents[:, 4]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated

def randtobest1(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 4:
        return population
    bestind = np.argmin(population_fitness)
    
    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i,bestind])
           r2 = random_indexes(1, population.shape[0], ignore=[i,bestind,r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2])
           a=[]
           a.append(bestind)
           a.append(r1)
           a.append(r2)
           a.append(r3)
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = population[parents[:, 1]]+ f * (population[parents[:, 0]] - population[parents[:, 1]])
    mutated += f * (population[parents[:, 2]] - population[parents[:, 3]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated

def randtobest2(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) <= 6:
        return population
    bestind = np.argmin(population_fitness)
    
    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i,bestind])
           r2 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2])
           r4 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2,r3])
           r5 = random_indexes(1, population.shape[0], ignore=[i,bestind, r1,r2,r3,r4])
           a=[]
           a.append(bestind)
           a.append(r1)
           a.append(r2)
           a.append(r3)
           a.append(r4)
           a.append(r5)
           parents.append(a)  
    parents=np.array(parents)
    
    # 2. Apply the formula to each set of parents
    mutated = population[parents[:, 1]]+ f * (population[parents[:, 0]] - population[parents[:, 1]])
    mutated += f * (population[parents[:, 2]] - population[parents[:, 3]])
    mutated += f * (population[parents[:, 4]] - population[parents[:, 5]])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    return mutated

def randtompbest1(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              mp: Union[float, np.ndarray, int],
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) < 4:
        return population
    #print(mp)
    #print(population_fitness)
    mp_best = []
    for p_i in mp:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        mp_best.append(np.random.choice(best_index))
    #print(mp_best)
    mp_best = np.array(mp_best)
    # 1. For each number, obtain 3 random integers that are not the number
    parents=[]
    for i in range(population.shape[0]):
           r1 = random_indexes(1, population.shape[0], ignore=[i])
           r2 = random_indexes(1, population.shape[0], ignore=[i,r1])
           r3 = random_indexes(1, population.shape[0], ignore=[i, r1,r2])
           a=[]
           a.append(r1)
           a.append(r2)
           a.append(r3)
           parents.append(a)  
    parents=np.array(parents)
    #print(parents)
    #print(population)
    # 2. Apply the formula to each set of parents
    mutated = population[parents[:, 0]]+ f * (population[mp_best] - population)
    mutated += f * (population[parents[:, 1]] - population[parents[:, 2]])
    #print(f)
    #print(mutated)
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    #print(mutated)
    return mutated

def pbadtopbestgbest(population: np.ndarray,
                              population_fitness: np.ndarray,
                              f: List[float],
                              p: Union[float, np.ndarray, int],
                              recgbest: np.ndarray,
                              bounds: np.ndarray,masterindex:int,popind:int,averageposition:np.ndarray,avgposguiding:str) -> np.ndarray:
   
    # If there's not enough population we return it without mutating
    if len(population) < 3:
        return population
    
    #print(population_fitness)
    
    p_best = []
    for p_i in p:
        best_index = np.argsort(population_fitness)[:max(2, int(round(p_i*len(population))))]
        p_best.append(np.random.choice(best_index))
    
    p_best = np.array(p_best)
    
    p_bad = []
    for p_i in p:
        bad_index = np.argsort(population_fitness)[len(population)-max(2, int(round(p_i*len(population)))):]
        p_bad.append(np.random.choice(bad_index))
    
    p_bad=np.array(p_bad)

    rnd = np.random.rand(*population.shape)
    #print(population)
    #print(population_fitness)
    #print(p_best)
    #print(p_bad)
    #print(rnd) 
    #print(population)
    # 2. Apply the formula to each set of parents
    mutated = population+ f * (population[p_best] - population[p_bad])
    mutated += rnd * (recgbest- population[p_best])
    if avgposguiding=='yes':
     if popind==masterindex:
       #print('True')
       delta=np.ones((len(population),1))*0.04
       
       xbar=np.ones((len(population),1))*averageposition
       mutated += delta * (xbar-population)
    #print(f)
    #print(mutated)
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    for i in range(mutated.shape[0]):
       for j in range(mutated.shape[1]):
           if mutated[i][j]<minimum[j]:
                mutated[i][j]=(minimum[j]+population[i][j])/2.0
           if mutated[i][j]>maximum[j]:
                mutated[i][j]=(maximum[j]+population[i][j])/2.0
    #print(mutated)
    return mutated



if __name__ == "__main__":
    import core.grammar as grammar
    import core.sge
    experience_name = "HighRMSE/"
    grammar = grammar.Grammar("grammars/highensemble.txt", 6, 17)
    evaluation_function = highrmse() 
    core.sge.evolutionary_algorithm(grammar = grammar, eval_func=evaluation_function, exp_name=experience_name)
    
