# -*- coding: utf-8 -*-

import numpy as np
import random

DNA_SIZE = 2
DNA_SIZE_MAX = 15
POP_SIZE = 100

# Population initialization
def generate(POP_SIZE,DNA_SIZE,DNA_SIZE_MAX):
    pop_layers = np.zeros((POP_SIZE, DNA_SIZE), np.int32)
    pop_layers[:, 0] = np.random.randint(6, 11, size=(POP_SIZE,))
    pop_layers[:, 1] = np.random.randint(1, 4, size=(POP_SIZE,))
    pop = np.zeros((POP_SIZE, DNA_SIZE_MAX), np.int32)
    for i in range(POP_SIZE):
        pop_neurons = []
        pool_num=0
        for j in range(pop_layers[i,0]):
            if np.random.rand() <= 0.5:
                if j==0:
                    pop_neurons.append(np.random.randint(32,64))
                if (j>0 and j<3):
                    pop_neurons.append(np.random.randint(32,128))
                if j>=3:
                    pop_neurons.append(np.random.randint(128,512))
            else:
                for num in pop_neurons:
                    if num==1:
                        pool_num+=1
                if pool_num<=4:
                    pop_neurons.append(1)
                else:
                    if j<=3:
                        pop_neurons.append(np.random.randint(32,128))
                    if j>3:
                        pop_neurons.append(np.random.randint(128,512))
       
        num_1=0
        for k in range(4):
            if pop_neurons[k]==1:
                num_1+=1
        if num_1<2:
            idex=[]
            for l in range(3):
                if pop_neurons[l]!=1:
                    idex.append(l)
            x = random.sample(idex,2-num_1)
            for a in x:
                pop_neurons[a]=1
        else:
             pop_neurons=pop_neurons
        
        
        num=0
        for p in pop_neurons:
            if p==1:
                num+=1
        if num<3:
            idex=[]
            for l in range(len(pop_neurons)):
                if pop_neurons[l]!=1:
                    idex.append(l)
            x = random.sample(idex,3-num)
            for a in x:
                pop_neurons[a]=1
        else:
             pop_neurons=pop_neurons
             
        for k in range(pop_layers[i,1]):
            pop_neurons.append(np.random.randint(30, 50))
        pop_stack = np.hstack((pop_layers[i], pop_neurons))
        for j, gene in enumerate(pop_stack):
            pop[i][j] = gene
    return pop

def pop_generate(POP_SIZE,DNA_SIZE,DNA_SIZE_MAX):
    pop_1 = generate(POP_SIZE,DNA_SIZE,DNA_SIZE_MAX)
    pop_2 = generate(POP_SIZE,DNA_SIZE,DNA_SIZE_MAX)
    
    pop=np.array([pop_1,pop_2])
    return pop

# selection
def select(pop,a,fitness_1,fitness_2,pop_num_last,pop_num):
    pop_new_1=[]
    pop_new_2=[]
    for i in range(pop_num):
        idx = np.random.choice(np.arange(pop_num_last), size=2, replace=True)
        if abs(fitness_1[idx[0]]-fitness_1[idx[1]])<a:
            if fitness_2[idx[0]]>fitness_2[idx[1]]:
                 pop_new_1.append(pop[0][idx[1]]) 
                 pop_new_2.append(pop[1][idx[1]])
            if fitness_2[idx[0]]<fitness_2[idx[1]]:
                 pop_new_1.append(pop[0][idx[0]]) 
                 pop_new_2.append(pop[1][idx[0]]) 
            if fitness_2[idx[0]]==fitness_2[idx[1]]:
                 k = np.random.choice(idx, size=1) 
                 pop_new_1.append(pop[0][k[0]]) 
                 pop_new_2.append(pop[1][k[0]]) 
        else:
            if fitness_1[idx[0]]>fitness_1[idx[1]]:
                 pop_new_1.append(pop[0][idx[0]])
                 pop_new_2.append(pop[1][idx[0]]) 
            if fitness_1[idx[0]]<fitness_1[idx[1]]:
                 pop_new_1.append(pop[0][idx[1]]) 
                 pop_new_2.append(pop[1][idx[1]])
            if fitness_1[idx[0]]==fitness_1[idx[1]]:
                 k = np.random.choice(idx, size=1) 
                 pop_new_1.append(pop[0][k[0]])
                 pop_new_2.append(pop[1][k[0]]) 
    pop_new=np.array([pop_new_1,pop_new_2])
    return np.array(pop_new)

# crossover
def cov_coress(p1,p2,cov_1,cov_2,pool_1,pool_2,child_1,child_2,mu=20):
    number=len(cov_2)
    randList = np.random.random(number)
    beta = np.array([0.]*number)
    cov_1_code=[]
    cov_2_code=[]
    for i in cov_1[0:number]:
        cov_1_code.append(p1[i])
    for i in cov_2[0:number]:
        cov_2_code.append(p2[i])
    for i in range(number):
        if randList[i] <= 0.5:
            beta[i] = (2.0 * randList[i]) ** (1.0 / (mu + 1))
        else:
            beta[i] = (1.0 / (2.0 * (1 - randList[i]))) ** (1.0 / (mu + 1))
    
    new_p1 = 0.5 * ((1 + beta) *  cov_1_code + (1 - beta) * cov_2_code)
    new_p2 = 0.5 * ((1 - beta) *  cov_1_code + (1 + beta) * cov_2_code)
   
    for i in range(2):
        child_1[i]=p1[i]
        child_2[i]=p2[i]
    for i in cov_1[number:len(cov_1)]:
        child_1[i]=p1[i]
        
    temp_1=np.array([cov_1[0:number],new_p1])
    temp_1=temp_1.transpose()
    for i in range(number):
        child_1[round(temp_1[i][0])]=temp_1[i][1]
    temp_2=np.array([cov_2,new_p2])
    temp_2=temp_2.transpose()
    for i in range(number):
        child_2[round(temp_2[i][0])]=temp_2[i][1]
    for i in pool_1:
        child_1[i]=1
    for i in pool_2:
        child_2[i]=1
    return child_1,child_2


def dense_coress(p1,p2,dense_1,dense_2,pool_1,pool_2,child_1,child_2,mu=20):
    # number=p2[1]
    number=len(dense_2)
    randList = np.random.random(number)
    beta = np.array([0.]*number)
    for i in range(number):
        if randList[i] <= 0.5:
            beta[i] = (2.0 * randList[i]) ** (1.0 / (mu + 1))
        else:
            beta[i] = (1.0 / (2.0 * (1 - randList[i]))) ** (1.0 / (mu + 1))

    new_p1 = 0.5 * ((1 + beta) * p1[p1[0]+2:p1[0]+2+number] + (1 - beta) * p2[p2[0]+2:p2[0]+2+number])
    new_p2 = 0.5 * ((1 - beta) * p1[p1[0]+2:p1[0]+2+number] + (1 + beta) * p2[p2[0]+2:p2[0]+2+number])
    
    for i in dense_1[number:len(dense_1)]:
        child_1[i]=p1[i]
             
    temp_1=np.array([dense_1[0:number],new_p1])
    temp_1=temp_1.transpose()
    for i in range(number):
        child_1[round(temp_1[i][0])]=temp_1[i][1]
    temp_2=np.array([dense_2,new_p2])
    temp_2=temp_2.transpose()
    for i in range(number):
        child_2[round(temp_2[i][0])]=temp_2[i][1]
    return child_1,child_2


def crossover(parent, alfa, pop_num, mu=1):
    F_child = []

    for index in range(0,pop_num, 2):
        child_1=np.array([0.]*DNA_SIZE_MAX)
        child_2=np.array([0.]*DNA_SIZE_MAX)
        r = np.random.rand()
        #小于交叉率才进行交叉
        if r < alfa:
              p1=parent[index]
              p2=parent[index+1]
              pool_1 = []
              cov_1 = []
              dense_1 = []
              pool_2 = []
              cov_2 = []
              dense_2 = []
              for idex in range(p1[0]):
                  if p1[idex+2]==1:
                      pool_1.append(idex+2)
                  if p1[idex+2]!=1:
                      cov_1.append(idex+2)
              for idex in range(p1[1]):
                  dense_1.append(idex+2+p1[0])
                 
              for idex in range(p2[0]):
                  if p2[idex+2]==1:
                      pool_2.append(idex+2)
                  if p2[idex+2]!=1:
                      cov_2.append(idex+2)
              for idex in range(p2[1]):
                  dense_2.append(idex+2+p2[0])
              
              
              if len(cov_1)>=len(cov_2):
                    child=cov_coress(p1,p2,cov_1,cov_2,pool_1,pool_2,child_1,child_2,mu)
                    child_1=child[0]
                    child_2=child[1]
                    
             
              if len(cov_1)<len(cov_2):
                    child=cov_coress(p2,p1,cov_2,cov_1,pool_2,pool_1,child_2,child_1,mu)
                    child_1=child[1]
                    child_2=child[0]
                    
              
              if len(dense_1)>=len(dense_2):
                    child=dense_coress(p1,p2,dense_1,dense_2,pool_1,pool_2,child_1,child_2,mu)
                    child_1=child[0]
                    child_2=child[1]
                
              
              if len(dense_1)<len(dense_2):
                    child=dense_coress(p2,p1,dense_2,dense_1,pool_2,pool_1,child_2,child_1,mu)
                    child_1=child[1]
                    child_2=child[0]
                   
        else:
            p1=parent[index]
            p2=parent[index+1]
            child_1=p1
            child_2=p2
        F_child.append(child_1)
        F_child.append(child_2)
        

    F_child=np.array(F_child)
    for i in range(F_child.shape[0]):
          for j in range(F_child.shape[1]):
              F_child[i][j]=round(abs(F_child[i][j]))

    F_child=F_child.astype(int)
    return F_child


def pop_crossover(pop_new,alfa, pop_num, mu=1):
    
    per = np.random.permutation(pop_new.shape[1])		
    pop_new = pop_new[:, per, :]	
    parent_1 = pop_new[0]
    parent_2 = pop_new[1]
    child_1 = crossover(parent_1, alfa, pop_num, mu=20)
    child_2 = crossover(parent_2, alfa, pop_num, mu=20)
    child=np.array([child_1,child_2])
    return  child
    

# mutation
def mutate(child,pop_num, mu=20):
    for j in range(pop_num):
       
        k = np.random.choice([1,2,3,4], size=1)
        if k==1:
          
            a = list(child[j])
            idex=[]
            for i in range(a[0]+a[1]):
                idex.append(i+2)
    
            idex_d = np.random.choice(idex, size=1)
            if ((a[0]-1)!=0) and ((a[1]-1)!=0):
                if idex_d[0]<a[0]+2:
                   a[0]=a[0]-1
                else:
                  if idex_d[0]>=a[0]+2:
                     a[1]=a[1]-1
                a.pop(idex_d[0])
                a.append(0)
                child[j]=np.array(a)
            else:
                k = np.random.choice([2,3], size=1)
            num=0
            for p in a:
                if p==1:
                    num+=1
            if num<3:
                a.pop()
                a[0]=a[0]+1
                a.insert(idex_d[0], 1)
                child[j]=np.array(a)
        if k==2:
            a = list(child[j])
            idex=[]
            for i in range(a[0]+a[1]):
                   idex.append(i+2)
            idex_i = np.random.choice(idex, size=1)
            if (a[0]+a[1])<(DNA_SIZE_MAX-2):
                if idex_i[0]<=(a[0]+2):
                   a[0]=a[0]+1
                   if np.random.rand() <=0.5:
                      if idex_i[0]<5:
                         obj=np.random.randint(32,128)
                      if idex_i[0]>=5:
                         obj=np.random.randint(128,512)
                   else:
                      pool_num=0
                      for num in a:
                          if num==1:
                             pool_num+=1
                      if pool_num<=4:
                           obj=1
                      else:
                          if idex_i[0]<5:
                               obj=np.random.randint(32,128)
                          if idex_i[0]>=5:
                               obj=np.random.randint(128,512)
                else:
                  if idex_i[0]>(a[0]+2):
                      a[1]=a[1]+1
                      obj=np.random.randint(30,50)
                a.insert(idex_i[0], obj)
                a.pop()
                child[j]=np.array(a)
            else:
                k = np.random.choice([1,3], size=1)
        if k==3:
            idex=[]
            for i in range(child[j][0]+child[j][1]):
                idex.append(i+2)
            for i in idex:
                if child[j][i]==1:
                    idex.remove(i)
            idex_m = np.random.choice(idex, size=1)
            
            #产生突变因子
            u = np.random.random(1)
            if u < 0.5:
                beta = ((2.0 *u) ** (1.0 / (mu + 1)))-1
            else:
                beta = (1.0 -(2.0 * (1 - u))) ** (1.0 / (mu + 1))
            #对非池化层进行突变
            if child[j][idex_m[0]]!=1:
                child[j][idex_m[0]]=child[j][idex_m[0]]+beta[0]
        if k==4:
            idx = np.random.choice(np.arange(pop_num), size=1, replace=True)
            child[j]=child[idx]
   
    child=np.array(child)
    for i in range(child.shape[0]):
        for j in range(child.shape[1]):
            child[i][j]=round(child[i][j])
    child=child.astype(int)
    return child

def pop_mutate(child,pop_num, mu=20):
    child_1=mutate(child[0],pop_num, mu=20)
    child_2=mutate(child[1],pop_num, mu=20)
    child=np.array([child_1,child_2])
    return child


#check:
def check(child,pop_num,k):
    for j in range(pop_num):
        a = list(child[j])
        num_1=0
        for i in range(a[0]+a[1]):
              if a[i+2]>0:
                  num_1+=1
        if num_1!=(a[0]+a[1]):
             child[j]= pop_generate(1,2,15)[k][0]
                    
       
        num_2=0
        for i in range(4):
             if a[i+2]==1:
                 num_2+=1
        if num_2<2:
             child[j]= pop_generate(1,2,15)[k][0]
       
        for i in range(a[1]):
              if a[a[0]+2+i]>80:
                 child[j]= pop_generate(1,2,15)[k][0]
        
        
        for i in range(a[0]):
             if a[2+i]>512:
                 child[j]= pop_generate(1,2,15)[k][0]
       
        
        for i in range(3):
             if a[2+i]>60:
                 child[j]= pop_generate(1,2,15)[k][0]
    return child

def pop_check(child,pop_num):
    child_1=check(child[0],pop_num,0)
    child_2=check(child[1],pop_num,1)
    child=np.array([child_1,child_2])
    return child









