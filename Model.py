# pylint: disable = import-error

# MESA modules
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner


# Other packages

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Income parameters 
gdp = 1*10**6 # GDP
δ_long = 0.95 # Inertia for long-term GDP tracking index
δ_short = 0.5 # Inertia for short-term GDP tracking index
ζ = 0.025 # Income inertia parameter2

# Length of different loans for the bank
L_xs = {'green':4,'efficiency':4,'quality':4}

# Consumer side parameters

# Supply side parameters
δ = 0.3 # Share of dividends
τ = 0.8 # Market-share threshold for firm exit
τ_2 = 0 # Market-share threshold
minAge = 12 # Minimum age of firms exit
Pr_new = 1/12 # Probability of new firm entrance 

class Firm(Agent):
    "An agent representing the different firms"

    def __init__(self, unique_id, model, scen, imit):

        super().__init__(unique_id, model)
        # randomly generate efficiency and then calculate the price. 
        self.e_f = np.random.uniform(scen['e_f_r'][0],scen['e_f_r'][1])  
        self.p_f = model.p_hat + np.log(model.M_e/self.e_f - 1)/model.γ_e # (1)
        self.b_f = np.random.uniform(scen['b_f_r'][0],scen['b_f_r'][1])  
        self.g_f = np.random.uniform(scen['g_f_r'][0],scen['g_f_r'][1])  

        # usefull to have them also here 
        # in order not to pass the whole model at every function
        self.λ_e = model.λ_e; self.λ_b = model.λ_b; self.λ_g = model.λ_g

        self.I_f = (self.e_f**model.λ_e)*(self.b_f**model.λ_b)*(self.g_f**model.λ_g) # (2)
        
        # just adding rest of variables them as variables
        self.ms_f = 0 # (3) market share 
        self.r_f = 0 # (4) revenue
        self.c_f = 0 # (5) cost 
        self.f_c_f = 0 # (6) fixed costs 
        self.π_f = 0 # (7) firm's profit
        self.s_f = 0 # (8) firm's wealth
        self.p_r_imit = 0 # (9) probability that this firm will be imitated

        # initialization of investment probabilities project 

        if imit:
            self.p_i_e = np.random.uniform() ;self.p_i_b = np.random.uniform()
            self.p_i_g = np.random.uniform()
            temp_sum = self.p_i_e + self.p_i_b + self.p_i_g
            self.p_i_e = self.p_i_e / temp_sum ;self.p_i_b = self.p_i_b / temp_sum
            self.p_i_g = self.p_i_g / temp_sum
        else:
            self.p_i_e = np.random.uniform(scen['p_i_e'][0],scen['p_i_e'][1])
            self.p_i_b = np.random.uniform(scen['p_i_b'][0],scen['p_i_b'][1])
            self.p_i_g = np.random.uniform(scen['p_i_g'][0],scen['p_i_g'][1])
            temp_sum = self.p_i_e + self.p_i_b + self.p_i_g
            self.p_i_e = self.p_i_e / temp_sum ;self.p_i_b = self.p_i_b / temp_sum
            self.p_i_g = self.p_i_g / temp_sum
        
        # probabilities of innovation project success
        self.P_r_x_succ = {"green":1, "efficiency":1, "quality":1}

        # attribute denoting whether the firms is engaged in an innovation project
        self.act_proj = False
        self.p_success = 0 
        self.innov = 'none'

        # value describing the current loan duration 
        self.loan_dur = 0; self.step_2 = 0; self.age = -1

    def update_agent_a(self, total_I_f, income,m_α,m_μ,m_Φ,m_ψ,m_δ):
        self.age += 1
        self.ms_f = self.I_f**m_α/total_I_f # (3)         
        self.r_f = self.ms_f*income # (4) revenues
        self.c_f = (self.p_f / (1 + m_μ)) # (5) variable costs    
        if self.age == 0:
            self.f_c_f = m_Φ*self.r_f
            self.π_f = (1 - m_Φ - 1/(1+m_μ)) * self.r_f 
            self.s_f = (1 - m_Φ - 1/(1+m_μ)) * self.r_f / δ 
        else:
            self.f_c_f = m_ψ*self.f_c_f + (1-m_ψ)*m_Φ*self.r_f # (6) fixed costs
            self.π_f = (self.p_f - self.c_f)*(self.r_f/self.p_f)-self.f_c_f # (7) profits
            self.s_f = (1-m_δ)*self.s_f + self.π_f # (8) wealth
    
    def update_agent_b(self,total_ms_f,m_η):
        self.p_r_imit = self.ms_f**m_η/total_ms_f # (9)
    def step(self):        
        self.I_f = (self.e_f**self.λ_e)*(self.b_f**self.λ_b)*(self.g_f**self.λ_g) # (2)
        self.step_2 += 1 
        
class EconomyModel(Model):
    """A model with some number of agents."""
    
    # initialization of the model
    def __init__(self ,seed, steps, dyn_firms, scenario, shock, con_pref, firm_scen,imit_scen,model_params):
        
        np.random.seed(seed)
        
        # here you define the attribute number of agents
        self.num_agents = sum([a['num'] for a in firm_scen])
        self.num_steps = steps
        self.total_I_f = 0 ; self.total_ms_f = 0
        self.YT = gdp; self.Y = gdp; self.MA_y_L = gdp; self.MA_y_S = gdp
        self.GT = 1
        self.Θ_g = 0; self.Θ_b = 0; self.Θ_c = 0; self.Θ_growth = 0
        self.current_growth = 0 # remove it from here
        self.total_diff_firms = 0 # remove it from here       
        self.dyn_firms = dyn_firms
        self.λ_e = scenario['λ_e'] ; self.λ_b = scenario['λ_b'] ; self.λ_g = scenario['λ_g']
        self.shock = shock # comment the parts referring to the shock
        self.pp = scenario["pp"]
        self.λ_g_b = (self.λ_g + self.pp/100)/(self.λ_e+self.λ_b+self.λ_g+self.pp/100)-self.λ_g
        self.λ_start = scenario["λ_start"]
        self.λ_dur = scenario["λ_dur"]
        self.λ_off = scenario["λ_off"]
        
        self.ω_g = model_params['ω_g'] ;self.ω_b = model_params['ω_b'] 
        self.ω_c = model_params['ω_c']
        self.G_min = model_params['G_min']; self.G_max = model_params['G_max'] 
        self.M_e = model_params['M_e']; self.γ_e = model_params['γ_e']
        self.p_hat = model_params['p_hat']
        self.α = model_params['α'] ; self.μ = model_params['μ']
        self.ψ = model_params['ψ'] ; self.Φ = model_params['Φ']
        self.δ = model_params['δ'] 
        self.Pr_new = model_params['Pr_new'] 
        self.η = model_params['η'] ; self.Ω = model_params['Ω']
        self.f_a = model_params['f_a'] 

        self.running = True

        # Metrics
        self.MSWA_g = 0; self.MSWA_b = 0; self.MSWA_e = 0
        
        # consumer preferences

        self.con = con_pref

        # Self.imit_scen is not used somewhere yet,
        # it could be used to define the characteristics of the entering firm 
        # without making them imitators
        self.imit_scen = imit_scen

        # Probability that commercial bank would finance a green loan
        self.P_r_l = {"green":scenario['P_r_l_g'], "quality":1, "efficiency":1}

        # Influence of SIB on probability of investments 
        self.σ = {"green":scenario['σ_g'], "quality":0, "efficiency":0}

        # Dictionary containing the lists of firms 
        # investing at each characteristic at every moment  
        self.invest = {'green':[],'efficiency':[],'quality':[]}

        self.schedule = RandomActivation(self)
        # the collection takes place after the step function 
        self.dc = DataCollector(model_reporters = {"agents": lambda m: m.schedule.get_agent_count(),
                                                   "Total_I_fs": lambda m: m.total_I_f,
                                                   "PotentialGDP": lambda m: m.YT,
                                                   "LongTermGDP": lambda m: m.MA_y_L,
                                                   "shortTermGDP": lambda m: m.MA_y_S,
                                                   "ActualGDP": lambda m: m.Y,
                                                   "GDPGrowth": lambda m: m.GT,
                                                   "InvestmentGrowth": lambda m: m.Θ_growth, 
                                                   "λ_e" : lambda m: m.λ_e,
                                                   "λ_b" : lambda m: m.λ_b,
                                                   "λ_g" : lambda m: m.λ_g,
                                                   "index I" : lambda m: m.I,
                                                   "MSWA_g"  : lambda m: m.MSWA_g,
                                                   "MSWA_b"  : lambda m: m.MSWA_b,
                                                   "MSWA_e"  : lambda m: m.MSWA_e,
                                                   "HHI"     : lambda m: m.HHI

                                                    },
                                agent_reporters = {"name": lambda a: a.unique_id, 
                                                   "e_fs": lambda a: a.e_f, 
                                                   "p_fs": lambda a: a.p_f,
                                                   "b_fs": lambda a: a.b_f, 
                                                   "g_fs": lambda a: a.g_f,
                                                   "I_fs": lambda a: a.I_f,
                                                   "ms_fs": lambda a: a.ms_f,
                                                   "r_fs": lambda a: a.r_f,
                                                   "c_fs": lambda a: a.c_f,
                                                   "f_c_fs": lambda a: a.f_c_f,
                                                   "π_fs": lambda a: a.π_f,
                                                   "s_fs": lambda a: a.s_f,
                                                   "step": lambda a: a.step_2,
                                                   "proj": lambda a: a.act_proj,
                                                   "p_success": lambda a: a.p_success,
                                                   "s_to_r": lambda a: a.s_f/a.r_f,
                                                   "green": lambda a: a.p_i_g,
                                                   "efficiency": lambda a: a.p_i_e,
                                                   "quality": lambda a: a.p_i_b,
                                                   "p_r_imits": lambda a: a.p_r_imit,
                                                   "age" : lambda a: a.age,
                                                   'innov': lambda a: a.innov,
                                                   "sum_of_innov": lambda a:
                                                   a.p_i_e + a.p_i_b + a.p_i_g})

        # Create firms
        temp_num = 0
        for el in firm_scen:
            for i in range(el['num']):
                a = Firm(temp_num, self, el, False)
                temp_num += 1
                self.schedule.add(a)    

    def agg_income(self):

        # Modult for calculating growth in investments
        # self.Θ_growth = 0 if self.step == 1 else self.Θ_growth
        self.Θ_b = 0; self.Θ_g = 0; self.Θ_c = 0
        for a in self.schedule.agents:
            if a.unique_id in self.invest['green']:
                self.Θ_g = self.Θ_g + a.ms_f
            if a.unique_id in self.invest['efficiency']:
                self.Θ_c = self.Θ_c + a.ms_f
            if a.unique_id in self.invest['quality']:
                self.Θ_b = self.Θ_b + a.ms_f
        
        self.Θ_growth = 0 if self.schedule.time == 1 else self.Θ_growth
        self.Θ_growth = (self.Θ_g + self.Θ_b +self.Θ_c) - self.current_growth
        self.current_growth = 0 if self.schedule.time == 1 else (self.Θ_g + self.Θ_b +self.Θ_c)

        # Module for calculating GDPs

        self.GT = (self.ω_g * self.Θ_g + self.ω_b * self.Θ_b - self.ω_c * self.Θ_c) / (self.ω_b)* (self.G_max-self.G_min)/2 + (self.G_max+self.G_min)/2

        # self.GT = (self.ω_g * self.Θ_g + self.ω_b * self.Θ_b - self.ω_c * self.Θ_c) * (self.G_max - self.G_min)/2 + self.G_min
        
        if self.schedule.time >= self.con['con_start'] and self.schedule.time <= self.con['con_start'] + self.con['con_duration']: 
            # self.GT += self.shock['shock_drop']/12
            self.λ_g += self.con['increase']/(self.con['con_duration'])
            self.λ_e -= self.con['increase']/(self.con['con_duration']*2)
            self.λ_b -= self.con['increase']/(self.con['con_duration']*2)

        # module for simulating the recession
        if self.schedule.time >= self.shock['shock_start'] and self.schedule.time <= self.shock['shock_start'] + self.shock['shock_duration']: 
            self.GT += self.shock['shock_drop']/12
        
        # if (self.schedule.time > self.shock['shock_start'] + self.shock['shock_duration']) and (self.schedule.time <= self.shock['shock_start'] + self.shock['shock_duration'] + 12):
        #     self.λ_e += self.shock['shock_drop']/12/3 * self.shock['shock_duration']/12
        #     self.λ_b -= self.shock['shock_drop']/24/3 * self.shock['shock_duration']/12
        #     self.λ_g -= self.shock['shock_drop']/24/3 * self.shock['shock_duration']/12

        if self.λ_dur > 0:
            if self.schedule.time == self.λ_start: 
                self.λ_e -= self.λ_g_b/2
                self.λ_b -= self.λ_g_b/2
                self.λ_g += self.λ_g_b
                for a in self.schedule.agents:
                    a.λ_e = self.λ_e
                    a.λ_b = self.λ_b
                    a.λ_g = self.λ_g                   
            if self.schedule.time >= self.λ_start + self.λ_dur and self.schedule.time < self.λ_start + self.λ_dur + self.λ_off :
                self.λ_e += self.λ_g_b/2/self.λ_off
                self.λ_b += self.λ_g_b/2/self.λ_off
                self.λ_g -= self.λ_g_b/self.λ_off
                for a in self.schedule.agents:
                    a.λ_e = self.λ_e
                    a.λ_b = self.λ_b
                    a.λ_g = self.λ_g                   

        self.YT = self.YT * self.GT
        if self.λ_dur > 0:
            if self.schedule.time == self.λ_start :
                self.YT = (self.pp/100+1) * self.YT
            if self.schedule.time >= self.λ_start + self.λ_dur and self.schedule.time < self.λ_start + self.λ_dur + self.λ_off:
                self.YT = (1-self.pp/100/self.λ_off) * self.YT
        self.Y = ζ * self.Y + (1-ζ) * self.YT
        self.MA_y_L = self.MA_y_L * δ_long + self.Y * (1 - δ_long) 
        self.MA_y_S = self.MA_y_S * δ_short + self.Y * (1 - δ_short)
        self.I = self.MA_y_S / self.MA_y_L
        
    def step(self):
        # Phase 1: Trading, assigning sales to firms and updating their accounts:
        self.schedule.step()
        
        self.agg_income()

        total_I_f = sum(np.array([a.I_f for a in self.schedule.agents])**self.α)
        for a in self.schedule.agents:
            a.update_agent_a(total_I_f, self.Y,self.α,self.μ,self.Φ,self.ψ,self.δ)
        
        total_ms_f = sum(np.array([a.ms_f for a in self.schedule.agents])**self.η)
        for a in self.schedule.agents:
            a.update_agent_b(total_ms_f,self.η)

        # create metrics inside the agents
        # MSWA
        self.MSWA_g = sum(np.array([a.ms_f*a.g_f for a in self.schedule.agents]))
        self.MSWA_b = sum(np.array([a.ms_f*a.b_f for a in self.schedule.agents]))
        self.MSWA_e = sum(np.array([a.ms_f*a.e_f for a in self.schedule.agents]))
        self.HHI = sum(np.array([a.ms_f**2 for a in self.schedule.agents]))


        self.dc.collect(self) # data collection after the accounts of firms have been updated

        # Phase 2: Each firm attempts or continuous an innovation project, whose successful           
        for a in self.schedule.agents:
                            
            if a.act_proj != False:
                if a.loan_dur < L_xs[a.act_proj]:
                    a.loan_dur += 1
                    # if a.unique_id in self.invest[a.act_proj]:
                    #     self.invest[a.act_pr3j].remove(a.unique_id)
                else:
                    a.loan_dur = 0 
                    self.invest[a.act_proj].remove(a.unique_id)
                    if np.random.binomial(1,a.P_r_x_succ[a.act_proj]):
                        if a.act_proj == "quality":
                            a.b_f = a.b_f + 0.2 if a.b_f < 39.5 else a.b_f
                            # a.g_f = a.g_f - 0.15 if a.g_f > 0 else a.g_f
                            # a.e_f = a.e_f - 0.15 if a.e_f > 0 else a.e_f
                            a.p_f = self.p_hat + np.log(self.M_e/a.e_f - 1)/self.γ_e # (1)
                        if a.act_proj == "green":
                            a.g_f = a.g_f + 0.2 if a.g_f < 39.5 else a.g_f
                            # a.b_f = a.b_f - 0.15 if a.b_f > 0 else a.b_f
                            # a.e_f = a.e_f - 0.15 if a.e_f > 0 else a.e_f
                            a.p_f = self.p_hat + np.log(self.M_e/a.e_f - 1)/self.γ_e # (1)
                        if a.act_proj == "efficiency":
                            a.e_f = a.e_f + 0.2 if a.e_f < 39.5 else a.e_f
                            # a.g_f = a.g_f - 0.15 if a.g_f > 0 else a.g_f
                            # a.b_f = a.b_f - 0.15 if a.b_f > 0 else a.b_f
                            a.p_f = self.p_hat + np.log(self.M_e/a.e_f - 1)/self.γ_e # (1) 
                    a.act_proj = False

            if a.act_proj == False:
                temp_choice = np.random.choice( ['efficiency', 'green', 'quality'], 1,
                p = [a.p_i_e, a.p_i_g, a.p_i_b])
                
                if a.s_f > 0: 
                    # this is a rather bad formula
                    a.p_success = self.f_a * a.s_f/a.r_f * (self.P_r_l[temp_choice[0]] + self.σ[temp_choice[0]]) * self.I
                else:
                    a.p_success = 0
                
                a.p_success = 1 if a.p_success > 1 else a.p_success

                if np.random.binomial(1,a.p_success):
                    a.act_proj = temp_choice[0]
              
                    a.loan_dur += 1    
                    self.invest[temp_choice[0]].append(a.unique_id)
            
        if self.dyn_firms:
            removed = False
            for a in self.schedule.agents:
                if a.age > minAge:
                    if a.s_f < 0:
                        if a.ms_f < τ:
                            self.schedule.remove(a)
                            removed = True
                    if a.ms_f < τ_2 and removed == False:
                        self.schedule.remove(a)
                        removed = True 
            num_of_entered_firms = 0
            for a in self.schedule.agents:
                if np.random.binomial(1,a.p_r_imit*self.Pr_new): 

                    a2 = Firm(self.num_agents + self.total_diff_firms,
                    self, self.imit_scen, True)
                    self.total_diff_firms += 1 
                    temp_random = np.random.uniform()
                    # here it decides what type of innovator the firm entering 
                    # firm will be
                    if temp_random < 1/3:
                        a2.e_f = a.e_f*self.Ω ; a2.b_f = a.b_f*self.Ω ; a2.g_f = a.g_f/self.Ω
                        a2.innov = 'Green'
                    elif temp_random < 2/3:
                        a2.e_f = a.e_f*self.Ω ; a2.b_f = a.b_f/self.Ω ; a2.g_f = a.g_f*self.Ω
                        a2.innov = 'Qual'
                    else:
                        a2.e_f = a.e_f/self.Ω ; a2.b_f = a.b_f*self.Ω ; a2.g_f = a.g_f*self.Ω
                        a2.innov = 'Effic'

                    a2.p_f = self.p_hat + np.log(self.M_e/a2.e_f - 1)/self.γ_e # (1)
                    a2.I_f = (a2.e_f**self.λ_e)*(a2.b_f**self.λ_b)*(a2.g_f**self.λ_g)
                    a2.step_2 = a.step_2
                    a2.loan_dur = 0
                    a2.age = -1

                    self.schedule.add(a2)
                    num_of_entered_firms += 1
        
    

    