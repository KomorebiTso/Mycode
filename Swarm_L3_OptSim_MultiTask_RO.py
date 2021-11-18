'''
# SWARM L3 Multitask RO (robust optimization) Model
The swarm design should cover mission requirements of three different tasks.
The target parameters are uncertaint, inlcuding the minimum attack power needed to destroy the target and the number of each target type.

## Input design variables (total design variable number: (m+2)*n_attc + (m+2)*n_obs + 2; m: target type number; n_attc: attack type number; n_obs: observation type number)
In this case: m=6, n_attc=3, n_obs=4, total design variable number = 24 + 32 + 2 = 58
>* **Lamda_Hete**: The expected value of detroyed heterogeneous node number within unit time step. Discrete variable,  within range [5, 20] with interval 1. The destroyed number obeys normal distribution. 
* **Speed_ave**: Flying speed. Normal uncertainty. Mean value is a discrete variable, within range[5, 10] with interval 1. Standard variation is 10% of the mean value.

* **attc_capa**: A n_attc*2 matrix. In this case, n_attc=3 represents there are three attck unit types. 
For the 1st row, the first element is a discrete variable representing the attack power within range [0.05, 0.15] with interval 0.05, and the second element is a discrete variable representing the attack success probability within range [0.55, 0.65] with interval 0.05.
For the 2nd row, the first element is a discrete variable representing the attack power within range [0.2, 0.35] with interval 0.05, and the second element is a discrete variable representing the attack success probability within range [0.65, 0.75] with interval 0.05.
For the 3rd row, the first element is a discrete variable representing the attack power within range [0.4, 0.55] with interval 0.05, and the second element is a discrete variable representing the attack success probability within range [0.75, 0.85] with interval 0.05.

* **obs_capa**: A n_obs*2 matrix. In this case, n_obs=4 represents there are four observation unit types. 
For the 1st row, the first element is a discrete variable representing the observation coverage within range [1, 1.5] with interval 0.5, and the second element is a discrete variable representing the attack success probability within range [0.55, 0.65] with interval 0.05.
For the 2nd row, the first element is a discrete variable representing the observation coverage within range [2, 2.5] with interval 0.5, and the second element is a discrete variable representing the attack success probability within range [0.65, 0.75] with interval 0.05.
For the 3rd row, the first element is a discrete variable representing the observation coverage within range [3, 3.5] with interval 0.5, and the second element is a discrete variable representing the attack success probability within range [0.75, 0.85] with interval 0.05.
For the 4th row, the first element is a discrete variable representing the observation coverage within range [4, 4.5] with interval 0.5, and the second element is a discrete variable representing the attack success probability within range [0.85, 0.95] with interval 0.05.

* **attc_num**: A m*n_attc matrix. The j_th element in the i_th row represents the number of type j action units allocated for target i. 
In this case study, m=6 (six targets) and n_attc=3 (three attck types). Each element is a discrete variable, within range [80, 300] with interval 10.


* **obs_num**: A m*n_obs matrix. The j_th element in the i_th row represents the number of type j observation units allocated for target i. 
In this case study, m=6 (six targets) and n_obs=4 (four observation types). Each element is a discrete variable, within range [80, 300] with interval 10.

## Output

>* ** maximize multitask_score_weighted_average**: The expected weighted score for different tasks.
* ** minimize multitask_variation**: The performance robustness among different tasks.  
* ** minimize total_cost**: The total cost of mission resource.
    


## Analysis function
    multitask_score_weighted_average,
    multitask_score_robustness,
    multitask_variation = MultiTask_Score(attc_num,obs_num,attc_capa,obs_capa,Lamda_Hete,Speed_ave)
    total_cost = Cost_Calc(attc_num,obs_num,attc_capa,obs_capa,Lamda_Hete,Speed_ave)

'''

import numpy as np
import math
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt

target_area = 15000
rescue_area = 18000
targ_distance = 50
target_time_coverage_req = 20
target_coverage_cycle_req = 4

rescue_time_coverage_req = 10  # Required continuous area searching time coverage for rescu
rescue_coverage_cycle_req = 3  # Required time cycle needed to cover the total area for rescuee
task_confi = 0.85  # The success confidence in task.

target_prop_base = np.array([[10, 15, 1], [8, 12, 2], [5, 10, 4], [3, 6, 8], [2, 4, 10], [1, 2, 20]])

verbos = False

# Analyze performances of taks 1: target searching in designated area
def MultiTask_Score(attc_num, obs_num, attc_capa, obs_capa, Lamda_Hete, Speed_ave, sample_size):
    obs_area_percent_mean, obs_area_percent_std, obs_cycle_percent_mean, obs_cycle_percent_std = Score_Task_Obs(obs_num,
                                                                                                                obs_capa,
                                                                                                                Speed_ave,
                                                                                                                sample_size)
    rescue_area_percent_mean, rescue_area_percent_std, rescue_cycle_percent_mean, rescue_cycle_percent_std = Score_Task_Rescue(
        obs_num, obs_capa, Speed_ave, sample_size)
    defense_breakthrough_mean, defense_breakthrough_std, target_destroypercent_mean, target_destroypercent_std = Score_Task_Attack(
        attc_num, obs_num, attc_capa, obs_capa, Lamda_Hete, Speed_ave, sample_size)
    if verbos:
        print("obs_area_percent_mean = %.2f" % (obs_area_percent_mean),
              "obs_area_percent_std = %.2f" % (obs_area_percent_std))
        print("obs_cycle_percent_mean = %.2f" % (obs_cycle_percent_mean),
              "obs_cycle_percent_std = %.2f" % (obs_cycle_percent_std))
        print("rescue_area_percent_mean = %.2f" % (rescue_area_percent_mean),
              "rescue_area_percent_std = %.2f" % (rescue_area_percent_std))
        print("rescue_cycle_percent_mean = %.2f" % (rescue_cycle_percent_mean),
              "rescue_cycle_percent_std = %.2f" % (rescue_cycle_percent_std))
        print("defense_breakthrough_mean = %.2f" % (defense_breakthrough_mean),
              "defense_breakthrough_std = %.2f" % (defense_breakthrough_std))
        print("target_destroypercent_mean = %.2f" % (target_destroypercent_mean),
              "target_destroypercent_std = %.2f" % (target_destroypercent_std))

    multi_score_mean = np.array(
        [obs_area_percent_mean, obs_cycle_percent_mean, rescue_area_percent_mean, rescue_cycle_percent_mean,
         defense_breakthrough_mean, target_destroypercent_mean])
    multi_score_std = np.array(
        [obs_area_percent_std, obs_cycle_percent_std, rescue_area_percent_std, rescue_cycle_percent_std,
         defense_breakthrough_std, target_destroypercent_std])

    # multitask_score_average = np.mean(multi_score_mean)
    multitask_score_weighted_average = np.sum(multi_score_mean * np.array([0.2, 0.1, 0.15, 0.15, 0.1, 0.3]))
    multitask_score_robustness = np.mean(multi_score_std)
    multitask_variation = np.std(multi_score_mean)
    return multitask_score_weighted_average, multitask_score_robustness, multitask_variation


# Analyze swarm cost for operation
def Cost_Calc(attc_num, obs_num, attc_capa, obs_capa, Lamda_Hete, Speed_ave):
    obs_price = np.ones(obs_capa.shape[0])
    for i in range(obs_capa.shape[0]):
        # obs_pool = np.array([[1.0,0.6,0.6],[2.0,1,0.7],[3.0,2,0.8],[4.0,3,0.9]])
        obs_price[i] = round(0.1 + obs_capa[i][0] * 0.4 * np.log2(obs_capa[i][1] * 4), 1)

    attc_price = np.ones(attc_capa.shape[0])
    for i in range(attc_capa.shape[0]):
        # attc_pool = np.array([[0.1,1,0.5],[0.2,2,0.7],[0.4,3,0.8]])        
        attc_price[i] = round(0.5 + attc_capa[i][0] * 5 * np.log2(attc_capa[i][1] * 4), 1)

    obs_cost = np.sum(np.dot(obs_num, obs_price))
    attc_cost = np.sum(np.dot(attc_num, attc_price))

    # The baseline: Lamda_Hete=10, time_horizon=10

    if Lamda_Hete >= 10:
        Lamda_cof = (10 / Lamda_Hete)
    else:
        Lamda_cof = 1.8 ** (10 / Lamda_Hete)

    if Speed_ave >= 5:
        speed_cof = 1.6 * (Speed_ave / 5)
    else:
        speed_cof = 1.6 ** (Speed_ave / 5)

    total_cost = (obs_cost + attc_cost) * Lamda_cof * speed_cof

    return total_cost


# Calculta the total score (expectation)

# Analyze score of task 1: searching the targeted area
def Score_Task_Obs(obs_num, obs_capa, Speed_ave, sample_size):
    # Obtain performances of task 1
    target_uncertainty = 0.2
    # Time lasting capability. Suppose the power system can at least cover the distance to the target in the attack mission wherein the deployment of the swarm should be outside the target defense area and the swarm should fly and breakthrough the defense by themselves. 
    # For area searching and rescue mission, suppose the swarm can be directly deployed in the target area, and the flying lasting capability can be fully used for searching.
    Power_ave = np.floor(targ_distance / Speed_ave)
    unit_time_search_capa = np.sum(np.sum(obs_num, 0) * obs_capa[:, 0] * obs_capa[:,
                                                                         1])  # Total area covered within unit time with all the observation uavs
    full_area_capa = unit_time_search_capa * Power_ave  # Total effective observation area
    full_coverage_cycle = np.zeros(sample_size)  # Time cycle to finish a full area coverage searching
    area_percent = np.zeros(sample_size)  # Time cycle to finish a full area coverage searching
    cycle_percent = np.zeros(sample_size)
    for k in range(sample_size):
        real_area = np.random.normal(loc=target_area, scale=target_uncertainty * target_area)
        full_coverage_cycle[k] = real_area / unit_time_search_capa
        cycle_percent[k] = np.minimum(target_coverage_cycle_req / full_coverage_cycle[k], 1.1)
        total_area_coverrage_req = real_area * target_time_coverage_req / target_coverage_cycle_req
        area_percent[k] = full_area_capa / total_area_coverrage_req

    obs_area_percent_mean = np.mean(area_percent)
    obs_area_percent_std = np.std(area_percent)
    obs_cycle_percent_mean = np.mean(cycle_percent)
    obs_cycle_percent_std = np.std(cycle_percent)
    return obs_area_percent_mean, obs_area_percent_std, obs_cycle_percent_mean, obs_cycle_percent_std


# Analyze score of task 2: searching the targeted area for rescue
def Score_Task_Rescue(obs_num, obs_capa, Speed_ave, sample_size):
    # Obtain performances of task 1
    rescue_uncertainty = 0.2
    # Time lasting capability. Suppose the power system can at least cover the distance to the target in the attack mission wherein the deployment of the swarm should be outside the target defense area and the swarm should fly and breakthrough the defense by themselves. 
    # For area searching and rescue mission, suppose the swarm can be directly deployed in the target area, and the flying lasting capability can be fully used for searching.
    Power_ave = np.floor(targ_distance / Speed_ave)
    unit_time_search_capa = np.sum(np.sum(obs_num, 0) * obs_capa[:, 0] * obs_capa[:,
                                                                         1])  # Total area covered within unit time with all the observation uavs
    full_area_capa = unit_time_search_capa * Power_ave  # Total effective observation area
    full_coverage_cycle = np.zeros(sample_size)  # Time cycle to finish a full area coverage searching
    area_percent = np.zeros(sample_size)  # Time cycle to finish a full area coverage searching
    cycle_percent = np.zeros(sample_size)
    for k in range(sample_size):
        real_area = np.random.normal(loc=rescue_area, scale=rescue_uncertainty * rescue_area)
        full_coverage_cycle[k] = real_area / unit_time_search_capa
        cycle_percent[k] = np.minimum(rescue_coverage_cycle_req / full_coverage_cycle[k], 1.2)
        total_area_coverrage_req = real_area * rescue_time_coverage_req / rescue_coverage_cycle_req
        area_percent[k] = full_area_capa / total_area_coverrage_req

    rescue_area_percent_mean = np.mean(area_percent)
    rescue_area_percent_std = np.std(area_percent)
    rescue_cycle_percent_mean = np.mean(cycle_percent)
    rescue_cycle_percent_std = np.std(cycle_percent)
    return rescue_area_percent_mean, rescue_area_percent_std, rescue_cycle_percent_mean, rescue_cycle_percent_std


# Sample the total score with sampling remaining unit number and sampling uncertain target property with given distribution
def Score_Task_Attack(attc_num, obs_num, attc_capa, obs_capa, Lamda_Hete, Speed_ave, sample_size):
    # 1. Caculate the expectation number of remaining units of each type of swarm units for each target type
    attc_type = attc_capa.shape[0]
    obs_type = obs_capa.shape[0]
    targ_type = target_prop_base.shape[0]
    time_horizon = targ_distance / (
            Speed_ave * 1.2)  # For attack mission, maximum speed is used to shorten the breakthrough time.

    attc_remain_mean = np.maximum(attc_num - np.ones((targ_type, attc_type)) * Lamda_Hete * time_horizon, 0)
    attc_remain_sigma = np.sqrt(np.maximum(attc_num - np.ones((targ_type, attc_type)) * Lamda_Hete * time_horizon, 1))
    obs_remain_mean = np.maximum(obs_num - np.ones((targ_type, obs_type)) * Lamda_Hete * time_horizon, 0)
    obs_remain_sigma = np.sqrt(np.maximum(obs_num - np.ones((targ_type, obs_type)) * Lamda_Hete * time_horizon, 1))
    attc_remain = np.ones((sample_size, targ_type, attc_type))
    obs_remain = np.ones((sample_size, targ_type, obs_type))
    defense_breakthrough_percent = np.zeros(sample_size)
    for k in range(sample_size):
        for i in range(targ_type):
            for j in range(attc_type):
                attc_remain[k][i][j] = np.maximum(
                    np.random.normal(loc=attc_remain_mean[i][j], scale=attc_remain_sigma[i][j]), 0)
        for i in range(targ_type):
            for j in range(obs_type):
                obs_remain[k][i][j] = np.maximum(
                    np.random.normal(loc=obs_remain_mean[i][j], scale=obs_remain_sigma[i][j]), 0)
        defense_breakthrough_percent[k] = (np.sum(attc_remain[k]) + np.sum(obs_remain[k])) / (
                np.sum(attc_num) + np.sum(obs_num))

    # 2. Calculat the capability with remaining units
    # 2.1 Obs capability    
    targ_score_u = np.zeros((sample_size, targ_type))
    total_score = np.zeros(sample_size)
    task_percent = np.zeros(sample_size)
    targ_conf = np.zeros((sample_size, targ_type))
    for k in range(sample_size):
        hete_capa_u = np.zeros(targ_type)
        hete_capa_sigma = np.zeros(targ_type)
        target_prop = target_prop_base.copy()

        for i in range(targ_type):
            obs_u = obs_remain[k][i] * obs_capa[:, 0] * obs_capa[:, 1]
            attc_u = attc_remain[k][i] * attc_capa[:, 0] * attc_capa[:, 1]
            attc_sigma = np.sqrt(attc_remain[k][i] * attc_capa[:, 0] * attc_capa[:, 1] * (1 - attc_capa[:, 1]))
            Obs_Para = np.sum(obs_u) / np.max([np.sum(attc_remain[k][i]), 1])
            target_prop[i][1] = np.random.normal(target_prop_base[i][1], target_prop_base[i][1] * 0.2)
            target_prop[i][2] = np.random.normal(target_prop_base[i][2], target_prop_base[i][2] * 0.3)
            targ_fullscore = np.dot(target_prop[:, 0], target_prop[:, 2])

            # Obs units are more than attck untis
            if Obs_Para > 1:
                # The maximum attack capa increasement induced by obs cooperation is 1.3
                attc_u_total = np.sum(attc_u)
                attc_sigma_total = np.sqrt(np.sum(attc_sigma ** 2))
                hete_capa_u[i] = attc_u_total * np.min([1 + np.log(Obs_Para), 1.3])
                hete_capa_sigma[i] = attc_sigma_total * np.min([1 + np.log(Obs_Para), 1.3])

            # Obs number is less than attck number. Then obs is the bottleneck for attack capa.
            # The remaining obs units are allocated to the more powerful attack units with priority.
            else:
                effc_num = np.sum(obs_u)
                if effc_num <= attc_remain[k, i, 2]:
                    hete_capa_u[i] = effc_num * attc_capa[2, 0] * attc_capa[2, 1]
                    hete_capa_sigma[i] = np.sqrt(effc_num * attc_capa[2, 0] * attc_capa[2, 1] * (1 - attc_capa[2, 1]))

                elif attc_remain[k, i, 2] < effc_num <= attc_remain[k, i, 2] + attc_remain[k, i, 1]:
                    hete_capa_u[i] = attc_remain[k, i, 2] * attc_capa[2, 0] * attc_capa[2, 1] + (
                            effc_num - attc_remain[k, i, 2]) * attc_capa[1, 0] * attc_capa[1, 1]
                    sigma_temp = np.zeros(2)
                    sigma_temp[0] = np.sqrt(
                        attc_remain[k, i, 2] * attc_capa[2, 0] * attc_capa[2, 1] * (1 - attc_capa[2, 1]))
                    sigma_temp[1] = np.sqrt(
                        (effc_num - attc_remain[k, i, 2]) * attc_capa[1, 0] * attc_capa[1, 1] * (1 - attc_capa[1, 1]))
                    hete_capa_sigma[i] = np.sqrt(np.sum(sigma_temp ** 2))
                else:
                    hete_capa_u[i] = attc_remain[k, i, 2] * attc_capa[2, 0] * attc_capa[2, 1] + attc_remain[k, i, 1] * \
                                     attc_capa[1, 0] * attc_capa[1, 1] + (
                                             effc_num - attc_remain[k, i, 2] - attc_remain[k, i, 1]) * attc_capa[
                                         0, 0] * attc_capa[0, 1]
                    sigma_temp = np.zeros(3)
                    sigma_temp[0] = np.sqrt(
                        attc_remain[k, i, 2] * attc_capa[2, 0] * attc_capa[2, 1] * (1 - attc_capa[2, 1]))
                    sigma_temp[1] = np.sqrt(
                        attc_remain[k, i, 1] * attc_capa[1, 0] * attc_capa[1, 1] * (1 - attc_capa[1, 1]))
                    sigma_temp[2] = np.sqrt(
                        (effc_num - attc_remain[k, i, 2] - attc_remain[k, i, 1]) * attc_capa[0, 0] * attc_capa[0, 1] * (
                                1 - attc_capa[0, 1]))
                    hete_capa_sigma[i] = np.sqrt(np.sum(sigma_temp ** 2))
            # Calculate the hitted target number directly with hete_capa_u  
            # tar_hit_num = np.minimum(np.floor(hete_capa_u[i]/target_prop[i,1]),target_prop[i,2])

            # 3.Calculate the hitted target number with task_confi 
            # The target destroy threshold and target number are random. Calculate the statistics of scores with given swarm scheme. 

            tar_hit_num = np.minimum(np.floor(
                (norm.ppf(1 - task_confi) * hete_capa_sigma[i] + hete_capa_u[i]) / np.maximum(target_prop[i, 1],
                                                                                              target_prop_base[i][1])),
                target_prop[i, 2])

            targ_score_u[k][i] = target_prop[i, 0] * int(tar_hit_num)
            if targ_score_u[k][i] == 0:
                targ_conf[k][i] = 1
            else:
                targ_conf[k][i] = 1 - norm.cdf(tar_hit_num * target_prop[i, 1], hete_capa_u[i], hete_capa_sigma[i])
        total_score[k] = np.sum(targ_score_u[k])
        task_percent[k] = total_score[k] / targ_fullscore

    # total_score_mean = np.mean(total_score)
    # total_score_std = np.std(total_score)
    # total_score_median = np.median(total_score)

    '''
    font = {'family':'Times New Roman','weight':'normal','size':10}
    plt.title("task complishment percentage distribution")
    plt.ylabel("percentage",font)
    plt.hist(task_percent,50,density=True)
    plt.show()
    '''

    target_destroypercent_mean = np.mean(task_percent)
    target_destroypercent_std = np.std(task_percent)
    # total_percent_median = np.median(task_percent)

    defense_breakthrough_mean = np.mean(defense_breakthrough_percent)
    defense_breakthrough_std = np.std(defense_breakthrough_percent)

    return defense_breakthrough_mean, defense_breakthrough_std, target_destroypercent_mean, target_destroypercent_std


def get_fitness(ind, m, n_attc, n_obs, sample_size):
    assert len(ind) == 58
    # target_prop = np.array([[10, 15, 1], [8, 12, 2], [5, 10, 4], [3, 6, 8], [2, 4, 10], [1, 2, 20]])
    # 拆分决策变量
    attc_num = np.reshape(ind[0:m * n_attc], (m, n_attc))
    obs_num = np.reshape(ind[m * n_attc:m * (n_attc + n_obs)], (m, n_obs))
    attc_capa = np.reshape(ind[m * (n_attc + n_obs):m * (n_attc + n_obs) + n_attc * 2], (n_attc, 2))
    obs_capa = np.reshape(ind[m * (n_attc + n_obs) + n_attc * 2:m * (n_attc + n_obs) + (n_attc + n_obs) * 2],
                          (n_obs, 2))
    Lamda_Hete = ind[-2]
    Speed_ave = ind[-1]

    # task_confi = 0.85

    multitask_score_weighted_average, multitask_score_robustness, multitask_variation = MultiTask_Score(attc_num,
                                                                                                        obs_num,
                                                                                                        attc_capa,
                                                                                                        obs_capa,
                                                                                                        Lamda_Hete,
                                                                                                        Speed_ave,
                                                                                                        sample_size)
    total_cost = Cost_Calc(attc_num, obs_num, attc_capa, obs_capa, Lamda_Hete, Speed_ave)
    return multitask_score_weighted_average, multitask_score_robustness, multitask_variation, total_cost

# Example
if __name__ == "__main__":
    # The target property, each row represents target importance (the score of each target), and the minimum attack power needed to destroy,
    # and the total target number of this type

    target_prop_base = np.array([[10, 15, 1], [8, 12, 2], [5, 10, 4], [3, 6, 8], [2, 4, 10], [1, 2, 20]])
    target_total_type = target_prop_base.shape[0]
    target_area = 15000
    rescue_area = 18000
    targ_distance = 50
    target_time_coverage_req = 20  # Required continuous area searching/surveillance time coverage
    target_coverage_cycle_req = 4  # Required time cycle needed to cover the total area
    rescue_time_coverage_req = 10  # Required continuous area searching time coverage for rescue
    rescue_coverage_cycle_req = 3  # Required time cycle needed to cover the total area for rescue
    task_confi = 0.85  # The success confidence in task.
    sample_size = 2000

    # Mean of Poisson distribution: the expected value of detroyed heterogeneous node number within unit time step
    Lamda_Hete = 8
    # Average speed. The actual speed range is within 80% to 120% of average.
    Speed_ave = 9
    # There are three attack types in this case. Each row represents act_capa and attc_prob of each attack type respectively.
    attc_capa = np.array([[0.1, 0.5], [0.2, 0.7], [0.4, 0.8]])
    attc_total_type = attc_capa.shape[0]
    attc_num = np.ones((target_total_type, attc_total_type)) * 120
    # There are four observation types in this case. Each row represents obs_capa and obs_prob of each observation unit respectively.
    obs_capa = np.array([[1.0, 0.6], [2.0, 0.7], [3.0, 0.8], [4.0, 0.9]])
    obs_total_type = obs_capa.shape[0]
    obs_num = np.ones((target_total_type, obs_total_type)) * 80

    multitask_score_weighted_average, multitask_score_robustness, multitask_variation = MultiTask_Score(attc_num,
                                                                                                        obs_num,
                                                                                                        attc_capa,
                                                                                                        obs_capa,
                                                                                                        Lamda_Hete,
                                                                                                        Speed_ave,
                                                                                                        sample_size)
    total_cost = Cost_Calc(attc_num, obs_num, attc_capa, obs_capa, Lamda_Hete, Speed_ave)
    # print("Sample with uncertainty",total_score,targ_score_u,targ_conf)
    print("multitask_score_weighted_average= %.2f, multitask_score_robustness= %.2f,multitask_variation= %.2f" % (
    multitask_score_weighted_average, multitask_score_robustness, multitask_variation),
          "total_cost = %.2f" % (total_cost))
