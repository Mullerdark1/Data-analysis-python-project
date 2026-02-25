import pandas as pd 
from scipy import stats

#Dataset of 10 students in sta264 exam
sta299_scores = [65, 70, 80, 75, 90, 85, 60, 95, 78, 88]

#One Sample T-test
#Testing if the mean of exam is 70
t_stat, p_value = stats.ttest_1samp(sta299_scores, 70)
#print("t_statistic:", t_stat)
#print("p_statistic:", p_value)

#Two sample T-test
sta264_scores = [65, 70, 80, 75, 90]
sta268_scores = [72, 68, 85, 80, 78]

t_stat, p_value = stats.ttest_ind(sta264_scores, sta268_scores)
#print("t_statistics:", t_stat)
#print("p_value:", p_value)

#Paired t-test
#before and after weights of 5 men in the gym 
before_gym = [60, 65, 70, 68, 72]
after_gym  = [62, 67, 75, 70, 74]

t_stat, p_value = stats.ttest_rel(before_gym, after_gym)
#print("t_statistics:", t_stat)
#print("p_value:", p_value)

#CHI-SQUARE TEST
import numpy as np

#Pass/fail student score in Department of statistics and biochemistry
observed = np.array([[30, 20],
        [25, 30],
        [20, 25]])

print("Contingency Table:")
print(observed)

chi2, p, dof, expected = stats.chi2_contingency(observed)
print("chi-square:", chi2)
print("p_value:", p)
