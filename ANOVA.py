import pandas as pd
from scipy.stats import f_oneway

#ONE WAY ANOVA EXAMPLE

#Compute exam score of 5 students in 3 departmental exams
stats_score = [85, 82, 90, 88, 78]
compsci_score = [75, 78, 80, 82, 88]
physics_score = [80, 83, 85, 88, 92]

#perform ANOVA
f_stat, p_value = f_oneway(stats_score, compsci_score, physics_score)
#print("F-statistic:", f_stat)
#print("p-value:", p_value)

#TWO WAY ANOVA
#Test on Department and Gender with Score
import statsmodels.api as sm
from statsmodels.formula.api import ols

#create a dataset of 5 students from based on Gender and Department
data = {
    "Department" : ["Math"]*10 +["Statistics"]*10 + ["Physics"]*10,
    "Gender" : ["Male"]*5 + ["Female"]*5 + ["Male"]*5 + ["Female"]*5 + ["Male"]*5 + ["Female"]*5,
    "Score" : [72,70,75,68,74, 65,67,69,71,66, 85,88,86,87,89, 82,84,83,81,85, 78,76,80,77,79, 73,75,74,76,72]
}
df = pd.DataFrame(data)

model = ols('Score ~ C(Department)+ C(Gender) + C(Department): C(Gender)', data =df).fit()
anova_table =  sm.stats.anova_lm(model, typ=2)
print(anova_table)
