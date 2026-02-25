print("Hello, Statistician welcome")
 

import statistics as stats

def stats_summary(data):

    #average
    mean_value = sum(data) / len(data)

    #minimum
    min_value = min(data)

    #maximum
    max_value = max(data)

    #range 
    range_value = max_value - min_value

    #median
    median_value = stats.median(data)



    return{
        "Mean": mean_value,
        "Median": median_value,
        "Minimum": min_value,
        "Maximum": max_value,
        "Range": range_value,
    }


#student scores
scores = [70, 85, 90, 60, 75]

summary = stats_summary(scores)

print("Dataset:", scores)
for key, value in summary.items():
    print(f"{key}: {value}")