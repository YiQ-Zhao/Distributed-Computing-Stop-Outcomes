# Predicting Traffic Stop Outcomes
## Intro
The goal of this analysis was to predict whether or not a traffic or pedestrain stop would result in an arrest or citation. Moreover, we picked this big dataset particularly for use of distributed computing resources on AWS, including S3, MongoDB, Spark EC2 cluster, Spark SQL, and Spark MLlib

## Dataset
- data from [the Stanford Open Policing Project](https://openpolicing.stanford.edu/data/)



## Methods
### Pipeline
![pipeline](/imgs/pipelines)

### Data preprocessing
- Handle missing data
- Split fields with concatenated values
- Create additional features
- Encode categorical variables

### Modeling
- Logistic Regression


## Results
### Processing time comparison
![comparison](/imgs/comparison)
### Findings
- Collision/aggressive driving are more likely to lead to an arrest (duh….)
- Female officers are 15% less likely to make an arrest
- Stops on Interstate highway are 30% more likely to lead to an arrest
- Asian american officers are 20% less likely to make an arrest
- Driver-Officer gender/race difference has little impact
