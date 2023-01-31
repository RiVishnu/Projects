#!/usr/bin/env python
# coding: utf-8

# In[53]:


#importing necessary libraries
import pandas as pd
import numpy as np
import datetime
from scipy.stats import chi2_contingency, beta
from IPython.display import Image
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# Experiment defnition :
# 
# Aim of the experiment : To see whether the new version of webpage has any effect in  purchase conversion.
# 
# Purchase conversion = #Converted User / # Exposed Users
# 
# The two types of sample thats compared are:
# Control - access the old page
# Treatment - access the new page
# 
# Exposure = User in the control or the treatment group seeing their respective version of their page (old/new version)
# 
# Conversion = User making a purchase within 7 days of exposure
# 
# Target difference between the control and treatment group = 2%

# 
# # Data Analysis

# In[2]:


#importing the data
ab_data = pd.read_csv('/Users/vishn/Documents/Vishnu Priya/ASU/AB testing/ab_data.csv')


# In[5]:


ab_data.sample()


# In[49]:



Conversion_rates = ab_data.groupby('group')['converted']

std_dev = lambda X : np.std(X, ddof = 0)
Std_err = lambda X :stats.sem(X, ddof = 0)

Conversion_rates = Conversion_rates.agg([np.mean, std_dev, Std_err])
Conversion_rates.columns = ['Conversion_rates', 'std_deviation', 'std_error']

Conversion_rates.style.format('{:.3f}')


# The conversion rate of the control and the treatment group is almost the same. 

# In[55]:


plt.figure(figsize=(8,6))

sns.barplot(x=ab_data['group'], y=ab_data['converted'], ci=False)

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);


# From the graph also, its evident, that the conversion rate in the control as well as the treatment group is almost the same

# In[7]:


start_time = datetime.datetime.strptime(ab_data['timestamp'].min(), '%Y-%m-%d %H:%M:%S.%f')
end_time = datetime.datetime.strptime(ab_data['timestamp'].max(), '%Y-%m-%d %H:%M:%S.%f')
experiment_duration = (end_time - start_time).days

print(f"Number of unique users in experiment: {ab_data['user_id'].nunique()}")
print(f"Data collected for {experiment_duration} days")
print(f"Landing pages to compare: {ab_data['landing_page'].unique().tolist()}")
print(f"Percentage of users in control: {round(ab_data[ab_data['group']=='control'].shape[0] * 100 / ab_data.shape[0])}%")


# In[ ]:


### Data Processing ######


# In[12]:


## To see if any users have been exposed to both the old and new page
counter = ab_data['user_id'].value_counts()
(counter > 1).value_counts()


# In[62]:


print ("% of double exposed users =",3894/290584*100 )


# 3894 users have been exposed to both the pages. This might be due to a technical glitch. Since the user who have been double exposed is in a very small propotion(1.34%) in comparisson to the data, we can delete them. 

# In[14]:


#Deleting the double exposed users
valid_users = pd.DataFrame(counter[counter == 1].index, columns=['user_id'])
ab_data = ab_data.merge(valid_users, on=['user_id'])


# In[17]:


# adding column 'week' to the data
ab_data['week'] = ab_data['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').isocalendar()[1])
ab_data.sample(3)


# In[18]:


# to see the timeline of the experiment
ab_data['week'].value_counts()


# From the above, we can conclude that the timeline of the experiment is 4 weeks

# In[21]:


timeline = 4
experiment_data = ab_data[ab_data['week'] <= timeline]
control = experiment_data[experiment_data['group']=='control']
treatment = experiment_data[experiment_data['group']=='treatment']

control_conversion_perc = round(control['converted'].sum() * 100/ control['converted'].count(), 3)
treatment_conversion_perc = round(treatment['converted'].sum() * 100/ treatment['converted'].count(), 3)
difference = round(treatment_conversion_perc - control_conversion_perc, 3)


# In[22]:


print(f"Treatment Conversion Rate: {treatment_conversion_perc}%")
print(f"Control Conversion Rate: {control_conversion_perc}%")
print(f"difference = {difference}%")


# # Chi- Squared test :

# Null Hypothesis  ===> Control and Treatement data are independent
# 
# Alternate Hypothesis ===> Control and Treatemet data are not independent

# In[23]:


# Creating Contigency Table
control_converted = control['converted'].sum()
treatment_converted = treatment['converted'].sum()
control_non_converted = control['converted'].count() - control_converted
treatment_non_converted = treatment['converted'].count() - treatment_converted
contingency_table = np.array([[control_converted, control_non_converted], 
                              [treatment_converted, treatment_non_converted]])
contingency_table


# In[25]:


chi, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
chi, p_value


# The p value (= 0.232) is greater than 0.05, we cannot reject Null Hypothesis. So we can conclude that Control and Independent data are independent

# # Z test 

# Null Hypothesis : The conversion rates of the treatment and the control group is equal
# 
# Alternate Hypothesis : The conversion rates of the treatment and the control group is not equal
# 
# Confidence level = 95% Alpha  = 0.05

# In[59]:


from statsmodels.stats.proportion import proportions_ztest, proportion_confint
control_conv = ab_data[ab_data['group'] == 'control']['converted']
treatment_conv = ab_data[ab_data['group'] == 'treatment']['converted']
n_con = control_conv.count()
n_treat = treatment_conv.count()
successes = [control_conv.sum(), treatment_conv.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'confidence interval 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'confidence interval 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


# The p value (=0.232) is greater than alpha(0.05). SO we cannot reject Null Hypothesis.
# Hence we can conclude that the new version of the webpage did not create impact in the purchase conversion.

# # Bayesian test

# In[28]:


prior = ab_data[(ab_data['week'] == 1) & (ab_data['group']=='control')]


# In[29]:


prior_means = []
for i in range(10000):
    prior_means.append(prior.sample(1000)['converted'].mean())


# In[31]:


prior_means[:8]


# In[32]:


# Model Beta Distribtion from sample means
prior_alpha, prior_beta, _, _ = beta.fit(prior_means, floc=0, fscale=1)


# In[39]:


NUM_WEEKS = 4 # Vary number to get experiment data at weekly points in time
experiment_data = ab_data[(ab_data['week'] > 1) & (ab_data['week'] <= NUM_WEEKS)]
control = experiment_data[experiment_data['group']=='control']
treatment = experiment_data[experiment_data['group']=='treatment']

control_conversion_perc = round(control['converted'].sum() * 100/ control['converted'].count(), 3)
treatment_conversion_perc = round(treatment['converted'].sum() * 100/ treatment['converted'].count(), 3)
difference = round((treatment_conversion_perc - control_conversion_perc) / control_conversion_perc , 3)

print(f"Treatment Conversion Rate: {treatment_conversion_perc}%")
print(f"Control Conversion Rate: {control_conversion_perc}%")
print(f"difference percentage = {difference}%")


# In[33]:


control_converted = control['converted'].sum()
treatment_converted = treatment['converted'].sum()
control_non_converted = control['converted'].count() - control_converted
treatment_non_converted = treatment['converted'].count() - treatment_converted

# Update Prior parameters with experiment conversion rates
posterior_control = beta(prior_alpha + control_converted, prior_beta + control_non_converted)
posterior_treatment = beta(prior_alpha + treatment_converted, prior_beta + treatment_non_converted)

# Sample from Posteriors
control_samples = posterior_control.rvs(1000)
treatment_samples = posterior_treatment.rvs(1000)
probability = np.mean(treatment_samples > control_samples)
print(f"Probability that treatment > control: {probability * 100}%")


# In[34]:


(control_mu), (control_var) = posterior_control.stats()
(treatment_mu), (treatment_var) = posterior_treatment.stats()
print(f"Control Posterior: Mean: {control_mu}, Variance: {control_var}") 
print(f"Treatment Posterior: Mean: {treatment_mu}, Variance: {treatment_var}") 


# In[43]:


Difference_percentage = (treatment_samples - control_samples) / control_samples
print(f"Probability of a 2% difference: {np.mean((100 * Difference_percentage) > 2) * 100}%")


# Conclusion :
#     
# The difference between the conversion rate of the control and the treatment group is less than 2% which is our target difference. So we can conclude that the new version of the webpage didnt cause any appreciable change in the purchase conversion.
