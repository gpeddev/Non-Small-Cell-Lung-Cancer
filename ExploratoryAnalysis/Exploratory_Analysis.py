import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
pd.options.mode.chained_assignment = None

# fix colors
plot_colors=['#FFAA56','#568BFF']

# load dataframe
data = pd.read_csv("./ExploratoryAnalysis/data.csv")
data.dtypes

# Exploratory Analysis
data.head()

# Age at Histological Diagnosis
data["Age at Histological Diagnosis"].describe()

# make boxplot

# Weight (lbs)
data["Weight (lbs)"].unique()
data["Weight (lbs)"].describe()

# Gender
data["Gender"].unique()
data["Gender"].value_counts()
# Pie chart for Recurrence
values = data["Gender"].value_counts()
labels = data["Gender"].unique().tolist()
plt.pie(values,labels=labels, radius=1,autopct='%1.1f%%',colors=plot_colors)
plt.title("Patients Gender")
plt.show()

#Ethnicity
data["Ethnicity"].unique()
data["Ethnicity"].value_counts()
# percentage of each group
data["Ethnicity"].value_counts()/data["Ethnicity"].count()*100


# create pie chart for ethnicity
values = data["Ethnicity"].value_counts()
labels = data["Ethnicity"].unique().tolist()
plt.pie(values,labels=labels, radius=1,autopct='%1.1f%%',colors=plot_colors)
plt.title("Ethnicity of patients")
plt.show()



# Smoking status
data["Smoking status"].unique()
data["Smoking status"].value_counts()

# percentage of smoking status
data["Smoking status"].value_counts()/data["Smoking status"].count()*100


# Pie chart for smoking status
values = data["Smoking status"].value_counts()
labels = data["Smoking status"].unique().tolist()
plt.pie(values,labels=labels, radius=1,autopct='%1.1f%%',colors=plot_colors)
plt.title("Smoking status of patients")
plt.show()

# Pack Years
data["Pack Years"].unique()
data["Pack Years"].describe()

data.dtypes
# TODO: Histogram pack years and recurrence

# %GG
data["%GG"].unique()
# make pie chart
values = data["%GG"].value_counts()
labels = data["%GG"].unique().tolist()
plt.pie(values,labels=labels, radius=1,autopct='%1.1f%%',colors=plot_colors)
plt.title("Ground Glass %")
plt.show()
data.dtypes

# Tumor Locations
data["Tumor Location (choice=RUL)"].unique()
data["Tumor Location (choice=RML)"].unique()
data["Tumor Location (choice=RLL)"].unique()
data["Tumor Location (choice=LUL)"].unique()
data["Tumor Location (choice=LLL)"].unique()
data["Tumor Location (choice=L Lingula)"].unique()


# Histology
data["Histology"].unique()
data["Histology"].value_counts()

# percentage of Histology
data["Histology"].value_counts()/data["Histology"].count()*100

# Pie chart for Histology
# values = data["Histology"].value_counts()
# labels = data["Histology"].unique().tolist()
# plt.pie(values,labels=labels, radius=1)
# plt.show()

# Pathological T stage
data["Pathological T stage"].unique()

# Pathological N stage
data["Pathological N stage"].unique()

# Pathological M stage
data["Pathological M stage"].unique()

# Histopathological Grade
data["Histopathological Grade"].unique()


# Pleural invasion (elastic, visceral, or parietal)
data["Pleural invasion (elastic, visceral, or parietal)"].unique()

# Adjuvant Treatment
data["Adjuvant Treatment"].unique()

# Plot of adjuvant treatment
# values = data["Adjuvant Treatment"].value_counts()
# labels = data["Adjuvant Treatment"].unique().tolist()
# plt.pie(values,labels=labels, radius=1)
# plt.show()
# data["Adjuvant Treatment"].value_counts()


# Chemotherapy
data["Chemotherapy"].unique()
data["Chemotherapy"].value_counts()


# Radiation
data["Radiation"].unique()
data["Radiation"].value_counts()

# Recurrence
data["Recurrence"].unique()
data["Recurrence"].value_counts()

# percentage recurrence
data["Recurrence"].value_counts()/data["Recurrence"].count()*100


# Pie chart for Recurrence
values = data["Recurrence"].value_counts()
labels = data["Recurrence"].unique().tolist()
plt.pie(values,labels=labels, radius=1,autopct='%1.1f%%',colors=plot_colors)
plt.title("Patients recurrence")
plt.show()


values = data["Recurrence"].value_counts()
labels = data["Recurrence"].unique().tolist()
##################################################################################################################

#sns.barplot(x='Smoking status', y='Recurrence', data=data, estimator=lambda x: sum(x==0)*100.0/len(x))
#sns.countplot(x='Smoking status',data=data, palette='rainbow',hue='Gender')
#plt.show()

sns.countplot(x='Smoking status',data=data, palette=plot_colors,hue='Recurrence')
plt.title("Smoking status by recurrence")
plt.show()

sns.boxplot(x='Recurrence',y='Pack Years',data=data, palette=plot_colors)
plt.title("Recurrence by pack years")
plt.show()

