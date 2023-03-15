# Python Projects!
```python
from types import GeneratorType
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
```
```python

denver = pd.read_csv('https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_denver/dwellings_denver.csv')
ml = pd.read_csv('https://raw.githubusercontent.com/byuidatascience/data4dwellings/master/data-raw/dwellings_ml/dwellings_ml.csv')
```
```python
#denver.head()
denver.arcstyle.unique()
```
```python
denver['before1980'] = denver.yrbuilt <= 1980
denver.head()
```
```python
subset_data = denver.sample(n= 4999)
chart = alt.Chart(subset_data).mark_boxplot().encode(
    x= alt.X('arcstyle'),
    y= alt.Y('yrbuilt', scale=alt.Scale(domain=(1850,2050)))
)
chart
```
```python
h_subset = ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)

sns.pairplot(h_subset, hue = 'before1980')

corr = h_subset.drop(columns = 'before1980').corr()
corr
```
```python
sns.heatmap(corr)
```
```python
denver.columns
alt.Chart(subset_data).mark_boxplot().encode(

    x= alt.X('numbaths'),
    y= alt.Y('yrbuilt', scale=alt.Scale(domain=(1850,2050)))
)
```
```python
x = ml.filter(["numbaths",'stories','livearea', 'gartype_None', 'gartype_Att', 'arcstyle_BI-LEVEL'])
y = ml.before1980
feature_names = x.columns
```
```python
#splits your data set randomly into test and training sets
# the x_train is to train predicting the y_train
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 192)

```
```python
x_train.head()
```
```python
# create the model
classifier = RandomForestClassifier()

# train the model
classifier.fit(x_train, y_train)

# make predictions
y_predictions = classifier.predict(x_test)
y_predictions
```
```python
# test how accurate predictions are
metrics.accuracy_score(y_test, y_predictions)
```
```python
print(metrics.classification_report(y_test,y_predictions))

```
```python
importances = classifier.feature_importances_
importances_df = pd.DataFrame({'Features' : feature_names, 'Importances': importances})
importances_df.head()
alt.Chart(importances_df).mark_bar().encode(
    x= alt.X('Features'),
    y= alt.Y('Importances')
)
```
