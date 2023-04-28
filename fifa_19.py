# -*- coding: utf-8 -*-
"""FIFA 19 (1).ipynb

##### Submitted By:
> Richard Honey

# FIFA 19 Analysis
"""

import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

df =  pd.read_csv('data.csv',delimiter = ',')

"""## Data Structure"""

df.head()

df.info()

df.shape

"""## Data Pre-processing

### Deleting Columns
"""

df = df.drop(columns="Unnamed: 0")

columns = ['Photo', 'Flag', 'Club Logo', 'Release Clause', 'Jersey Number', 'Loaned From', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM',
       'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB',
       'RCB', 'RB', 'LS', 'ST']
df = df.drop(columns, axis=1, inplace=False )

column = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

df = df.drop(column, axis=1, inplace=False )

"""#### Checking the columns"""

df.columns

"""### Data Conversion

#### Height Conversion from inch to centimeter
"""

#in centimeter
def height_conversion(height):
    if(pd.isna(height))!= True:
        chk = str(height)
        h = []
        h = chk.split("'")  
        ft = float(h[0])
        if( h[1] != ''):
            inch = float(h[1])
        else:
            inch = 0
        tot_inc = inch + ft*12
        h = tot_inc * 2.54
        return h
    else:
        return height
    
df['Height'] = df['Height'].apply(height_conversion)

"""#### Weight conversion: lbs to kg"""

#in kg
def weight_conversion(weight):
    if(pd.isna(weight))!= True:
        w = int(weight[0:-3])*0.453592
        return w
    else:
        return weight
    
df['Weight'] = df['Weight'].apply(weight_conversion)

"""#### Getting rid of all the elements that makes difficult to convert the different columns datatypes"""

df['Value'] = df['Value'].str.replace('€', '')
df['Value'] = df['Value'].str.replace('M', '')
df['Value'] = df['Value'].str.replace('K', '000')
df['Wage'] = df['Wage'].str.replace('€', '')
df['Wage'] = df['Wage'].str.replace('K', '000')

"""#### Renaming Columns"""

df.rename(columns = {'Value':"Value(millions)"}, inplace = True)

"""#### Changing the datatypes of the selected columns"""

df = df.astype({"Name":'category', "Value(millions)":'float', "Wage":'int64'})

"""#### Changing the datatype of date"""

df['Joined'] = pd.to_datetime(df['Joined'])

"""### Treating Null Values

#### Checking for Null values
"""

df.columns[df.isnull().any()]

df.isnull().sum()

"""#### Replacing Null values with most frequent values"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Club']])
df['Club'] = imputer.transform(df[['Club']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Preferred Foot']])
df['Preferred Foot'] = imputer.transform(df[['Preferred Foot']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['International Reputation']])
df['International Reputation'] = imputer.transform(df[['International Reputation']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Weak Foot']])
df['Weak Foot'] = imputer.transform(df[['Weak Foot']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Skill Moves']])
df['Skill Moves'] = imputer.transform(df[['Skill Moves']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Work Rate']])
df['Work Rate'] = imputer.transform(df[['Work Rate']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Body Type']])
df['Body Type'] = imputer.transform(df[['Body Type']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Real Face']])
df['Real Face'] = imputer.transform(df[['Real Face']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Position']])
df['Position'] = imputer.transform(df[['Position']])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer = imputer.fit(df[['Contract Valid Until']])
df['Contract Valid Until'] = imputer.transform(df[['Contract Valid Until']])

"""#### Replacing null values by forward filling"""

df['Joined'] = df['Joined'].fillna(value = df['Joined'].ffill())

"""#### Replacing Null values with mean"""

df['Height'] = df['Height'].fillna(value = df['Height'].mean())
df['Weight'] = df['Weight'].fillna(value = df['Weight'].mean())

"""#### Checking for any Null values"""

df.columns[df.isnull().any()]

df.info()

"""#### Saving the pre-processed data into an Excel sheet"""

df.to_csv('Pre-processed.csv')

"""## EDA

## Univariate Analysis
"""

df.describe()

"""#### Table of Indian footballers"""

def country(x):
    return df[df['Nationality'] == x][['Name','Overall','Potential','Position', 'Value(millions)', 'Height', 'Weight']]

country('India')

"""#### Players from different countries present in FIFA-2021"""

plt.style.use('dark_background') #top 50 nations that the players represent in FIFA 2021
plt.figure(figsize = (15,7))
df['Nationality'].value_counts().head(50).plot.bar(color = 'orangered')
plt.title('Players from different countries present in FIFA-2021')
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()

"""#### Ages in which maximum players are present"""

plt.figure(figsize=(12, 10))
sns.countplot(x=df.Age)
plt.xticks(rotation=90);

"""#### Word Cloud of nationalities of players in the shape of World Cup Trophy"""

from PIL import Image
from wordcloud import STOPWORDS
mask = np.array(Image.open('fifaimg.jpg'))
nationality = " ".join(n for n in df['Nationality'])
from wordcloud import WordCloud
plt.figure(figsize=(10,10))
wc = WordCloud(stopwords=STOPWORDS,
               mask=mask, background_color="black",
               max_words=2000, max_font_size=256,
               random_state=42, width=mask.shape[1],
               height=mask.shape[0]).generate(nationality)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

"""#### Football players preferred feet"""

preferred_foot_labels = df["Preferred Foot"].value_counts().index # (Right,Left) 
preferred_foot_values = df["Preferred Foot"].value_counts().values # (Right Values, Left Values)
explode = (0, 0.1) # used to separate a slice of cake

# Visualize
plt.figure(figsize = (7,7))
plt.pie(preferred_foot_values, labels=preferred_foot_labels,explode=explode, autopct='%1.2f%%')
plt.title('Football Players Preferred Feet',color = 'darkred',fontsize = 15)
plt.legend()
plt.show()

"""#### Distribution of overall rating for all players."""

sns.distplot(df['Overall'], bins=10, color='b')
plt.title("Distribution of Overall ratings of all Players")
plt.show()

"""#### Popular clubs around the world"""

df['Club'].value_counts().head(10)

"""#### Distribution of overall score in different popular clubs"""

some_clubs = ('AS Monaco', 'FC Barcelona', 'Valencia CF', 'Fortuna Düsseldorf', 'Cardiff City', 'Rayo Vallecano',
             'CD Leganés', 'Frosinone', 'Newcastle United', 'Southampton')

data_clubs = df.loc[df['Club'].isin(some_clubs) & df['Overall']]

plt.rcParams['figure.figsize'] = (15, 8)
ax = sns.boxplot(x = data_clubs['Club'], y = data_clubs['Overall'], palette = 'inferno')
ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 9)
ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)
ax.set_title(label = 'Distribution of Overall Score in Different popular Clubs', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()

"""## Bivariate Analysis

#### Pair plots for the following variables: Overall, Value(millions, Wage, International Reputation, Height and Weight
"""

sns.set()
cols = ['Overall', 'Value(millions)', 'Wage', 'International Reputation', 'Height', 'Weight'] 
sns.pairplot(df[cols], height = 2.5)
plt.show()

"""#### Heatmap of attributes of football players"""

import seaborn as sns
plt.figure(figsize = (12,10))
sns.heatmap(df.corr(), annot = True, fmt = '.1f')
plt.title("Corelation between the attributes of football players")
plt.show()

"""#### Country vs Overall Ratings of players belonging to them"""

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
rating = pd.DataFrame(df.groupby(['Nationality'])['Overall'].sum().reset_index())
count = pd.DataFrame(rating.groupby('Nationality')['Overall'].sum().reset_index())

plot = [go.Choropleth(
            colorscale = 'inferno',
            locationmode = 'country names',
            locations = count['Nationality'],
            text = count['Nationality'],
            z = count['Overall'],
)]

layout = go.Layout(title = 'Country vs Overall Ratings of players belonging to them')

fig = go.Figure(data = plot, layout = layout)
py.iplot(fig)

"""## Multivariate Analysis

### Data Pre-processing for PCA and K-Mean Clustering
"""

df2 = df.drop(columns= ['ID', 'Name', 'Nationality', 'Club', 'Value(millions)', 'Wage', 'Preferred Foot', 'Work Rate', 'Body Type',
              'Real Face','Position','Contract Valid Until', 'Height', 'Weight', 'Joined'])
df2.head()

df2.dtypes

X = df2.values
# Using the standard scaler method to standardize all of the features by converting them into values between -3 and +3.
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
X

"""### EDA for PCA and K-Mean Clustering

### Principal Component Analysis

#### Using PCA to reduce dimensionality of the data
"""

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents2 = pca.fit_transform(X)

"""#### Reduced features"""

principalComponents2

"""#### Dataframe featuring  the 3 principal components"""

PCA_dataset2 = pd.DataFrame(data = principalComponents2, columns = ['component3', 'component4', 'component5'] )
PCA_dataset2.head()

"""#### Extracting the three features"""

principal_component3 = PCA_dataset2['component3']
principal_component4 = PCA_dataset2['component4']
principal_component5 = PCA_dataset2['component5']

"""### 3D PCA"""

ax = plt.figure(figsize=(10,10)).gca(projection='3d')
plt.title('3D Principal Component Analysis (PCA)')
ax.scatter(
    xs=principal_component3, 
    ys=principal_component4, 
    zs=principal_component5, 
    #c = x_kmeans
)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.show()

"""## K-Mean

#### K-Mean clustering algorithm
"""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 100, init = 'k-means++', random_state = 1)
x_kmeans = kmeans.fit_predict(principalComponents2)

"""#### Adding 3 principal component features along with cluster features"""

df2['Principal Component 3'] = principal_component3
df2['Principal Component 4'] = principal_component4
df2['Principal Component 5'] = principal_component5
df2['Cluster2'] = x_kmeans

df2['Name'] = df['Name']

"""# 3D K-Mean"""

import plotly.express as px
fig = px.scatter_3d(df2, x='Principal Component 3', y='Principal Component 4', z='Principal Component 5',
              color=x_kmeans, log_x=True, hover_name="Name", hover_data=["Overall"])
fig.show()