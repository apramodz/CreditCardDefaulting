import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#pulling from dataset
url = "https://github.gatech.edu/raw/rchopra30/Hackyltic/master/default%20of%20credit%20card%20clients.csv?token=AAAIT6S4GESOY7BWVTWHLY26LLYFO"
df = pd.read_csv(url, header=0 ,names=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1",
 "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
 "PAY_AMT5", "PAY_AMT6", "default payment next month"])
features= ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1",
 "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
 "PAY_AMT5", "PAY_AMT6", "default payment next month"]

# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, ['default payment next month']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['default payment next month']]], axis = 1)

counter = 0
targets = [0, 1]
colors = ['r', 'g']
#iterate through "target" binary vals for default and no default loans
for color, target in zip(colors,targets):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    ax.legend(targets)
    ax.grid()
    indicesToKeep = finalDf['default payment next month'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    counter +=1
    plt.savefig(f'PCA fig{counter}')
    plt.show()
    plt.clf()
