import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'], index=df.index)
finalDf = pd.concat([principalDf, df[['default payment next month']]], axis = 1)

counter = 0
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # xAxisLine = ((min(finalDf['principal component 1']), max(finalDf['principal component 1'])), (0, 0), (0,0))
    # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    # yAxisLine = ((0, 0), (min(finalDf['principal component 2']), max(finalDf['principal component 2'])), (0,0))
    # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    # zAxisLine = ((0, 0), (0,0), (min(finalDf['principal component 3']), max(finalDf['principal component 3'])))
    # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
    ax.legend(targets)
    ax.grid()
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 Component PCA', fontsize=20)
    indicesToKeep = finalDf['default payment next month'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50, alpha=0.5)
    counter+=1
    plt.savefig(f'PCA3D fig{counter}.png')
    plt.show()
    plt.clf()

exit()