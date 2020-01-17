#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data =  pd.read_csv("data.csv")
data.shape


# In[3]:


data = pd.DataFrame(data)
X = data
X = data.drop('Label', axis = 1)
#X = data.drop('Brightness', axis = 1)
Y = data['Label']
X


# In[16]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 0) 
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[17]:


print("Accuracy for Test : ",metrics.accuracy_score(y_test, y_pred))
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[20]:


for i in range(1,11):
    clf = DecisionTreeClassifier(criterion='gini', min_samples_leaf=i)
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Accuracy for Test for Gini when leaf node is ",i," : ", metrics.accuracy_score(y_test, y_pred))
    #print("AUROC Score for Test for Gini when leaf node is ",i," : ", roc_auc_score(y_test, y_pred))


# In[ ]:


from sklearn import svm

kernel = ["linear", "poly", "rbf"]
penalty = [0.1, 1]

for i in kernel:
    for j in penalty:
        svmc = svm.SVC(kernel=i, C=j)
        svmc.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        print("Accuracy for Test : ",metrics.accuracy_score(y_test, y_pred))
        print("Accuracy for Test when Kernel is = ",i," and Penalty is: ",j,": ",metrics.accuracy_score(y_test, y_pred))


# In[6]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



clf=RandomForestClassifier(n_estimators=100, max_depth=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_train_pred=clf.predict(X_train)
y_test_pred=clf.predict(X_test)


print("Tree: 100, Max_Depth: 3, Train Accuracy:",metrics.accuracy_score(y_train, y_train_pred))
print("Tree: 100, Max_Depth: 3, Test Accuracy:",metrics.accuracy_score(y_test, y_test_pred))


# In[7]:


clf=RandomForestClassifier(n_estimators=50, max_depth=5)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_train_pred=clf.predict(X_train)
y_test_pred=clf.predict(X_test)
print("Tree: 50, Max_Depth: 5, Train Accuracy:",metrics.accuracy_score(y_train, y_train_pred))
print("Tree: 50, Max_Depth: 5, Test Accuracy:",metrics.accuracy_score(y_test, y_test_pred))


# In[8]:


print ("...........................LDA..........................")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy for Test LDA : ",metrics.accuracy_score(y_test, y_pred))


# In[14]:


print ("...................... PCA...............")
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 5)
x_scaled = PCA().fit_transform(X)
x_scaled_train = PCA().fit_transform(X_train)
#x_scaled_test = PCA().fit_transform(scale(X_test))

pcamodel = PCA(n_components=11)
x_scaled_test = PCA().fit_transform(scale(X_test))
pca = LinearRegression().fit(x_scaled_train, Y_train)
y_pred = pca.predict(x_scaled_test)
pcaerror = mean_squared_error(Y_test, y_pred)
print ("MSE error for PCA: ", pcaerror)


# In[9]:


# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 13).fit(X_train, y_train) 

accuracy = knn.score(X_train, y_train) 
print (accuracy)
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print (accuracy) 
  
# creating a confusion matrix 
knn_predictions = knn.predict(X_test)  
#cm = confusion_matrix(y_test, knn_predictions) 


# In[ ]:


from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'poly', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
print (accuracy)


# In[5]:


# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
print (accuracy) 
accuracy = gnb.score(X_train, y_train) 
print (accuracy) 


# In[11]:


# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
score = dtree_model.score(X_test, y_test)
print (score)


# In[17]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
           # don't forget this if you're using jupyter!


model = RandomForestClassifier()
model.fit(X, Y)

(pd.Series(model.feature_importances_, index=X.columns)
   .nsmallest(8)
   .plot(kind='barh'))  


# In[18]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
           # don't forget this if you're using jupyter!


model = DecisionTreeClassifier()
model.fit(X, Y)

(pd.Series(model.feature_importances_, index=X.columns)
   .nsmallest(8)
   .plot(kind='barh')) 


# In[22]:


# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(data)


# In[28]:


# Build a DataFrame from the classification_report output_dict.
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=6))


# In[39]:


def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(4)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')

sampleClassificationReport = """             precision    recall  f1-score   support

           IDLE   0.988962  0.986784  0.987872 454
           Weight_Lifting   0.990196  0.990196  0.990196 204
           Running   0.964912  0.973451  0.969163 113
           Cycling   1.000000  1.000000  1.000000 134

avg / total       0.77      0.57      0.49       858"""


plot_classification_report(sampleClassificationReport)


# In[ ]:




