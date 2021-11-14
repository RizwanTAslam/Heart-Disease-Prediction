# Heart-Disease-Prediction

## Introduction

Heart diseases is a term covering any disorder of the heart. Heart diseases have become a major concern to deal with as studies show that the number of deaths due to heart diseases have increased significantly over the past few decades. Thus preventing Heart diseases has become more than necessary. Good data-driven systems for predicting heart diseases can improve the entire research and prevention process, making sure that more people can live healthy lives. This is where Machine Learning comes into play. Machine Learning helps in predicting the Heart diseases, and the predictions made are quite accurate.

## Dataset

The Dataset used in this project is UCI Heart Disease Dataset (https://archive.ics.uci.edu/ml/datasets/Heart+Disease?spm=5176.100239.blogcont54260.8.TRNGoO) This is also available in Kaggle platform.

## Load Dataset

  dataset=pd.read_csv("heart.csv")
  
  
  
### Checking for Null Values

  dataset.isnull().sum()
  
This dataset doesnt have any null values, so we dont need to preprocess the data

## Data Visualization

### Diagram for showing whether the person have heart disease or not

![have or not](https://user-images.githubusercontent.com/46325271/141688058-4533c198-9331-450e-a32f-3f5dec98bda9.png)

### Diagram for Age Vs Heart Disease

![ageXHD](https://user-images.githubusercontent.com/46325271/141688076-796cc59a-486a-4d63-8c0a-1d25b2b6bb8b.png)

## Evaluating the models

### Creating Training and Test datasets

  array = dataset.values
  x = array[:,0:13]
  y = array[:,13]
  x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.30, random_state = 1)
  
### Logistic Regression

  model_LR = LogisticRegression()
  model_LR.fit(x_train,y_train)
  predictions_LR = model_LR.predict(x_test)
  print (predictions_LR)
  
#### Confusion Matrix of Logistic Regression

  conf_mat_LR = confusion_matrix(predictions_LR,y_test)
  print (" Confusion Matrix for Logistic Regression Model: ")
  conf_mat_LR
  
  Confusion Matrix for Logistic Regression Model: 
  
  array([[30,  7],
       [11, 43]], dtype=int64)
  
#### Accuracy Score of Logistic Regression

  accuracy_LR = accuracy_score(predictions_LR,y_test)
  print ("Accuracy for Logistic Regression Model: ")
  accuracy_LR
  
  
  Accuracy for Logistic Regression Model: 

  0.8021978021978022


### Decision Tree

  model_DT = DecisionTreeClassifier()
  model_DT.fit (x_train, y_train)
  predictions_DT = model_DT.predict(x_test)
  
  
#### Confusion Matrix of Decision Tree

  conf_mat_DT = confusion_matrix (y_test, predictions_DT) 
  print (" Confusion Matrix for Decission Tree Model: ")
  conf_mat_DT
  
   Confusion Matrix for Decission Tree Model: 

   array([[31, 10],
       [15, 35]], dtype=int64)
       
#### Accuracy Score of Decision Tree

   accuracy_DT = accuracy_score (y_test, predictions_DT)
  print ("Accuracy for Decision Tree Model: ")
  accuracy_DT
  
  
  Accuracy for Decision Tree Model: 

  0.7252747252747253
  

#### Plotting the Tree Diagram

  fig, ax = plt.subplots(figsize=(20, 20))
  plot_tree(model_DT)

![DTDia](https://user-images.githubusercontent.com/46325271/141688327-090107f5-bddb-4c28-99bb-8bcd32a2d722.png)

### Random Forest

  model_RF = RandomForestClassifier()
  model_RF.fit(x_train,y_train)
  Predictions_RF = model_RF.predict(x_test)
  
  
#### Confusion Matrix of Random Forest

  conf_mat_RF = confusion_matrix (y_test, Predictions_RF)
  print (" Confusion Matrix for Random Forest Model: ")
  conf_mat_RF
  
  
   Confusion Matrix for Random Forest Model: 

  array([[31, 10],
       [12, 38]], dtype=int64)
       
#### Accuracy Score of Random Forest

  accuracy_RF = accuracy_score (y_test, Predictions_RF)
  print ("Accuracy for Random Forest Model: ")
  accuracy_RF
  
  
  Accuracy for Random Forest Model: 

  0.7582417582417582
  
  
#### Plotting the First Tree Diagram of Random Forest

  fig, ax = plt.subplots(figsize=(20, 20))
  plot_tree(model_RF.estimators_[0])
  
  ![RF0](https://user-images.githubusercontent.com/46325271/141688470-2606c2cc-b9de-4c78-ad6a-886a821b7b54.png)
  
## Selecting the Best Model

### Comparing Accuracy Score of all the above models

  score = { 'Logistic_Reg': [accuracy_LR] , 'Decision_Tree': [accuracy_DT], 'Random_Forest': [accuracy_RF]}
  score_df = pd.DataFrame(score)
  score_df
  
   	Logistic_Reg 	Decision_Tree 	Random_Forest
   	0.802198 	    0.725275 	      0.758242
    
  ![plotmodels](https://user-images.githubusercontent.com/46325271/141688589-f5c2198b-ea71-4307-a939-489751dc4227.png)

### Comparing Confusion Matrix of all the above models

  conf_mat_final = {'Logistic_Reg': conf_mat_LR, 'Decision_Tree': conf_mat_DT, 'Random_Forest': conf_mat_RF}
  for label,matrix in conf_mat_final.items():
    plt.title (label)
    sns.heatmap(matrix, annot=True)
    plt.show()
    
    
  ![conf_LR](https://user-images.githubusercontent.com/46325271/141688643-85b8b46f-ba04-472d-bb5e-c21a46992a51.png)
  ![conf_DT](https://user-images.githubusercontent.com/46325271/141688649-c8c4c423-1db9-4f41-b22c-d4b2b286dc78.png)
  ![conf_RF](https://user-images.githubusercontent.com/46325271/141688655-d9844afe-39b1-4b6f-95a9-0bf88a234a1a.png)


### Result

  The best model is Logistic Regression with accuracy: 
  0.8021978021978022



