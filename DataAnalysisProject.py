# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:59:17 2024

@author: amajid
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


st.header("""
          App for data analysis and predictive modeling
         
         """)
p_type=st.selectbox('Which type of problem would like to study?', ['Regression','Classification']) 

if p_type=='Classification':        
         
             
    usr_choice=st.selectbox('Please choose one of the following',['Run Example','Upload Data'])
    
    try:
    
        if usr_choice=='Run Example':
            st.write('In this example we will study titanic data and predict the survival.')
            data=df_titan=sns.load_dataset('titanic')
        else:
            data=st.file_uploader('Please choose your file') 
            data=pd.read_csv(data)  
    except ValueError:
        st.write('Please enter a valid data')
    
         
    
    #X=pd.DataFrame(data.data, columns=data.feature_names)
    #y=data.target
    st.dataframe(data,use_container_width=True)
    cn=data.columns
    
    target_name=st.selectbox('Please select target', cn)
    y=data[target_name]
    
    st.dataframe(y,use_container_width=True)
    
    st.subheader("Explore features of the data")
    
    f_name=st.multiselect('Please select the feature', cn,default=cn[0])
    
    
    
    no_graphs=len(f_name)
    #st.write(no_graphs)
    fig,ax=plt.subplots(no_graphs,1,squeeze=False,figsize=(8,10))
    plt.tight_layout()
    for i in range(no_graphs):
        sns.histplot(data=data,x=f_name[i],kde=True,ax=ax[i,0],hue=target_name)
    st.pyplot(fig)
    
    
    f_select=st.multiselect('Please choose the features you would like to use in training',cn)
    
    X_data=data[f_select]
    st.dataframe(X_data,use_container_width=True)
    from sklearn.preprocessing import LabelEncoder
    
    label=LabelEncoder()
    
    y=label.fit_transform(y)
    #st.dataframe(y.T)
    usr_miss=st.radio('Would you like to check for the missing values ?',['Yes','No'])
    
    if usr_miss=='Yes':
        missing_data=X_data.isnull().sum()
        missing_data.columns=['missing values']
        st.dataframe(missing_data,use_container_width=True)
    
    st.subheader('Filling the missing values')

    method=st.selectbox('Which method would you like to use for imputing?',['BackFill','ForwardFill','None'])

    if method=='BackFill':
    
        X_data.fillna(method='bfill',inplace=True)   
    elif method=='ForwardFill':
        X_data.fillna(method='ffill',inplace=True)
        
    st.dataframe(X_data,use_container_width=True)
    
    st.subheader('We will apply OneHot encoding to convert categorical features into numerical values ')
    X_data=pd.get_dummies(X_data)
    
    st.dataframe(X_data,use_container_width=True)
    
    
    eliminate_col=st.multiselect('You must drop unnecessary columns. \
                                 Please indicate the column you would like to drop.', X_data.columns)
    
    X_data.drop(eliminate_col,axis=1,inplace=True)
    
       
    x=st.slider('Decide the size of test data',min_value=0.1, max_value=0.4)
    
    from sklearn.model_selection import train_test_split
    
    X_train,X_test,y_train,y_test=train_test_split(X_data,y,test_size=x,random_state=10)
    
    col_names=X_train.columns
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scale=st.selectbox('Please select the scaling method',['StandardScaler','MinMax','None'])
    
    if scale=='StandardScaler':
            
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_train=pd.DataFrame(X_train,columns=col_names)
            X_test=scaler.transform(X_test)
    elif scale=='MinMaxScaler':
            scaler=MinMaxScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)
    
    st.dataframe(X_train,use_container_width=True)
    
           
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    rgrsr_name=['LogisticRegression','SupportVector','RandomForest','KNNeighbors']
    LogsRegression=LogisticRegression()
    SupportVector=SVC(probability=True)
    RandomForest=RandomForestClassifier()
    KNNeighbors=KNeighborsClassifier()
    
    model=[LogsRegression,SupportVector,RandomForest,KNNeighbors]
    rgrsr_list=['LogisticRegression','SupportVector','RandomForest', 'KNNeighbors']
    error_dict={}
    rgrsr=st.multiselect('Please choose the classifier(s) you would like to run',rgrsr_list)
    
    fig1,ax1=plt.subplots()
    
    fig0,ax0=plt.subplots()
    
    for n,cl in zip(rgrsr,model):
        cl.fit(X_train,y_train)
        y_predict=cl.predict(X_test)
        y_prob=cl.predict_proba(X_test)
        y_prob1=y_prob[:,1]
        y_prob0=y_prob[:,0]
        
        score1=roc_auc_score(y_test,y_prob1)
        
        fpr1,tpr1,threhold=roc_curve(y_test,y_prob1,pos_label=1)
        ax1.plot(fpr1,tpr1,label=n+' '+str(round(score1,2)))
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title('ROC Curve and Area Under the Curve')
        
        
        score0=roc_auc_score(y_test, y_prob0)
        fpr0,tpr0,threhold=roc_curve(y_test,y_prob0,pos_label=0)
        ax0.plot(fpr0,tpr0,label=n+' '+str( round(score0,2)))
        ax0.set_xlabel('FPR')
        ax0.set_ylabel('TPR')
        ax0.set_title('ROC Curve and Area Under the Curve')
        
        report=classification_report(y_test,y_predict)
        error_dict[n]=[report]
       
    #results=pd.DataFrame(error_dict)
    #results.index=['Error']
    st.subheader('Following table shows several possible measures of accuracy for classification problem')
    for i in error_dict.keys():
        st.write(i)
        st.write(error_dict[i])
    #st.write(results,use_container_width=True)
        
    st.subheader('ROC curve for each label') 
    classLabel=st.selectbox('Which label roc curve you would like to see?',[0,1])
    
    ax1.plot([0,0,1],[0,1,1],label='Best Curve 1.00')
    ax1.plot([0,1],[0,1],label='Random Guess 0.50')
    ax1.legend()
    
    ax0.plot([0,0,1],[0,1,1],label='Best Curve 1.00')
    ax0.plot([0,1],[0,1],label='Random Guess 0.50')
    ax0.legend()
    
    
    if classLabel==0:
        st.pyplot(fig0)
    else:
        st.pyplot(fig1)
        
    st.subheader('We have completed the tranning and you a have comprehensive\
              guideline to choose the best estimator for prediction purpose')    
    estimator=st.selectbox('Which estimator would you like to use for prediction?',rgrsr_list ) 
    
    cl=model[rgrsr_list.index(estimator)]
    
    cl.fit(X_train,y_train)
    features_set=X_train.columns
    feature_values=[]
    for feature in features_set:
        x=st.number_input(f"Please enter the value of {feature}",label_visibility="visible")
        feature_values.append(x)
    
    
    label=cl.predict([feature_values])
    
    st.write(f'The predicted label is {label}')

else:
     
    usr_choice=st.selectbox('Please choose one of the following',['Run Example','Upload Data'])
    try:
        if usr_choice=='Run Example':
            st.write('In this example we will study data of tips given to the waiter in a restaurants.')
            data=sns.load_dataset('tips') 
        else:
            data=st.file_uploader('Please choose your file')
            data=pd.read_csv(data)
    except ValueError:
           st.write('Please choose a valid data')
    
    st.dataframe(data,use_container_width=True)
    cn=data.columns
    
    target_name=st.selectbox('Please select a target', cn)
    y=data[target_name]
    
    st.dataframe(y,use_container_width=True)
    
    st.subheader("Explore features of the data")
    
    f_name=st.multiselect('Please select the feature', cn,default=cn[0])
    
        
    #X=pd.DataFrame(data.data, columns=data.feature_names)
    #y=data.target
    #st.dataframe(X)
    #f_name=st.multiselect('Distribution of features', data.feature_names,default=['AveRooms'])
    
    no_graphs=len(f_name)
    #st.write(no_graphs)
    fig,ax=plt.subplots(no_graphs,1,squeeze=False,figsize=(8,10))
    plt.tight_layout()
    for i in range(no_graphs):
        sns.histplot(data=data,x=f_name[i],kde=True,ax=ax[i,0])
    st.pyplot(fig)
    
    

    #no_graphs=len(f_name)
    #fig,ax=plt.subplots(no_graphs)
    #plt.tight_layout()
    #for i in range(no_graphs):
     #   sns.histplot(data=X,x=f_name[i],kde=True,ax=ax[i])
    #st.pyplot(fig)
    f_select=st.multiselect('Please choose the features you would like to use in training',cn)
    
    X_data=data[f_select]
    st.dataframe(X_data,use_container_width=True)
    #from sklearn.preprocessing import LabelEncoder
    
    #label=LabelEncoder()
    
    #y=label.fit_transform(y)
    #st.dataframe(y.T)
    usr_miss=st.radio('Would you like to check for the missing values ?',['Yes','No'])
    
    if usr_miss=='Yes':
        missing_data=X_data.isnull().sum()
        missing_data.columns=['missing values']
        st.dataframe(missing_data,use_container_width=True)
    
    st.subheader('Filling the missing values')

    method=st.selectbox('Which method would you like to use for imputing?',['BackFill','ForwardFill','None'])

    if method=='BackFill':
    
        X_data.fillna(method='bfill',inplace=True)   
    elif method=='ForwardFill':
        X_data.fillna(method='ffill',inplace=True)
        
    st.dataframe(X_data,use_container_width=True)
    
    st.subheader('We will apply OneHot encoding to convert categorical features into numerical values ')
    X_data=pd.get_dummies(X_data)
    
    st.dataframe(X_data,use_container_width=True)
    
    
    eliminate_col=st.multiselect('You must drop unnecessary columns. \
                                 Please indicate the column you would like to drop.', X_data.columns)
    
    X_data.drop(eliminate_col,axis=1,inplace=True)
    
       
    x=st.slider('Decide the size of test data',min_value=0.1, max_value=0.4)
    
    from sklearn.model_selection import train_test_split
    
    X_train,X_test,y_train,y_test=train_test_split(X_data,y,test_size=x,random_state=10)
    
    col_names=X_train.columns
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scale=st.selectbox('Please select the scaling method',['StandardScaler','MinMax','None'])
    
    if scale=='StandardScaler':
            
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_train=pd.DataFrame(X_train,columns=col_names)
            X_test=scaler.transform(X_test)
    elif scale=='MinMaxScaler':
            scaler=MinMaxScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)
    
    st.dataframe(X_train,use_container_width=True)


            
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
    rgrsr_name=['LinearRegression','SupportVector','RandomForest','KNNeighbors']
    LinRegression=LinearRegression()
    SupportVector=SVR()
    RandomForest=RandomForestRegressor()
    KNNeighbors=KNeighborsRegressor()

    model=[LinRegression,SupportVector,RandomForest,KNNeighbors]
    rgrsr_list=['Linear Regression','Support Vector','Random Forest', 'KNNeighbors']
    error_dict={}
    rgrsr=st.multiselect('Please choose the regressor you would like to run',rgrsr_list)

    for n,cl in zip(rgrsr_list,model):
        cl.fit(X_train,y_train)
        y_predict=cl.predict(X_test)
        error1=mean_squared_error(y_test,y_predict)
        error2=root_mean_squared_error(y_test,y_predict)
        error3=mean_absolute_error(y_test,y_predict)
        
        error_dict[n]=[error1,error2,error3]
        
    results=pd.DataFrame(error_dict)
    results.index=['MeanSquared','RootMeanSquared','MeanAbsolute']

    st.dataframe(results,use_container_width=True)
    
    st.subheader('We have completed the tranning and you a have comprehensive\
              guideline to choose the best estimator for prediction purpose.')    
    estimator=st.selectbox('Which estimator would you like to use for prediction?',rgrsr_list ) 
    
    cl=model[rgrsr_list.index(estimator)]
    
    cl.fit(X_train,y_train)
    features_set=X_train.columns
    feature_values=[]
    for feature in features_set:
        x=st.number_input(f"Please enter the value of {feature}",label_visibility="visible")
        feature_values.append(x)
    
    
    label=cl.predict([feature_values])
    
    st.write(f'The predicted value of the target variable is {label}')

        
       
        
        
        
        


    

