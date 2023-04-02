import pandas as pd
import streamlit as st
from sklearn import svm
from sklearn.svm import SVC 
st.title('Model Deployment: SVM')
st.sidebar.header('User Input Parameters')
def user_input_features():
    industrial_risk=st.sidebar.selectbox('Indusrialrisk',(0.0,0.5,1.0),key='1')
    management_risk=st.sidebar.selectbox('Managementrisk',(0.0,0.5,1.0),key='2')
    financial_flexibility=st.sidebar.selectbox('FinancialFlexibility',(0.0,0.5,1.0),key='3')
    credibility=st.sidebar.selectbox('Credibility',(0.0,0.5,1.0),key='4')
    competitiveness=st.sidebar.selectbox('competitiveness',(0.0,0.5,1.0),key='5')
    operating_risk=st.sidebar.selectbox('operatingrisk',(0.0,0.5,1.0),key='6')
    data = {'industrial_risk':industrial_risk,
            ' management_risk':management_risk,
            ' financial_flexibility':financial_flexibility,
            ' credibility':credibility,
            ' competitiveness':competitiveness,
            ' operating_risk':operating_risk}
    features= pd.DataFrame(data,index=[0])
    return features

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)
bank=pd.read_csv('bankruptcy-prevention.csv')
bank['class_num'] = pd.factorize(bank[' class'])[0]
bank.drop([' class'],axis=1,inplace=True)
X=bank.iloc[:,:-1]
Y=bank.iloc[:,-1]
svm=SVC(kernel='rbf')
svm.fit(X,Y)
prediction = svm.predict(df)
st.subheader('Predicted result')
st.write('Bankruptcy' if prediction==0 else 'Non-Bankruptcy')
