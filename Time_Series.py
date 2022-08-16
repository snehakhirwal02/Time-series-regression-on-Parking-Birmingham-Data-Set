# # Data Visualization
# Import necessary libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from typing import NamedTuple
import math

import streamlit as st


# User Inputs:
                # Syscodenumber
                # Date
                # Time

# Output:
                # R2 -score
                # RMSE
                # Plot of predicted and optimal Value
                # Occupancy Rate 
                # Occupancy for a particular systemcodenumber at a given time and date

# Methods():
                # DataPrep(): Occupancy rate and binning for the minute column obtained from the given time
                # fullmodel(): Initialization of features and target for the whole dataset
                # model(): fitting and prediction on the whole training and test dataset
                # SelectedSysCode(): Initialization of features and target on the basis of filtered dataframe accordingly to the given systemcodenumber 
                # UserSelectmodel():
                # Results(): To display the R2 score, RMSE and the prediction plot

class Timeseries:
    """Class Timeseries: Random forest Regressor for time series implementation
    This Class contains the methods for Random forest Regressor for time series implementation
    Class input parameters:
    :param df: The input data frame
    :type df: Pandas DataFrame
    :param estimator: Number of decision trees to be build before taking the average of all predictions to be specified by the user
    :type estimator: Integer
    :param test_size: User Input - Proportion of test data specified by user in which dataset is to be splitted.
    :type test_size: float
    Class Output Parameters:
    :param Y_pred: The resulting output of the Regression test
    :type Y_pred: float 
    :param Y_test: The expected output of the Regression test
    :type Y_test: float  
    :param R-squared score: Model accuracy on the Training data
    :type : float 
    :param RMSE: Root mean squared error   
    :type RMSE: float
    :param Target: Occupancy to be predicted at a user specified time.
    :type Target: int
    :param Error_message: Error message if an exception was encountered during the processing of the code
    :type Error_message: str 
    :param flag: internal flag for marking if an error occurred while processing a previous method
    :type flag: bool 
       
    """
    flag=False
    Error_message='No errors during processing'
    # instance attribute
    def __init__(self,df,Input,base,target,time_col,sysCodeNo):
        """Class Constructor
        :param df: dataframe
        :type df: pandas dataframe
        :param Input: user inputs 
        :type Input: tuple
        :param base: Capacity of the parking lot
        :type base: array
        :param target: Target variable that is to be classified
        :type target: array
        :param time_col: Time Column 
        :type time_col: datetime
        :param sysCodeNo: System Code Number
        :type sysCodeNo: array
        
        """
        self.df=df
        self.Input = Input
        self.base=base
        self.target=target
        self.time_col=time_col
        self.sysCodeNo=sysCodeNo

    #
    def model(self,estimator, test_size):
        """ model Method : 
            This method splits the data into train and test sets, then creates a model based on the user input n_estimator and test_size. 
            
            It calls model 'RandomizedSearchoptim' that returns the best parameters on which the model can be fitted.
            
            It then fits the model based on the best parameters obtained after Randomized search cross validation and test it on the test dataset, then returns the predicted value 'Y_pred' 
            
            :param estimator: User Input - Number of decision trees to be build before taking the average of all prediction.
            :type estimator: Integer
            :param test_size: User Input - Proportion of test data specified by user in which dataset is to be splitted.
            :type test_size: float
            :return: Modified set of class parameters
        """
        if self.flag!=True:
            try:
                self.estimator=estimator 
                self.test_size=test_size  
                # Split the data into training set and testing set
                X_train,X_test,Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=test_size,random_state=123)

                # Create a model
                reg=RandomForestRegressor(n_estimators=estimator)

                c = self.RandomizedSearchoptim()
                # st.write(c)

                rf_reg=RandomizedSearchCV(estimator= reg ,param_distributions = c, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

                # Fitting training data to the model
                rf_reg.fit(X_train,Y_train)
                self.Y_pred = rf_reg.predict(X_test)

                # st.write(self.Y_test, self.Y_pred)
            except Exception as e:
                self.Error_message = 'Error while creating model: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)
                self.Y_pred=[]
                self.Y_test=[]
       
            return(self.Y_pred,self.Y_test)
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')     


        
    def UserSelectedmodel(self,estimator, test_size):
        """ UserSelectedmodel Method: 
            This method splits the data into train and test sets, then creates a model based on the user input n_estimator,test_size and the specific system code number.
            
            It calls model 'RandomizedSearchoptim' that returns the best parameters on which the model can be fitted.
            
            It then fits the model based on the best parameters obtained after Randomized search cross validation and test it on the test dataset, then returns the predicted value 'Y_pred' i.e Occupancy at a user specified date and time. 
            :param estimator: User Input - Number of decision trees to be build before taking the average of all prediction.
            :type estimator: Integer
            :param test_size: User Input - Proportion of test data specified by user in which dataset is to be splitted.
            :type test_size: float
        
        """
        if self.flag !=True:
            try:  
                self.estimator=estimator 
                self.test_size=test_size  
                # Split the data into training set and testing set
                X_train1,self.X_test1,Y_train1,self.Y_test1 = train_test_split(self.X1,self.Y1,test_size=test_size,random_state=123)
                # st.write(X_train1,Y_train1)

                # Create a model
                reg=RandomForestRegressor(n_estimators=estimator)
                b = self.RandomizedSearchoptim()
                
                rf_reg=RandomizedSearchCV(estimator= reg ,param_distributions = b, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

                # Fitting training data to the model
                rf_reg.fit(X_train1,Y_train1)
                self.Y_pred1 = rf_reg.predict(self.X_testIP)
                # self.Y_pred1 = round(self.Y_pred1,0)
                
                st.write( 'Rate: ', self.Y_pred1, '%')
                # df.to_excel('Filtered1.xlsx', sheet_name='Filtered Data')

                Total = self.df[self.base].unique()
                o = (Total * self.Y_pred1)/100
                st.write('Predicted ',self.target,' equals', math.floor(o))
                st.write('on given date and time') 
            except Exception as e:
                self.Error_message = 'Error while creating single user model: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')     

    def RandomizedSearchoptim(self):
        """ RandomizedSearchoptim Method : 
                This method returns the best parameters on which the model is to be fitted.
                Parameters:  
                    max_features:      Number of features to consider at every split
                    max_depth :        Maximum number of levels in tree
                    min_samples_split: Minimum number of samples required to split a node
                    in_samples_leaf:   Minimum number of samples required at each leaf node
                    bootstrap:         Method of selecting samples for training each tree
                :return: Best parameter values on which regressor is fitted.
        """
        try:

            # Find the best parameters for the model
            max_features = ['auto', 'sqrt']                                                  # Number of features to consider at every split
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]                     # Maximum number of levels in tree
            max_depth.append(None)
            min_samples_split = [2, 5, 10]                                                   # Minimum number of samples required to split a node
            min_samples_leaf = [1, 2, 4]                                                     # Minimum number of samples required at each leaf node
            bootstrap = [True, False]                                                        # Method of selecting samples for training each tree
        
            # Create the random grid
            random_grid = {'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap
            }
        except Exception as e:
            self.Error_message = 'Error while searching for random grid: ' +str(e)
            self.flag=True
            st.warning(self.Error_message)
            random_grid=[]
        return random_grid

    def DataPrep(self):
        """ DataPrep Method : 
                This method includes all the preprocessing of data required for the regressor model. It extracts the year, month, day, hours and minutes from the time column.
                It also calculates the rate based on target and base column. 
                The minutes column of the given dataset have been binned in an interval of 15 mins for better prediction of the target value.
        """
        try:
        # Convert LastUpdated column into Time and Date column. Add new columns for occupancy rate in percentage and Day of Week.
            self.df['Rate'] = round((100.0*self.df[self.target])/self.df[self.base],1)
            nxm=np.shape(self.df)
            n=nxm[0]
            m=nxm[1]
            date_time_column=self.df[self.time_col]
            X=np.ndarray(shape=(n,6),dtype=float, order='F')
            
            for a in range(0,n):

                date_time_obj = date_time_column.iloc[a]
                year=date_time_obj.year
                month=date_time_obj.month
                day=date_time_obj.day
                hour=date_time_obj.hour
                minute=date_time_obj.minute
                second=date_time_obj.second
                X[a,0]=year
                X[a,1]=month
                X[a,2]=day
                X[a,3]=hour
                X[a,4]=minute
                X[a,5]=second

            df_internal=pd.DataFrame(data=X,columns=["year","month","day", "hour","minute","second"])
            self.df.reset_index(drop=True, inplace=True)
            df_internal.reset_index(drop=True, inplace=True)
            self.df=pd.concat([self.df,df_internal],axis=1)
            

            #Categorizing the occupancy rate: Bining
            bins = [0,14,29,44,59]
            labels =[1,2,3,4]
            self.df['newmin'] = pd.cut(self.df['minute'], bins,labels=labels,include_lowest= True)
        except Exception as e:
            self.Error_message = 'Error while preparing data: ' +str(e)
            self.flag=True
            st.warning(self.Error_message)

    def fullmodel(self):
        """ fullmodel Method : 
        
                This method includes the initialization of features and target for training and testing the regressor model on the whole dataset.
        """
        if self.flag != True:
            try:
            # To initialize the features and self.target 
                self.X = self.df.loc[:,[self.base,'year','month','day','hour','newmin']]
                #self.X.to_csv('DatasetYY2.csv')
                self.Y= self.df['Rate'] 
            except Exception as e:
                self.Error_message = 'Error while creating full model: ' +str(e)
                self.flag=True
                st.warning(self.Error_message) 
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')     
    
    def SelectedSysCode(self,Input):
        """ DataPrep Method : 
                This method includes all the preprocessing of data required for the regressor model for the user input data. It extracts the year, month, day, hours and minutes from the time column.
                The minutes column of the user specified time have been binned in an interval of 15 mins for better prediction of the target value.
                The data frame is filtered according to the user specified syscodeno(system code number) in order to obtain the target value for the user specified system code number.
                Last but not the least the method also initializes the features and target column.
        """
        if self.flag!=True:

            try:
                # Grouping the minute column into two bins(one being 0-30 mins and other being 30-60 mins)
                if((Input.Time.minute >= 0) & (Input.Time.minute < 15)):
                    newmin = 1
                elif((Input.Time.minute >= 15) & (Input.Time.minute < 30)):
                    newmin = 2
                elif((Input.Time.minute >= 30 ) & (Input.Time.minute < 45)):
                    newmin = 3
                else:
                    newmin = 4       

                self.df = self.df[self.df[self.sysCodeNo] == Input.Syscodenumber] #Filtering dataframe according to user input
                
                self.df.drop(self.df.filter(regex="Unname"),axis=1, inplace=True)           
                
                self.X_testIP = pd.DataFrame(columns=['year','month','day','hour','newmin'])
                self.X_testIP.loc[0, ['year']]     = Input.Date.year 
                self.X_testIP.loc[0, ['month']]    = Input.Date.month
                self.X_testIP.loc[0, ['day']]      = Input.Date.day
                self.X_testIP.loc[0, ['hour']]     = Input.Time.hour
                self.X_testIP.loc[0, ['newmin']]   = newmin

                # To initialize the features and self.target 
                self.X1 = self.df.loc[:,['year','month','day','hour','newmin']]

                self.Y1= self.df['Rate']
            except Exception as e:
                self.Error_message = 'Error while selecting system code: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)   
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')     
    
    def Results (self):
        """ Results Method :
            This method displays metrics 'R-squared score' and 'RMSE - Root Mean Squared Error' value to analyze the performace of model,
            
            R-Squared Score: It is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. 
            RMSE: Mean Squared Error represents the average of the squared difference between the original and predicted values in the data set. 
            It measures the variance of the residuals. Root Mean Squared Error is the square root of Mean Squared error. It measures the standard deviation of residuals.
         
            This method prints a plot that represents the real prediction and optimal prediction. 
            Optimal prediction are the true values that are plotted as a line plot and real prediction are values predicted by the classification model that are plotted as a scatter plot.
        """
        
        
        if self.flag !=True:
                
            try:
                # R-Square value
                r2 = r2_score(self.Y_test, self.Y_pred)
                st.write('R-squared score of Predicted model: ', round(r2, 5))
                st.write('RMSE of Predicted model: ',MSE(self.Y_test,self.Y_pred)**(0.5))
                
                # To display the 2D-plot for the actual vs predicted values
                df1=pd.DataFrame({'Y_Actual':self.Y_test,'Y_Pred':self.Y_pred})
                fig = plt.figure(figsize=(10, 4))
                sns.scatterplot(x='Y_Actual',y='Y_Pred',data=df1,label='Real Prediction')
                sns.lineplot(x='Y_Actual',y='Y_Actual',data=df1,color='red',alpha=0.5,label='Optimal Prediction')
                plt.title('Y_Actual vs Y_Pred')
                plt.legend()
                st.pyplot(fig)
            except Exception as e:
                self.Error_message = 'Error while outputing results: ' +str(e)
                self.flag=True
                st.warning(self.Error_message)   
        else:
            st.write('Error occurred in previous methods, Refer to Error Message Warning')     
        

class rf_Inputs(NamedTuple):
    """Class rf_Inputs: 
    This Class comprises the inputs provided by the user.There are three user inputs namely system code number, date and time based on which the target value is determined.
    
    Class input parameters:
    """
    Syscodenumber:  str;""" System code number """
    Date:           datetime;""" Date """
    Time:           datetime;""" Time """
