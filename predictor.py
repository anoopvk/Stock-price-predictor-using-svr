import csv
import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt 


def get_csv_data(filename,datesindex,pricesindex):
    datecount=[]
    dates=[]
    prices=[]
    i=1
    with open(filename,'r') as csvfile:
        csvFileReader =csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            datecount.append(i)
            i+=1
            dates.append(int(row[datesindex].split('-')[0]))
            prices.append(float(row[pricesindex]))
    # return dates,prices
    return datecount,prices


def predict_prices(dates,prices,date_to_predict):
    dates=np.reshape(dates,(len(dates),1))
    date_to_predict=np.reshape(date_to_predict,(len(date_to_predict),1))
    # svr_linear=SVR(kernel="linear", C=1e3)
    # svr_polynomial=SVR(kernel="poly", C=1e3,degree=2)
    svr_rbf=SVR(kernel="rbf", C=1e3,gamma=0.1)
    # svr_linear.fit(dates,prices)
    # svr_polynomial.fit(dates,prices)
    svr_rbf.fit(dates,prices)

    # plt.scatter(dates,prices,color="black", label="original data")
    # plt.plot(dates,svr_linear.predict(dates),color="red",label="Linear model")
    # plt.plot(dates,svr_polynomial.predict(dates),color="green",label="polynomial model")
    # plt.plot(dates,svr_rbf.predict(dates),color="blue",label="rbf model")

    # plt.xlabel("date")
    # plt.ylabel("prices")
    # plt.title("svm")
    # plt.legend()
    # plt.show()


    # return svr_linear.predict(date_to_predict),svr_polynomial.predict(date_to_predict),svr_rbf.predict(date_to_predict)
    return svr_rbf.predict(date_to_predict)








dates,prices=get_csv_data("mydata_2.csv",0,1)
print(dates,prices)
testing_dataset_size=3
dates_to_train=dates[:-testing_dataset_size]
dates_to_test=dates[-testing_dataset_size:]
prices_to_train=prices[:-testing_dataset_size]
prices_to_test=prices[-testing_dataset_size:]

print("the entire data : ")
for i in range(len(dates)):
    print(dates[i],"---",prices[i])
# print(type(dates[0]))
# print(predict_prices(dates,prices,[i for i in range(250,300)]))
# result= predict_prices(dates,prices,[i for i in range(250,300)])
print("calculation started")
result= predict_prices(dates_to_train,prices_to_train,dates_to_test)
print("calculation completed")

for i in range(len(result)):
    print(dates_to_test[i],prices_to_test[i],"------",result[i],"diff = ",abs(prices_to_test[i]-result[i]))