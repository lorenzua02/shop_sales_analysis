import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from prophet import Prophet

#------------------------------------------------------------------------
#INIZIALIZZAZIONE
#------------------------------------------------------------------------
df = pd.read_csv("./data/train.csv", parse_dates=True, index_col='date')
transactions = pd.read_csv("./data/transactions.csv", parse_dates=True, index_col='date')
#Non ci sono valori NA

#------------------------------------------------------------------------
#DEFINIZIONE DEL NEGOZIO DA ANALIZZARE E PRE-PROCESSINF
#------------------------------------------------------------------------
df_store = df[df.store_nbr == 6]
df_store= pd.DataFrame(data= df.iloc[:, 3:])
df_store= df_store.groupby("date", as_index= True).sum()

transactions = transactions[transactions.store_nbr == 6].groupby("date", as_index= True).sum()
del transactions["store_nbr"]

df_tot= df_store.merge(transactions, on= "date", how= "left")
df_train = df_tot["2013":"2016"]

df_train = df_train.reset_index()
df_train.columns = ['ds', 'y', 'onpromotion', 'transactions']

df_test = df_tot["2017"]
df_test = df_test.reset_index()
df_test.columns = ['ds', 'y', 'onpromotion', 'transactions']

#------------------------------------------------------------------------
#DEFINIZIONE HOLIDAYS
#------------------------------------------------------------------------
hd = pd.read_csv("./data/holidays_events.csv", parse_dates=True, index_col='date')
#print(hd.isna().any()) #Non ce ne sono

t = hd[hd.locale == "National"]
t = t[
    (t.locale_name == "Ecuador") & 
    (t.type != "Work Day") & 
    (t.transferred == False) & 
    (t.type != "Additional") &
    (t.type != "Transfer")
]
t.drop(["transferred", "description", "locale_name", "locale"], axis=1, inplace=True)

res = []
a = ['11-02','11-03','12-25','05-01','08-10','01-01','05-24','10-09']
b = ["a", "b", "c", "d", "e", "f", "g", "h"]
for x in range(2013, 2017+1):
    for y in a:
        res.append(str(x)+"-"+y)
#le holiday si ripetono per i 5 anni del dataset
holidays = pd.DataFrame({'ds': res, 'holiday': b*5})

#------------------------------------------------------------------------
#DEFINIZIONE DEL MODELLO
#------------------------------------------------------------------------
model = Prophet(holidays=holidays, yearly_seasonality=5, seasonality_mode='multiplicative')
model.fit(df_train)
future = model.make_future_dataframe(periods=df_test.shape[0], include_history=False)
predictions = model.predict(future)
model.plot_components(predictions)

pred_ = predictions.yhat.sum()
test_ = df_test.y.sum()
diff_perc = abs(pred_/test_)*100
print(f"Somma vendite predette: {round(pred_,2)}\nSomma vendite effettive: {round(test_,2)}\nDiff: {round(abs(pred_-test_),2)}\nPerc accuratezza:{diff_perc}%")
print("MAE 2017: {}".format(mean_absolute_error(y_true=df_test.y, y_pred=predictions.yhat)))

predictions = predictions.set_index('ds')

#------------------------------------------------------------------------
#GRAFICO DELLA PREDIZIONE
#------------------------------------------------------------------------
plt.figure(figsize= (15, 8))
plt.plot(df_tot.index, df_tot["sales"], label= "Dati")
plt.plot(predictions.index, predictions["yhat"], label= "Previsioni")
plt.title("Confronto dati e predizioni")
plt.legend()

#------------------------------------------------------------------------
#ALTRI GRAFICI
#------------------------------------------------------------------------
#1
df_sales= pd.DataFrame(data= df["sales"])
df_sales= df_sales.groupby("date", as_index= True).sum()
figure, axis = plt.subplots(2, 3, figsize= (20, 10))
n_negozio= 1
for i in range(2):
    for e in range(3):
        df_i= df[df["store_nbr"]==n_negozio]
        df_i_sales= pd.DataFrame(data= df_i["sales"])
        df_i_sales= df_i_sales.groupby("date", as_index= True).sum()
        axis[i, e].plot(df_i_sales.index, df_i_sales["sales"], c="orange")
        axis[i, e].plot(df_i_sales.index, df_i_sales["sales"].rolling(365, min_periods= 1, center=True).mean(), c="red")
        axis[i, e].set_title(f"Negozio numero {n_negozio}")
        n_negozio= n_negozio+1
plt.show()
#si nota che all'inizio del 2015 c'è una riduzione delle vendite per tutti i negozi

#2
figure, axis = plt.subplots(2, 3, figsize= (20, 10))
n_negozio= 1
for i in range(2):
    for e in range(3):
        df_i= df[df["store_nbr"]==n_negozio]
        df_i_promo= pd.DataFrame(data= df_i["onpromotion"])
        df_i_promo= df_i_promo.groupby("date", as_index= True).sum()
        axis[i, e].plot(df_i_promo.index, df_i_promo["onpromotion"], c="red")
        axis[i, e].set_title(f"Negozio numero {n_negozio}")
        n_negozio= n_negozio+1
plt.show()

#3
oil_data= pd.read_csv("data/oil.csv", parse_dates=True, index_col='date')
oil_data= oil_data.dropna()
oil_data= oil_data.groupby("date", as_index= True).sum()
df_sales = df_sales.merge(oil_data, on='date', how='left')
plt.figure(figsize= (15, 8))
plt.plot(df_sales.index, np.log(df_sales["sales"]), label= "Vendite")
plt.plot(df_sales.index, np.log(df_sales["dcoilwtico"]), label= "Petrolio")
plt.title("Prezzo del petrolio e vendite dei negozi")
plt.legend()
plt.show()
#si deduce che il prezzo del petrolio non influenza il numero di vendite, tranne all'inizio del 2015

#4
#plot sul negozio 43(Esmeraldas) per vedere gli effetti del terremoto
plt.figure(figsize= (15, 8))
df_43= df[df["store_nbr"]==43]
df_43= df_43.groupby("date", as_index=True).sum()

plt.plot(np.arange(0, 8), df_43["2013-04-14":"2013-04-21"]["sales"], c="k", label="Gli altri anni")
plt.plot(np.arange(0, 8), df_43["2014-04-14":"2014-04-21"]["sales"], c="k")
plt.plot(np.arange(0, 8), df_43["2015-04-14":"2015-04-21"]["sales"], c="k")
plt.plot(np.arange(0, 8), df_43["2016-04-14":"2016-04-21"]["sales"], c="r", label= "2016")
plt.plot(np.arange(0, 8), df_43["2017-04-14":"2017-04-21"]["sales"], c="k")
plt.title("Vendite del negozio 43 dopo il terremoto di Esmeraldas")
plt.legend()
plt.show()

#5
plt.figure(figsize=(10, 10))
stores= pd.read_csv("data/stores.csv", index_col='city')
stores["cluster"]=1
stores_distribution= pd.DataFrame(stores["cluster"])
stores_distribution= stores_distribution.groupby("city", as_index= True).sum()
plt.pie(stores_distribution["cluster"], labels= stores_distribution.index, shadow=True)
plt.axis('equal')
plt.show()