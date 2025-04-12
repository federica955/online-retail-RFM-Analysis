import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#carico il dataset
data= pd.read_csv("Online_Retail.csv",encoding="ISO-8859-1")
print(data.head())

#PULIZIA DATASET
data.isnull().sum()
data=data.dropna()
data=data.drop_duplicates()

data=data[data["Quantity"]<0]

data.info()
data ["InvoiceDate"]=pd.to_datetime(data["InvoiceDate"],format="%m/%d/%y %H:%M")

#analisi esplorativa

data["Description"].unique()
data["Description"].value_counts().head(10)
data["CustomerID"].value_counts().head(10)
data["Country"].nunique()
data["Country"].unique()
data["Total"]=data["Quantity"]*data["UnitPrice"]
print("fatturato totale",data["Total"].sum())
data["Total"].sum

#creo colonna anno e mese
data["YearMonth"]=data["InvoiceDate"].dt.to_period("M")
sales_per_month=data.groupby("YearMonth")["Total"].sum()
#grafico
sales_per_month.plot(kind="line",figsize=(10,5))
plt.title("vendite totali per mese")
plt.ylabel("totale vendite (£)")
plt.xlabel("Data")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#top 10 prodotti più venduti
top_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

top_products.plot(kind='bar', figsize=(10,5))
plt.title("Top 10 Prodotti più venduti")
plt.ylabel("Quantità")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#fatturato x paese(escluso uk)
revenue_by_country = data[data['Country'] != 'United Kingdom'].groupby('Country')['Total'].sum().sort_values(ascending=False).head(10)

revenue_by_country.plot(kind='bar', figsize=(10,5))
plt.title("Top 10 Paesi per Fatturato (escluso UK)")
plt.ylabel("Fatturato (£)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#RFM
import datetime as dt

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])  # già fatto ma lo assicuriamo
snapshot_date = data['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',   # Frequency
    'Total': 'sum'            # Monetary
})
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Total': 'Monetary'
}, inplace=True)
print(rfm.head())

# Recency inverso (1 = cliente dormiente, 5 = recentissimo)
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

# Frequency: 1 = ha comprato poco, 5 = molto spesso
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

# Monetary: 1 = ha speso poco, 5 = spende tanto
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
rfm['Segment'] = 'Altro'
rfm['R_score'] = rfm['R_score'].astype(int)
rfm['F_score'] = rfm['F_score'].astype(int)
rfm['M_score'] = rfm['M_score'].astype(int)
rfm.loc[(rfm['R_score'] == 5) & (rfm['F_score'] >= 4), 'Segment'] = 'Clienti Recenti e Fedeli'
rfm.loc[(rfm['R_score'] == 5) & (rfm['M_score'] >= 4), 'Segment'] = 'Clienti Recenti e Redditizi'
rfm.loc[(rfm['F_score'] == 5) & (rfm['M_score'] == 5), 'Segment'] = 'Top Spender Frequenti'
rfm.loc[(rfm['R_score'] == 1) & (rfm['F_score'] <= 2), 'Segment'] = 'Clienti Persi'
rfm['Segment'].value_counts().plot(kind='bar', figsize=(9,5))
plt.title("Distribuzione dei Segmenti Clienti")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
vip = rfm[rfm['RFM_score'] == '555']
print(vip.head())
print("Clienti VIP totali:", vip.shape[0])
rfm['Segment'] = 'Altro'
rfm.loc[rfm['RFM_score'] == '555', 'Segment'] = 'Cliente VIP'
rfm.loc[rfm['R_score'] == '5', 'Segment'] = 'Recenti'
rfm.loc[rfm['F_score'] == '5', 'Segment'] = 'Frequenti'
rfm.loc[rfm['M_score'] == '5', 'Segment'] = 'Spendaccioni'
print(rfm['Segment'].value_counts())


rfm['Segment'].value_counts().plot(kind='bar', figsize=(8,5))
plt.title("Distribuzione dei Segmenti RFM")
plt.ylabel("Numero di clienti")
plt.xlabel("Segmento")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
rfm.to_csv("RFM_Analysis.csv")