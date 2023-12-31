# Laporan Proyek Machine Learning

### Nama : Fauzan Fadhillah Arisandi

### Nim : 211351055

### Kelas : Pagi B

## Domain Proyek

Grocery Store atau toko kelontong adalah toko ritel jasa makanan yang terutama menjual berbagai macam produk makanan, yang mungkin segar atau dikemas. Namun, dalam penggunaan sehari-hari di AS, "toko kelontong" adalah sinonim untuk supermarket, dan tidak digunakan untuk merujuk pada jenis toko lain yang menjual bahan makanan. Di Inggris, toko-toko yang menjual makanan dibedakan menjadi grocers atau grocery shops (walaupun dalam penggunaan sehari-hari, orang biasanya menggunakan istilah "supermarket" atau "corner store"[4] atau "toko serba ada").

## Business Understanding

Dikarnakan pelanggan terkadang bingung mencari barang yang ingin di beli , maka kami ingin memprediksi keseringan suatu barang di beli dengan mengetahui terlebih dahulu barang apa yang sebelum nya di beli. Sehingga kami ingin memindahkan barang yang kedua dekat dengan barang yang pertama.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

-   Pembeli kesulitan mencari barang yang ingin di beli
-   Toko tidak dapat mempermudah pelanggan di dalam mencari barang 

### Goals Dengan Solution Statements

Menjelaskan tujuan dari pernyataan masalah:

-   Membuat penelitian untuk dapat memprediksi barang selanjutnya yang akan di beli pelanggan (Menggunakan algorita apriori)
-   Mendapatkan wawasan pada barang yang biasa di beli bersamaan oleh pelanggan 

## Import Dataset

\[Groceries dataset for Market Basket Analysis(MBA)\]\[<https://www.kaggle.com/datasets/rashikrahmanpritom/groceries-dataset-for-market-basket-analysismba/data>)

Pertama-tama saya upload kaggle.json untuk memiliki akses pada kaggle

``` python
from google.colab import files
files.upload()
```
Selanjutnya membuat direktori dan permission pada skrip ini

``` python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Lalu mendownload dataset tersebut

``` python
!kaggle datasets download -d anubhavgoyal10/laptop-prices-dataset/
```
Mengunzip dataset

```python
!mkdir laptop-prices-dataset
!unzip laptop-prices-dataset.zip -d laptop-prices-dataset
!ls laptop-prices-dataset
```

## Import Library

Mengimpor Library yang dibutuhkan yakni matplotlib , seaborn ,pandas dan
numpy

``` python
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
```

## Data Discovery

Membaca data csv
``` python
df = pd.read_csv("groceries-dataset-for-market-basket-analysismba/Groceries data.csv")
bk = pd.read_csv("groceries-dataset-for-market-basket-analysismba/basket.csv")
df.head()
```
![image](https://github.com/Shelford21/FauzanML/assets/122199835/3c874ef8-84ee-43b7-b7d8-d796200e87e7)


Memeriksa berapa baris dan kolom

``` python

df.shape
```

Mengetahui deskripsi pada data

``` python

df.describe()
```



``` python
df.info()
```



### Variabel-variabel pada Laptop Prices Dataset adalah sebagai berikut:

-   Member_number : Merupakan nomor identitas pelanggan \[Bertipe:Integer ,Contoh: 1187,4941\]
-   Date : Merupakan tahun , bulan dan hari \[Bertipe:Date , Contoh: 2015-12-12\]
-   ItemDescription : Merupakan bahan groceries \[Bertipe:String , Contoh: Whole milk, Bread\]
-   year : Merupakan tahun \[Bertipe:Integer , Contoh: 2015\]
-   month : Merupakan bulan \[Bertipe:Integer, Contoh: 12\]
-   day : Merupakan hari \[Bertipe:Integer ,Contoh: 5,4\]
-   day_of_week : Merupakan hari ke berapa di minggu itu\[Bertipe:Integer ,Contoh: 5,4\]
  
## EDA

Mengetahui Top 40 Items 
``` python

freq_items = df['itemDescription'].value_counts()
freq_items.head(10)
fig = px.bar(data_frame=freq_items.head(40), title='Top 40 Items', color=freq_items.head(40),
                 labels={
                     "index": "Items",
                     "values": "Quantity",
                     'lift': 'Lift'
                 })
fig.update_layout(title_x=0.5, title_y=0.86)
fig.show()
```


Mengetahui Top 25 Groceries yang terbeli
```python
top_25=df.itemDescription.value_counts().sort_values(ascending=False)[0:25]
fig = px.bar(top_25,color=top_25.index, labels={'value':'Quantity Sold','index':'GroceryItems'})
fig.update_layout(showlegend=False, title_text='Top 25 Groceries Sold',title_x=0.5, title={'font':{'size':20}})
fig.show()
```


Mengetahui Bottom 25 groceries yang terbeli
```python
bot_25=df.itemDescription.value_counts().sort_values(ascending=False)[-25:]
fig = px.bar(bot_25,color=bot_25.index, labels={'value':'Quantity Sold','index':'GroceryItems'})
fig.update_layout(showlegend=False, title_text='Bottom 25 Groceries Sold',title_x=0.5, title={'font':{'size':20}})
fig.show()
```

Mengetahui Top 25 Customer
```python
top_25c = df.groupby('Member_number').agg(PurchaseQuantity=('itemDescription','count')).sort_values(by='PurchaseQuantity',ascending=False)[0:25]
top_25c.plot(kind='bar', figsize=(15,7), legend=None)
plt.title('Top 25 customers', fontsize=20)
plt.xlabel('Customer Number', fontsize=15)
plt.ylabel('Purchase Quantity', fontsize=15)
plt.show()
```

Mengetahui frekuensi item terjual
```python
item_freq = df.groupby(pd.Grouper(key='itemDescription')).size().reset_index(name='count')
fig = px.treemap(item_freq, path=['itemDescription'], values='count')
fig.update_layout(title_text='Frequency of the Items Sold', title_x=0.5, title_font=dict(size=18))
fig.update_traces(textinfo="label+value")
fig.show()
```

Mengetahui Top 10 produk terjual pada musim panas
```python
df_sum=df[(df['month']>1)&(df['month']<6)]
top_10s=df_sum.itemDescription.value_counts().sort_values(ascending=False)[0:10]
fig = px.bar(top_10s,color=top_10s.index, labels={'value':'Quantity Sold','index':'GroceryItems'})
fig.update_layout(showlegend=False, title_text='Top 10 Products Sold',title_x=0.5, title={'font':{'size':20}})
fig.show()
```





## Data Preprocessing

Sebelum data di jadikan model , data perlu di proses kembali dan di siapkan untuk permodelan. 

```python
baskets=df.groupby(['Member_number','itemDescription'])['itemDescription'].count().unstack()
baskets
```

Memeriksa berapa banyak fitur yang memiliki nilai null

``` python

baskets.notnull().sum()
```

Memberi nilai nol pada data yang memiliki nilai null atau NaN

``` python

baskets=baskets.fillna(0).reset_index()
baskets.head()
```

Mengkonversikan nilai yang <=0 menjadi 0 dan nilai yang >=1 menjadi 1

```python

def convert_values(value):
    if value <= 0:
        return 0
    elif value >=1:
        return 1
```

Menerepkan fungsi convert_values pada dataset

```python

baskets = baskets.iloc[:, 1:baskets.shape[1]].applymap(convert_values)
```

Membuat dataset baru yang sudah di perbarui dan sudah siap untuk di buatkan model

```python
df_new = pd.DataFrame(baskets)
```

## Modeling

Menerapkan algoritma apriori

``` python
freq_items = apriori(df_new, min_support=0.05, use_colnames=True, max_len=3).sort_values(by='support')
freq_items.head(10)
```


``` python
rules=association_rules(freq_items, metric="lift", min_threshold=1).sort_values('lift',ascending=False)
rules=rules[['antecedents','consequents','support','confidence','lift']]
rules.head()
```

``` python
rules['antecedents']=rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents']=rules['consequents'].apply(lambda a: ','.join(list(a)))
print(rules[['antecedents','consequents']])
```
## Visualisasi Hasil Algoritma

```python
support_table = rules.pivot(index='consequents', columns='antecedents', values='support')
support_table.shape
```

```python
fig=ff.create_annotated_heatmap(support_table.to_numpy().round(2),x=list(support_table.columns),y=list(support_table.index),colorscale=['violet','indigo','blue'],font_colors=['white','white','white'])
fig.update_layout(template='simple_white',
    autosize=False,
    width=1600,
    height=1600,
    title="Support Matrix",
    xaxis_title='Consequents',
    yaxis_title='Antecedents',
    legend_title="Legend Title",
    font=dict(
        family="Caliber",
        size=14,
        color="Black"
    )
)
fig.update_layout(title_x=0.22, title_y=0.98)
fig.update_traces(showscale=True)
fig.show()
```
support suatu item adalah sebagian kecil transaksi dalam kumpulan data transaksi yang memuatnya. support akan menjadi yang tertinggi jika kedua item tersebut disertakan dalam semua transaksi.

support tertinggi yaitu 0,19 dapat diamati antara susu murni dan sayuran lainnya yang menunjukkan bahwa 19% transaksi mengandung susu murni dan sayuran lainnya.

```python
conf=rules.pivot(index='antecedents', columns='consequents', values='confidence')
fig=ff.create_annotated_heatmap(conf.to_numpy().round(2),x=list(conf.columns),y=list(conf.index),colorscale=['green','orange','red'],font_colors=['white','white','white'])
fig.update_layout(template='simple_white',
    autosize=False,
    width=1600,
    height=1600,
    title="Confidence Matrix",
    xaxis_title='Consequents',
    yaxis_title='Antecedents',
    legend_title="Legend Title",
    font=dict(
        family="Caliber",
        size=14,
        color="Black"
    )
)
fig.update_layout(title_x=0.22, title_y=0.98)
fig.update_traces(showscale=True)
fig.show()
```
confidence tertinggi : 0,6 (antara sayuran lain/air minum kemasan dan susu murni)

confidence terendah : 0,12 (antara sayuran beku dan susu murni)

confidence yang tinggi menunjukkan bahwa kemungkinan besar jika satu produk dibeli, produk lainnya juga akan dibeli.

```python
lift_val=rules.pivot(index='antecedents', columns='consequents', values='lift')
fig=ff.create_annotated_heatmap(lift_val.to_numpy().round(2),x=list(lift_val.columns),y=list(lift_val.index),colorscale=['green','orange','red'],font_colors=['white','white','white'])
fig.update_layout(template='simple_white',
    autosize=False,
    width=1600,
    height=1600,
    title="Lift Matrix",
    xaxis_title='Consequents',
    yaxis_title='Antecedents',
    legend_title="Legend Title",
    font=dict(
        family="Caliber",
        size=14,
        color="Black"
    )
)
fig.update_layout(title_x=0.22, title_y=0.98)
fig.update_traces(showscale=True)
fig.show()
```
lift >1 : menunjukkan bahwa kehadiran item di sebelah kiri telah meningkatkan kemungkinan munculnya item di sebelah kanan dalam transaksi.

lift <1 : menunjukkan bahwa keberadaan barang-barang di sebelah kiri telah menurunkan kemungkinan munculnya barang-barang di sebelah kanan dalam transaksi.

lift = 1 : menandakan keberadaan item di kiri dan kanan bersifat independen.

Nilai lift tertinggi : 1,37 antara sayuran lainnya, susu murni dan air minum kemasan

Nilai lift terendah : 1,04 antara buah-buahan tropis dengan sayuran lainnya dan sebaliknya.


## Deployment
![image](https://github.com/Shelford21/FauzanML/assets/122199835/b937ecb4-82a6-4068-9c0d-bc3b03515314)

[linkStreamlit](https://fauzanml-ambition.streamlit.app/)
