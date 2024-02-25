import pandas as pd
from collections import Counter
df = pd.read_csv('Mobot_Trees_20210127_v2.csv',index_col=0)
result

df1 = df.groupby(['Family','Genus'])['Genus'].count()
df_count = df1.to_frame()

x = df_count['Genus'].groupby(['Family','Genus']).aggregate(['max','min'])
f = list(df['Family'])
ff = Counter(f)
ff = ff.most_common()
ff[0][1]
x['flag'] = x['max']-x['min']
x.loc()
df['tot'] = df['Family']+df['Genus']
count = df['Family'].value_counts(); # 返回series类型结果。通过series.index访问索引值，series.values访问值
for i in len(count.index):
    f =  count.index[i]
    df_f = df[df['Family'] == f]
    count_f = df["Family"].value_counts();

df_count = count.to_frame(); #count：series类型。"count.index：为其索引——数字类型，count.values：为其值"

df_count = pd.DataFrame(df_count.values.T, columns=df_count.index); #原先series的values类似，dataframe的某一列是纵向，而加入".T"便改为横向
df_count.to_csv('family.csv')
len(count.index)
df[df['Family'] == count.index[0]]
