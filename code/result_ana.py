import geopandas
import pandas as pd
# 读文件
gdf = geopandas.GeoDataFrame.from_file("TowerGroveTreesShifted/TowerGrove_Trees_Shifted.shp", encoding='gb18030')
gdf_1 = pd.read_csv('Mobot_Trees_20210127_v2.csv',index_col=0)
# 显示
gdf  #输出属性表
gdf_1

a = []
count = gdf['Genus'].value_counts(); # 返回series类型结果。通过series.index访问索引值，series.values访问值
for i in range(len(count.index)):
    f =  count.index[i]
    df_f = gdf_1[gdf_1['Genus'] == f]
    count_f = df_f["Genus"].value_counts()
    if len(count_f.index) == 0:
        a.append([f,count.values[i],0])
    else:
        a.append([f,count.values[i],count_f.values[0]])

a
# 保存
df_f = pd.DataFrame(columns=['Genus', 'label_count', 'train_count'], data=a)
# index=False表示存储csv时没有默认的id索引
df_f.to_csv("result_g.csv", encoding='utf-8', index=False)
