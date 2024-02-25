def test():
    list_file = listdir('testdata',[])
    for data in list_file:
        if '.las' not in data:
            continue
        spd = data.replace('.las','.spd')
        spd_pmfgrd = data.replace('.las','_pmfgrd.spd')
        dsm = data.replace('.las','_dsm.img')
        dtm = data.replace('.las','_dtm.img')
        chm = data.replace('.las','_chm_m.tif')
        comman(data,spd,spd_pmfgrd,dsm,dtm,chm)
        dtm_m = dtm.replace('.img','_m.tif')
        dsm_m = dsm.replace('.img','_m.tif')
        folder = data.split('/')
        folder = folder[:-1]
        outpath = folder
        outpath.append('result')
        outpath = '/'.join(outpath)
        os.mkdir(outpath)
        a=subprocess.call(['python','code/pycrownmaster/example/example.py',chm, dtm_m, dsm_m, data,outpath])
        if a == 1:
            continue
        folder = folder[:-1]
        merge(outpath, folder)
        if '08232021' in data or '11012021' in data:
            extract_single_tree_6(folder)
        else:
            extract_single_tree_4(folder)
    results = open('results_temp.csv','w',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Family','Genus','Species','Latitude','Longitude'])
    with open('dict/family_dict.pkl','rb') as f:
        family_dict = pickle.load(f)
    with open('dict/genus_dict.pkl','rb') as f:
        genus_dict = pickle.load(f)
    with open('dict/spices_dict.pkl','rb') as f:
        spices_dict = pickle.load(f)

    file_list = listdir('testdata',[])
    for line in file_list:
        if '/single_tree/' not in line:
            continue
        if '08232021' in data:
            flag = '_6_08_'
        elif '11012021' in data:
            flag = '_6_11_'
        else:
            flag = '_4_'
        x,lat,lon =load_data_test(line)
        pred_f = pred(x,'f',flag)
        f_name = family_dict[str(pred_f)]
        pred_g = pred(x,f_name,flag)
        g_name = genus_dict[str(pred_f)]
        pred_s = pred(x,g_name,flag)
        s_name = spices_dict[str(pred_f)]
        csv_writer.writerow([f_name,g_name,s_name,str(lat),str(lon)])
    csv_filter()
