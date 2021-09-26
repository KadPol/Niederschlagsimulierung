from scripts import dataio as dataio
from netCDF4 import Dataset
import numpy as np
import datetime


def cnn_lstm_data(areas,startyear, endyear):
    wholeData=[]
    wholeStreamflow=[]
    for area in areas:
        prcppath = './data_daymet/{}/{}_{}_prcp.nc'.format(area,area,startyear)
        shape=Dataset(prcppath,'r')
        data=np.empty((0,shape.variables['prcp'][:].shape[0],shape.variables['prcp'][:].shape[1],shape.variables['prcp'][:].shape[2],5))
        streamflow_data= np.empty((0,365))
        streamflow_path= './data_daymet/{}_streamflow_qc.txt'.format(area)
        streamflow = dataio.load_streamflow(streamflow_path)
        streamflow['streamflow']=streamflow['streamflow']*0.028316846592
        streamflow.drop(['qc_flag','gauge_id'], inplace=True, axis=1)
        counter=1
        for i in range(startyear,endyear):
            path = './data_daymet/{}/{}_{}_prcp.nc'.format(area,area,i)
            srad_path = './data_daymet/{}/{}_{}_srad.nc'.format(area,area,i)
            tmin_path = './data_daymet/{}/{}_{}_tmin.nc'.format(area,area,i)
            tmax_path = './data_daymet/{}/{}_{}_tmax.nc'.format(area,area,i)
            vp_path = './data_daymet/{}/{}_{}_vp.nc'.format(area,area,i)
            f=Dataset(path,'r')
            f_srad=Dataset(srad_path,'r')
            f_tmin=Dataset(tmin_path,'r')
            f_tmax=Dataset(tmax_path,'r')
            f_vp=Dataset(vp_path,'r')
            prcp = np.reshape(f.variables['prcp'][:], (counter,f.variables['prcp'][:].shape[0], f.variables['prcp'][:].shape[1], f.variables['prcp'][:].shape[2],1))
            srad = np.reshape(f_srad.variables['srad'][:], (counter,f.variables['prcp'][:].shape[0], f.variables['prcp'][:].shape[1], f.variables['prcp'][:].shape[2],1))
            tmin = np.reshape(f_tmin.variables['tmin'][:], (counter,f.variables['prcp'][:].shape[0], f.variables['prcp'][:].shape[1], f.variables['prcp'][:].shape[2],1))
            tmax = np.reshape(f_tmax.variables['tmax'][:], (counter,f.variables['prcp'][:].shape[0], f.variables['prcp'][:].shape[1], f.variables['prcp'][:].shape[2],1))
            vp = np.reshape(f_vp.variables['vp'][:], (counter,f.variables['prcp'][:].shape[0], f.variables['prcp'][:].shape[1], f.variables['prcp'][:].shape[2],1))
            prcp = np.append(prcp,srad,axis=4)
            prcp = np.append(prcp,tmax,axis=4)
            prcp = np.append(prcp,tmin,axis=4)
            prcp = np.append(prcp,vp,axis=4)
            data= np.append(data,prcp,axis=0)
            year='{}'.format(i)
            streamflow_year= np.array(streamflow['streamflow'][year][0:365])
            streamflow_year=np.reshape(streamflow_year,(1,len(streamflow_year)))
            streamflow_data =np.append(streamflow_data,streamflow_year,axis=0)
        wholeData.append(data)
        wholeStreamflow.append(streamflow_data)
    return wholeData,wholeStreamflow


def lstm_data(basin,start_date, end_date,days):
    forcing=[]
    streamflow=[]
    startdate_date = datetime.datetime.strptime(start_date, "%y-%m-%d").date()
    enddate_date = datetime.datetime.strptime(end_date, "%y-%m-%d").date()
    streamflow_startdate= startdate_date + datetime.timedelta(days=days)
    streamflow_enddate= enddate_date + datetime.timedelta(days=days)
    for i in basin:
        forcings_path = "./data/{}_lump_cida_forcing_leap.txt".format(i)
        streamflow_path = "./data/{}_streamflow_qc.txt".format(i)
        # Initialisierung der Daten
        df_forcings = dataio.load_forcings(forcings_path)
        df_forcings.drop(['swe(mm)','dayl(s)'], inplace=True, axis=1)
        forcings = df_forcings[startdate_date:enddate_date]
        df_streamflow = dataio.load_streamflow(streamflow_path)
        df_streamflow['streamflow']=df_streamflow['streamflow']*0.028316846592
        df_streamflow.drop(['qc_flag','gauge_id'], inplace=True, axis=1)
        streamflows = df_streamflow[streamflow_startdate:streamflow_enddate].to_numpy()
        forcing.append(forcings)
        streamflow.append(streamflows)
    return forcing,streamflow