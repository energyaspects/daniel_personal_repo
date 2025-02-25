from helper_functions_ea.shooju_helper import ShoojuTools
from helper_functions_ea import check_env, ShoojuTools, Logger
from datetime import datetime, date, timedelta
sj = ShoojuTools()
from helper_functions_ea import check_env
check_env()
sj = ShoojuTools()
import numpy as np
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, peak_prominences,peak_widths, chirp
from plotly.subplots import make_subplots
from scipy import integrate
import os
import plotly.express as px
import re

import warnings
warnings.filterwarnings('ignore')

logger = Logger("ScedScroller").logger  # Creates a logger
shooju_job = sj.register_and_check_job(job_name = f"battery model {datetime.today().strftime('%Y-%m-%d-%H')}")

class battery_model:

    def __init__(self):
        self.ISO = "ERCOT"
        self.sd = datetime(2024, 8, 1)
        self.ed = datetime(2024, 8, 31)
        self.resolution = "5min"  #options are "60min", "15min", "5min"
        resolution_to_interval_map = {"60min": 1,"15min": 4,"5min": 12}
        self.resolution_interval = resolution_to_interval_map.get(self.resolution, "String not found")
        self.width_slices = 0.01 #DO NOT CHANGE FOR NOW
        self.round_trip_losses_percentage = 0.15
        self.plot_charts = False
        self.runner()

    #Extractors
    def get_thermal_call_sid_data(self):

        #Define thermal call SID
        thermal_call_query = rf'sid=teams\power\models\na_gen\{self.ISO}\generation_parameters\thermal_call@localize:America/Chicago' #call this thermal call in the future

        #Pull data from SJ
        thermal_call = sj.sj.get_df(query=thermal_call_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")

        #Clean up index and add other date columns
        thermal_call.index = pd.to_datetime(thermal_call.index)
        thermal_call.index = thermal_call.index.tz_localize(None)
        thermal_call.columns = ["thermal_call"]
        thermal_call["month"] = thermal_call.index.month
        thermal_call["year"] = thermal_call.index.year
        thermal_call["day"] = thermal_call.index.day
        thermal_call['hour'] = thermal_call.index.hour
        thermal_call['date'] = thermal_call.index.date
        thermal_call.reset_index(inplace=True)

        return thermal_call

    def get_battery_capacity_sid_data(self):

        battery_capacity_query = r'=({{sid=teams\power\models\na_gen\ERCOT\capacity\fuel\bess}}*1000)'
        battery_capacity = sj.sj.get_df(query=battery_capacity_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")
        battery_capacity.index = pd.to_datetime(battery_capacity.index)
        battery_capacity.index = battery_capacity.index.tz_localize(None)
        battery_capacity.columns = ["battery_capacity"]
        battery_capacity["month"] = battery_capacity.index.month
        battery_capacity["year"] = battery_capacity.index.year

        return battery_capacity

    def get_battery_duration_sid_data(self):

        duration_sid = fr'sid=users\daniel.pyrek\power\models\na_gen\{self.ISO}\capacity_adj\battery_duration'
        battery_duration = sj.sj.get_df(query=duration_sid, series_axis="columns", df=self.sd, dt=self.ed,max_points=-1, fields="series_id")
        battery_duration.columns = ["duration"]
        battery_duration['year'] =battery_duration.index.year
        battery_duration['month'] = battery_duration.index.month
        return battery_duration

    def get_modeled_battery_sid_data(self):

        modeled_bess_sid = fr'sid=users\daniel.pyrek\power\models\na_gen\{self.ISO}\generation_forecasts\batteries'
        modeled_bess = sj.sj.get_df(query=modeled_bess_sid, series_axis="columns", df=self.sd, dt=self.ed,max_points=-1, fields="series_id")
        modeled_bess.columns = ["modeled_bess_output"]
        modeled_bess['year'] = modeled_bess.index.year
        modeled_bess['month'] = modeled_bess.index.month
        modeled_bess.index = modeled_bess.index.tz_localize(None)
        return modeled_bess

    def get_battery_ancillary_sid_data(self):

        battery_as_vol_sid = fr'sid=users\daniel.pyrek\power\models\na_gen\{self.ISO}\capacity_adj\battery_ancillary_capacity'
        battery_as_vol = sj.sj.get_df(query=battery_as_vol_sid, series_axis="columns", df=self.sd, dt=self.ed,max_points=-1, fields="series_id")
        battery_as_vol.columns = ["avg_bess_as_vol"]
        battery_as_vol['year'] = battery_as_vol.index.year
        battery_as_vol['month'] = battery_as_vol.index.month

        """
        if self.ISO == "ERCOT":
            #We have actual ancillary service data for batteries so can pull that

            #ERCOT 60 day sced data sids
            battery_regdn_vol_query = r'''=F.multi_sum(r'sid:ERCOT\60dsced\* source_obj.aspect= ("Ancillary Service REGDN") source_obj.resource_type="PWRSTR"',operators=("@localize:America/Chicago"),key_fields=[])'''
            battery_regup_vol_query = r'''=F.multi_sum(r'sid:ERCOT\60dsced\* source_obj.aspect= ("Ancillary Service REGUP") source_obj.resource_type="PWRSTR"',operators=("@localize:America/Chicago"),key_fields=[])'''
            battery_ecrs_vol_query = r'''=F.multi_sum(r'sid:ERCOT\60dsced\* source_obj.aspect= ("Ancillary Service ECRS") source_obj.resource_type="PWRSTR"',operators=("@localize:America/Chicago"),key_fields=[])'''
            battery_rrs_vol_query = r'''=F.multi_sum(r'sid:ERCOT\60dsced\* source_obj.aspect= ("Ancillary Service RRS") source_obj.resource_type="PWRSTR"',operators=("@localize:America/Chicago"),key_fields=[])'''
            battery_rrsffr_vol_query = r'''=F.multi_sum(r'sid:ERCOT\60dsced\* source_obj.aspect= ("Ancillary Service RRSFFR") source_obj.resource_type="PWRSTR"',operators=("@localize:America/Chicago"),key_fields=[])'''

            #Pull data
            battery_regdn_vol = sj.sj.get_df(query=battery_regdn_vol_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")
            battery_regdup_vol = sj.sj.get_df(query=battery_regup_vol_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")
            battery_ecrs_vol = sj.sj.get_df(query=battery_ecrs_vol_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")
            battery_rrs_vol = sj.sj.get_df(query=battery_rrs_vol_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")
            battery_rrsffr_vol = sj.sj.get_df(query=battery_rrsffr_vol_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")

            #Sum to total participation and add other time columns
            battery_as_vol = battery_regdn_vol + battery_regdup_vol + battery_ecrs_vol + battery_rrs_vol + battery_rrsffr_vol
            battery_as_vol.columns = ["avg_bess_as_vol"]
            battery_as_vol["year"] = battery_as_vol.index.year
            battery_as_vol["month"] = battery_as_vol.index.month
            battery_as_vol["day"] = battery_as_vol.index.day
            battery_as_vol["hour"] = battery_as_vol.index.hour
            battery_as_vol = battery_as_vol.groupby(['year','month','day','hour'])["avg_bess_as_vol"].mean().reset_index()
            battery_as_vol["avg_bess_as_vol"] = battery_as_vol["avg_bess_as_vol"]/1000
            
            """

        return battery_as_vol

    def get_actual_battery_output_sid_data(self):

        battery_gen_query = r'''=F.multi_sum(r'sid:ERCOT\60dsced\* source_obj.aspect= ("Telemetered Net Output") source_obj.resource_type="PWRSTR"',operators=("@localize:America/Chicago@A:h"),key_fields=[])'''
        battery_gen = sj.sj.get_df(query=battery_gen_query, series_axis="columns", df=self.sd, dt=self.ed,max_points=-1, fields="series_id")
        battery_gen.index = battery_gen.index.tz_localize(None)
        battery_gen.columns = ["battery_gen"]

        battery_charge_query = r'=({{sid=teams\power\ercot\gen_mix\WSL}}*1000)@A:h@localize:America/Chicago'
        battery_charge = sj.sj.get_df(query=battery_charge_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1,fields="series_id")
        battery_charge.index = battery_charge.index.tz_localize(None)
        battery_charge.columns = ["battery_charge"]

        battery_data = battery_charge.join(battery_gen)
        battery_data["battery_net_output"] = battery_data["battery_charge"] + battery_data["battery_gen"]
        #battery_data.reset_index(inplace=True)

        """
        #Define the region
        if self.ISO == "ERCOT":
             battery_gen_query = r'=(({{sid=teams\power\ercot\gen_mix\WSL}} + {{sid=teams\power\ercot\gen_mix\other}})*1000)@A:h@localize:America/Chicago'

        #Pull the data
        battery_gen_data = sj.sj.get_df(query=battery_gen_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1,fields="series_id")
        battery_gen_data.columns = ["actual_battery_gen"]
        battery_gen_data.index = battery_gen_data.index.tz_localize(None)
        battery_gen_data.reset_index(inplace=True)
        """

        return battery_data

    #Function that runs the extractors and merges them
    def Extract_Merge(self):

        modeled_bess_data = self.get_modeled_battery_sid_data()
        logger.info(f'Pulled {len(modeled_bess_data)} rows of battery duration data')

        actual_battery_output_data = self.get_actual_battery_output_sid_data()
        logger.info(f'Pulled {len(actual_battery_output_data)} rows of actual battery output data')

        battery_capacity_data = self.get_battery_capacity_sid_data()
        logger.info(f'Pulled {len(battery_capacity_data)} rows of battery capacity data')

        battery_duration_data = self.get_battery_duration_sid_data()
        logger.info(f'Pulled {len(battery_duration_data)} rows of battery duration data')

        battery_ancillary_data = self.get_battery_ancillary_sid_data()
        logger.info(f'Pulled {len(battery_ancillary_data)} rows of battery ancillary data')

        #Merge everything together
        df = modeled_bess_data.join(actual_battery_output_data)
        df.reset_index(inplace=True)
        df = df.merge(battery_capacity_data)
        df = df.merge(battery_duration_data)
        df = df.merge(battery_ancillary_data)
        logger.info(f'Merged all input data')

        #Calculate additional columns
        df["battery_gwh"] = df['duration'] * df['battery_capacity'] #calculate raw battery gwh
        df["arb_bess_capacity"] = df["battery_capacity"] - df["avg_bess_as_vol"] #calculate battery capacity able to arb
        df["arb_bess_gwh"] = df["battery_gwh"] - df["avg_bess_as_vol"] * df["duration"] #calculate battery arb gwh

        #Clean up data and save df to self
        df.set_index("index",inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]  # remove duplicates as daylight change with extra hour breaks code
        df["date"] = df.index.date
        self.input_data = df
        logger.info(f'Completed data ingestion')



    def runner(self):

        # Pull all input data
        self.Extract_Merge()
        df = self.input_data

        ############### Plot comparison ###############

        # Specify the date range you want to display
        start_date = '2024-08-16'
        end_date = '2024-08-24'

        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add the main line plot
        fig.add_trace(go.Scatter(x=df.index, y=df["battery_net_output"], mode='lines',
                                 name='battery_net_output', line=dict(color='rgba(0,71,171,1)')),
                      secondary_y=False)

        # Add the main line plot
        fig.add_trace(go.Scatter(x=df.index, y=df["modeled_bess_output"], mode='lines',
                                 name='modeled_bess_output', line=dict(color='rgba(250,10,10,1)')),
                      secondary_y=False)

        # Update layout
        fig.update_layout(title='Battery model vs actuals',
                          xaxis_title='Datetime',
                          yaxis_title='GW',
                          # yaxis2_title='Battery Charge Factor',
                          xaxis=dict(
                              range=[start_date, end_date],  # Set the initial date range
                              type='date'  # Ensure the x-axis is treated as a date axis
                          ),
                          )

        # Show the plot
        fig.show()

        ############### Create figure comparing actual daily cyling vs installed capacity ###############

        #Find actual battery charging and discharging during modeled peaks
        positive_net_output = df.loc[df["battery_net_output"]>0]
        daily_positive_net_output = positive_net_output.groupby('date')[['battery_net_output']].sum()

        daily_discharge_volume = df.groupby('date')[['battery_gen']].sum()
        daily_discharge_volume.columns = ["battery_discharge"]

        daily_charge_volume = df.groupby('date')[['battery_charge']].sum()
        daily_charge_volume.columns = ["battery_charge"]

        daily_df = daily_discharge_volume.join(daily_charge_volume)
        daily_df = daily_df.join(daily_positive_net_output)
        daily_df = daily_df[:-4]
        daily_df = daily_df.dropna()

        daily_df["actual_losses"] = daily_df["battery_charge"]/daily_df["battery_discharge"]

        #Find modeled charging and discharging
        positive_modeled_net_output = df.loc[df["modeled_bess_output"]>0]
        daily_positive_modeled_net_output = positive_modeled_net_output.groupby('date')[['modeled_bess_output']].sum()
        negative_modeled_net_output = df.loc[df["modeled_bess_output"]<0]
        daily_negative_modeled_net_output = negative_modeled_net_output.groupby('date')[['modeled_bess_output']].sum()
        daily_negative_losses = daily_negative_modeled_net_output/daily_positive_modeled_net_output
        daily_negative_losses.columns = ["daily_modeled_losses"]
        daily_df = daily_df.join(daily_negative_losses)
        daily_df["month"] = pd.to_datetime(daily_df.index).month

        df_monthly = df.groupby("month")[["battery_capacity"]].mean().reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add box plots for each group

        for month in daily_df["month"].unique():
            fig.add_trace(go.Box(y=daily_df.loc[(daily_df["month"] == month)]["battery_discharge"],name=f'{month}', marker_color="blue", showlegend=False))
            fig.add_trace(go.Box(y=daily_df.loc[(daily_df["month"] == month)]["battery_net_output"], name=f'{month}',marker_color="red", showlegend=False))

        fig.add_trace(go.Scatter(x=df_monthly["month"], y=df_monthly["battery_capacity"], mode='lines+markers', name='battery_capacity',marker_color="black"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="blue"), name="battery_discharge"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="red"), name="battery_net_output"))


        # Update layout
        fig.update_layout(
            title='Compare actual daily cycling vs installed capacity, losses',
            xaxis_title='Month',
            yaxis_title='MW',
            xaxis=dict(type='category')  # Ensure the x-axis is treated as categorical
        )
        # Show the plot
        fig.show()

        ############### Create figure for losses ###############

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for month in daily_df["month"].unique():
            fig.add_trace(go.Box(y=daily_df.loc[(daily_df["month"] == month)]["actual_losses"],name=f'{month}', marker_color="blue", showlegend=False))
            fig.add_trace(go.Box(y=daily_df.loc[(daily_df["month"] == month)]["daily_modeled_losses"], name=f'{month}', marker_color="red",showlegend=False))
        fig.update_layout(
            title='Compare monthly losses',
            xaxis_title='Month',
            yaxis_title='%',
            xaxis=dict(type='category')  # Ensure the x-axis is treated as categorical
        )

        fig.show()

        ############### Create figure comparing actual vs daily cyling for only peak ranges ###############

        #Find actual battery charging and discharging during modeled peaks
        discharge_peaks = df.loc[df["modeled_bess_output"]>0]
        daily_discharge_volume = discharge_peaks.groupby('date')[['battery_gen']].sum()
        daily_discharge_volume.columns = ["battery_discharge_peaks"]

        charge_peaks = df.loc[df["modeled_bess_output"]<0]
        daily_charge_volume = charge_peaks.groupby('date')[['battery_gen']].sum()
        daily_charge_volume.columns = ["battery_charge_peaks"]

        # Find modeled battery charging and discharging during modeled peaks
        discharge_peaks = df.loc[df["modeled_bess_output"] > 0]
        modeled_daily_discharge_volume = discharge_peaks.groupby('date')[['modeled_bess_output']].sum()
        modeled_daily_discharge_volume.columns = ["modeled_battery_discharge_peaks"]

        charge_peaks = df.loc[df["modeled_bess_output"] < 0]
        modeled_daily_charge_volume = charge_peaks.groupby('date')[['modeled_bess_output']].sum()
        modeled_daily_charge_volume.columns = ["modeled_battery_charge_peaks"]

        sums = self.input_data.groupby(self.input_data.date)[["battery_charge", "battery_gen","battery_net_output"]].sum()
        averages = self.input_data.groupby(self.input_data.date)[["battery_gwh", "arb_bess_gwh"]].mean()
        daily_df = averages.join(sums)
        daily_df = daily_df.join(daily_discharge_volume)
        daily_df = daily_df.join(daily_charge_volume)
        daily_df = daily_df.join(modeled_daily_discharge_volume)
        daily_df = daily_df.join(modeled_daily_charge_volume)
        daily_df["month"] = pd.to_datetime(daily_df.index).month

        daily_df = daily_df[:-2]
        daily_df = daily_df.dropna()

        df1_name = "battery_discharge_peaks"
        df1 = daily_df[["month",df1_name]]
        df1.rename(columns={df1_name:"value"},inplace=True)
        df1["group"] = df1_name

        df2_name = "modeled_battery_discharge_peaks"
        df2 = daily_df[["month",df2_name]]
        df2.rename(columns={df2_name:"value"},inplace=True)
        df2["group"] = df2_name

        df_combined = pd.concat([df1, df2])
        df_combined["value"] = np.round(df_combined["value"])

        df_monthly = daily_df.groupby("month")[["arb_bess_gwh","battery_gwh"]].mean().reset_index()

        fig = go.Figure()
        # Add box plots for each group
        colors = ["blue", "red"]
        groups = 0
        months = 0
        for month in df_combined["month"].unique():
            groups = 0
            for group in df_combined["group"].unique():
                color = colors[groups]
                groups = groups + 1
                fig.add_trace(
                    go.Box(
                        y=df_combined.loc[(df_combined["month"] == month) & (df_combined["group"] == group)]["value"],
                        name=f'{month}', marker_color=color, showlegend=False))

            months = months+1


        fig.add_trace(go.Scatter(x=df_monthly["month"], y=df_monthly["arb_bess_gwh"], mode='lines+markers', name='arb_bess_gwh',marker_color="black"))
        fig.add_trace(go.Scatter(x=df_monthly["month"], y=df_monthly["battery_gwh"], mode='lines+markers', name='total_battery_gwh',marker_color="purple"))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=colors[0]), name=df1_name))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=colors[1]), name=df2_name))

        # Update layout
        fig.update_layout(
            title='Compare modeled and actual 2024 daily discharge sum for modeled peaks',
            xaxis_title='Month',
            yaxis_title='MW',
            xaxis=dict(type='category')  # Ensure the x-axis is treated as categorical
        )
        # Show the plot
        fig.show()
        print("test")

net_battery_output = battery_model().net_battery_output


########################## PLOT RESULTS ##########################################
plot_results = False

if plot_results:

    net_battery_output["model_state_of_charge"] = net_battery_output["batteries"].cumsum()*-1
    net_battery_output["model_daily_state_of_charge"] = net_battery_output.groupby(net_battery_output.index.date)['batteries'].cumsum()*-1
    #net_battery_output["residuals"] = net_battery_output["batteries"] - net_battery_output["actual_battery_gen"]




