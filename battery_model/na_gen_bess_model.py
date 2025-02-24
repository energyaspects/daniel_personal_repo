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
import re

import warnings
warnings.filterwarnings('ignore')

logger = Logger("ScedScroller").logger  # Creates a logger
shooju_job = sj.register_and_check_job(job_name = f"battery model {datetime.today().strftime('%Y-%m-%d-%H')}")

class battery_model:

    def __init__(self):
        self.ISO = "ERCOT"
        self.sd = datetime(2025, 1, 1)
        self.ed = datetime(2028, 12, 31)
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
        #THERE ARE THREE DIFFERENT SERIES TO USE SEE OPTIONS BELOW

        #OPTION 1: PREFERRED - Use na_gen capacity_adj SIDs
        #sid=teams\power\models\na_gen\CAISO\capacity\fuel\bess
        #battery_capacity_query = r'=({{sid=teams\power\models\na_gen\ERCOT\capacity_adj\batteries}}/1000)'  # call this thermal call in the future
        #battery_capacity = sj.sj.get_df(query=net_load_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1, fields="series_id")

        #OPTION 2: Pull battery capacity data from hardcoded xlsx sheet that has commercial vs synchronized split
        battery_capacity = pd.read_excel("2025-01-16 EA ERCOT battery COD forecast.xlsx",header=1)
        battery_capacity["battery_capacity_cod"] = battery_capacity["battery_capacity_cod"] / 1000

        #OPTION 3: Pull battery capacity data from EIA sids
        #battery_capacity_query = r"=F.multi_sum(r'sid:(tests\asset\timeseries\power_plant) country_iso=US powerplant_specifics_obj.eia_iso=ERCO asset_subtype_name=Batteries table_level=asset_detail timeseries_type=nameplate_capacity timeseries_status=confirmed NOT asset_detail_status=(cancelled,postponed,planned)',[])/1000"
        #battery_capacity_data = sj.sj.get_df( query=battery_capacity_query, series_axis="columns", df=date_start, dt=date_finish, max_points=-1, fields="series_id")

        return battery_capacity

    def get_battery_duration_sid_data(self):

        duration_sid = fr'sid=users\daniel.pyrek\power\models\na_gen\{self.ISO}\capacity_adj\battery_duration'
        battery_duration = sj.sj.get_df(query=duration_sid, series_axis="columns", df=self.sd, dt=self.ed,max_points=-1, fields="series_id")
        battery_duration.columns = ["duration"]
        battery_duration['year'] =battery_duration.index.year
        battery_duration['month'] = battery_duration.index.month
        return battery_duration

    def get_battery_ancillary_sid_data(self):

        battery_as_vol_sid = fr'sid=users\daniel.pyrek\power\models\na_gen\{self.ISO}\capacity_adj\battery_ancillary_capacity'
        battery_as_vol = sj.sj.get_df(query=battery_as_vol_sid, series_axis="columns", df=self.sd, dt=self.ed,max_points=-1, fields="series_id")
        battery_as_vol.columns = ["avg_bess_as_vol"]
        battery_as_vol["avg_bess_as_vol"] = battery_as_vol["avg_bess_as_vol"]/1000
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
        #THIS IS NOT NECESSARY IN THE MODEL BUT USED FOR RESIDUALS, TESTING ETC.

        #Define the region
        if self.ISO == "ERCOT":
             battery_gen_query = r'=({{sid=teams\power\ercot\gen_mix\WSL}} + {{sid=teams\power\ercot\gen_mix\other}})@A:h@localize:America/Chicago'

        if self.ISO == "CAISO":
            battery_gen_query = r'={{sid=teams\power\caiso\gen_mix_new\batteries}}/1000@A:h@localize:America/Los_Angeles'

        #Pull the data
        battery_gen_data = sj.sj.get_df(query=battery_gen_query, series_axis="columns", df=self.sd, dt=self.ed, max_points=-1,fields="series_id")
        battery_gen_data.columns = ["actual_battery_gen"]
        battery_gen_data.index = battery_gen_data.index.tz_localize(None)
        battery_gen_data.reset_index(inplace=True)

        return battery_gen_data

    #Helper functions that convert ordinal times to datetime and vice versa
    def DatetimeToOrdinal(self,dt):
        # Converts datetime to ordinal time format
        ordinal = dt.toordinal()
        # Calculate the start of the day for the given datetime
        start_of_day = datetime.combine(dt.date(), datetime.min.time())
        # Calculate the number of seconds in the day dynamically
        seconds_in_day = (start_of_day + timedelta(days=1) - start_of_day).total_seconds()
        # Calculate the fraction of the day
        fraction = (dt - start_of_day).total_seconds() / seconds_in_day
        return ordinal + fraction

    def OrdinalToDatetime(self,ordinal):
        #Converts ordinal time format to datetime
        plaindate = date.fromordinal(int(ordinal))
        date_time = datetime.combine(plaindate, datetime.min.time())
        return date_time + timedelta(days=ordinal - int(ordinal))

    def ensure_no_negatives(self, df):
        #Takes a dataframe with one column and ensures no values are negative
        min_value = df.iloc[:, 0].min()

        if min_value < 0:
            df.iloc[:, 0] += abs(min_value)

        return df

    #Function that runs the extractors and merges them
    def Extract_Merge(self):
        #Call all functions to pull data
        thermal_call_data = self.get_thermal_call_sid_data()
        logger.info(f'Pulled {len(thermal_call_data)} rows of thermal call data')

        battery_capacity_data = self.get_battery_capacity_sid_data()
        logger.info(f'Pulled {len(battery_capacity_data)} rows of battery capacity data')

        battery_duration_data = self.get_battery_duration_sid_data()
        logger.info(f'Pulled {len(battery_duration_data)} rows of battery duration data')

        battery_ancillary_data = self.get_battery_ancillary_sid_data()
        logger.info(f'Pulled {len(battery_ancillary_data)} rows of battery ancillary data')

        actual_battery_output_data = self.get_actual_battery_output_sid_data()
        logger.info(f'Pulled {len(actual_battery_output_data)} rows of actual battery output data')

        #Merge everything together
        df = thermal_call_data.merge(battery_capacity_data)
        df = df.merge(battery_duration_data)
        df = df.merge(battery_ancillary_data)
        #df = df.merge(actual_battery_output_data)
        logger.info(f'Merged all input data')

        #Calculate additional columns
        df["battery_gwh"] = df['duration'] * df['battery_capacity_cod'] #calculate raw battery gwh
        df["arb_bess_capacity"] = df["battery_capacity_cod"] - df["avg_bess_as_vol"] #calculate battery capacity able to arb
        df["arb_bess_gwh"] = df["battery_gwh"] - df["avg_bess_as_vol"] * df["duration"] #calculate battery arb gwh
        #df["total_battery_cf"] = df["actual_battery_gen"] / df["battery_capacity_cod"] #calculate total battery cf
        #df["arb_battery_cf"] = df["actual_battery_gen"] / df["arb_bess_capacity"] #calculate arb battery cf
        df['thermal_call_ramp'] = df['thermal_call'].diff() #calculate ramping - not needed but could be used in future

        #Clean up data and save df to self
        df.set_index("index",inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]  # remove duplicates as daylight change with extra hour breaks code
        self.input_data = df
        logger.info(f'Completed data ingestion')

    #Components of the battery model and the runner
    def find_peaks(self, thermal_call_data):
        #takes datetime index and the associated data as series and calculates the peak data
        x = thermal_call_data.index
        y = thermal_call_data.iloc[:,0]

        #map datetimes to ordinal number
        datetime_map = np.array([self.DatetimeToOrdinal(dt) for dt in x])

        ############################# WE WILL LIKELY TUNE THIS P #############################
        # Find peaks and calculate prominences
        peaks, peak_info = find_peaks(y, width=1, prominence=1.5)
        #peaks, peak_info = find_peaks(y,width=1,prominence=1,distance=100)
        #peaks, peak_info = find_peaks(y, width=3)
        prominences = peak_info['prominences']
        contour_heights = y[peaks] - prominences

        #calculate width information at varying heights of the prominences
        peaks_data = pd.DataFrame()
        rel_heights = np.arange(0, 1.0, self.width_slices)

        ############################# WE COULD AVOID ALL rel_heights especially for the smaller areas #############################
        for rel_height in rel_heights:
            discharge_widths_raw = peak_widths(y, peaks, rel_height=rel_height)
            discharge_widths_raw = pd.DataFrame(discharge_widths_raw).transpose()
            discharge_widths_raw.columns = ["widths","width_heights","left_ips","right_ips"]
            discharge_widths = discharge_widths_raw.copy()
            discharge_widths["left_ips"] = discharge_widths["left_ips"]#np.floor(discharge_widths["left_ips"])
            discharge_widths["right_ips"] = discharge_widths["right_ips"] #np.ceil(discharge_widths["right_ips"])
            discharge_intercepts = discharge_widths[['left_ips','right_ips']]
            discharge_intercepts["left_ips"] = discharge_intercepts["left_ips"]/24/self.resolution_interval + datetime_map[0]
            discharge_intercepts["right_ips"] = discharge_intercepts["right_ips"]/24/self.resolution_interval + datetime_map[0]
            discharge_intercepts["left_ips"] = np.array([self.OrdinalToDatetime(dt) for dt in discharge_intercepts["left_ips"]])
            discharge_intercepts["right_ips"] = np.array([self.OrdinalToDatetime(dt) for dt in discharge_intercepts["right_ips"]])
            discharge_intercepts["left_ips"] = discharge_intercepts["left_ips"].dt.floor(self.resolution)
            discharge_intercepts["right_ips"] = discharge_intercepts["right_ips"].dt.ceil(self.resolution)
            discharge_intercepts["prominence"] = prominences
            discharge_intercepts["height"] = prominences*rel_height
            discharge_intercepts["width"] = discharge_widths_raw["widths"]/self.resolution_interval
            discharge_intercepts["width_height"] = discharge_widths_raw["width_heights"]
            discharge_intercepts["peaks"] = peaks / 24 / self.resolution_interval + datetime_map[0]
            discharge_intercepts["peaks"] = np.array([self.OrdinalToDatetime(dt) for dt in discharge_intercepts["peaks"]])
            discharge_intercepts["peaks"]  = discharge_intercepts["peaks"].dt.round(self.resolution)
            discharge_intercepts["rel_height"] = rel_height
            discharge_intercepts["contour_heights"] = contour_heights
            #width_data.append(discharge_intercepts)
            peaks_data = pd.concat([peaks_data,discharge_intercepts])

        peaks_data.set_index("peaks",inplace=True)

        ###### MAKE A PLOT TO CHECK THE PEAKS ######

        display_data = pd.DataFrame(y, x)
        display_data.index = pd.to_datetime(display_data.index)
        #display_data.reset_index(inplace=True)
        peak_dates = peaks / 24 / self.resolution_interval + datetime_map[0]
        peak_dates = np.array([self.OrdinalToDatetime(dt) for dt in peak_dates])
        peak_dates = pd.Series(peak_dates).dt.round(self.resolution)

        if self.plot_charts:
            #PLOT PEAKS
            # Specify the date range window that defaults in the display
            start_date = '2024-08-16'
            end_date = '2024-08-24'

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add the main line plot
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Signal', line=dict(color='rgba(0,71,171,1)')),
                          secondary_y=False)

            # Add vertical lines for peak prominences
            for peak, contour_height in zip(peak_dates, contour_heights):
                fig.add_trace(go.Scatter(
                    x=[peak, peak],
                    y=[contour_height, display_data.loc[peak][0]],
                    mode='lines',
                    line=dict(color='rgba(0,71,171,1)', dash='dot'),
                    showlegend=False),
                    secondary_y=False)

            # Update layout
            fig.update_layout(title='Signal with Peaks and Prominences',
                              xaxis_title='X',
                              yaxis_title='Amplitude',
                              yaxis2_title='Battery Charge Factor',
                              xaxis=dict(
                                  range=[start_date, end_date],  # Set the initial date range
                                  type='date'  # Ensure the x-axis is treated as a date axis
                              ),
                              )

            # Show the plot
            fig.show()

        return peaks_data

    def integrate_peaks(self,peaks_data, thermal_call_data, power_requirements, duration_requirements):

        #Clean and merge peaks data and power requirements
        peaks_data["date"] = peaks_data.index.date
        peaks_data.reset_index(inplace=True)
        power_requirements.reset_index(inplace=True)
        peaks_data = peaks_data.merge(power_requirements, on="date")

        #Filter out scenarios where the model would dispatch the battery beyond its power limit
        peaks_data = peaks_data.loc[peaks_data['height'] <= peaks_data['arb_bess_capacity']]

        peaks_data.set_index("peaks", inplace=True)
        #prepare data and loop integrating the areas
        peak_dates = pd.Series(peaks_data.index.unique())

        peak_data_with_area = pd.DataFrame()


        #loop through each peak
        for peak in peak_dates:
            #Grab the scenarios for each peak
            peak_areas = []
            peak_data = peaks_data.loc[peak]
            # Pull duration requirement
            peak_date = peak_data['date'][0]
            daily_duration_requirement = duration_requirements.loc[peak_date][0]
            energy_limit_flag = True
            #peak_data = peak_data.reset_index(drop=True)

            #loop through each relative height
            for rel_height_row in peak_data["rel_height"]:
                if energy_limit_flag: #If the code has not yet exceeded the total energy limit

                    ## I THINK THE NEXT TWO LINES ARE SLOWING DOWN THE CODE ##
                    #Find the start and end timestamp of the peak
                    left_ips = peak_data.loc[peak_data['rel_height'] == rel_height_row, 'left_ips'].iloc[0]
                    right_ips = peak_data.loc[peak_data['rel_height'] == rel_height_row, 'right_ips'].iloc[0]
                    #left_ips = pd.Timestamp(peak_data.loc[peak_data['rel_height'] == rel_height_row, 'left_ips'].values[0])
                    #right_ips = pd.Timestamp(peak_data.loc[peak_data['rel_height'] == rel_height_row, 'right_ips'].values[0])

                    #Pull the thermal call values associated with that range
                    data = thermal_call_data.iloc[:,0].loc[left_ips:right_ips]

                    #Take the integral I used scipy but np.trapz might be more efficient (although less accurate according to the web)
                    #area_under_curve = np.trapz(data.values) / 12
                    area_under_curve = integrate.simpson(data.values) / self.resolution_interval

                    #Calculate the width
                    width = (right_ips - left_ips).total_seconds() / 60 / 60

                    # Get area difference height
                    height_area_to_subtract = peak_data.loc[peak_data['rel_height'] == rel_height_row, 'width_height'].iloc[0]

                    #If negative???? do we need this
                    #if height_area_to_subtract < 0:
                    #    height_area_to_subtract = height_area_to_subtract * -1

                    #Calc the area under the peak to subtract
                    lower_area = width * height_area_to_subtract

                    #Calculate the peak and add to list
                    peak_area = area_under_curve - lower_area

                    if peak_area >= daily_duration_requirement:
                        energy_limit_flag = False

                else:
                    peak_area = np.NAN

                peak_areas.append(peak_area)

            # peak_width_data["peak_areas"] = peak_areas
            peak_data.loc[:, "peak_areas"] = peak_areas
            peak_data_with_area = pd.concat([peak_data_with_area, peak_data])

        peak_data_with_area = peak_data_with_area.loc[peak_data_with_area["peak_areas"] >= 0]  #get rid of negative noise and NAs
        return peak_data_with_area

    def select_peaks(self,peak_data_with_area, thermal_call_data, duration_requirements):

        def create_sum_matrix(arrays):
            if len(arrays) == 0:
                return np.array([])  # Return an empty array if the list is empty
            elif len(arrays) == 1:
                return arrays[0]  # Return the array itself if there's only one array
            elif len(arrays) == 2:
                arr1, arr2 = arrays
                return arr1[:, None] + arr2  # Use broadcasting to create a matrix of sums

        def create_divide_matrix(arrays):
            if len(arrays) == 0:
                return np.array([])  # Return an empty array if the list is empty
            elif len(arrays) == 1:
                return arrays[0]  # Return the array itself if there's only one array
            elif len(arrays) == 2:
                arr1, arr2 = arrays
                return arr1[:, None] / arr2  # Use broadcasting to create a matrix of sums

        #For each day in the data set, pick up to 2 peaks that fit all the requirements
        dates = peak_data_with_area["date"].unique()
        battery_output = pd.DataFrame()
        for date in dates:
            # Narrow down data to day
            daily_peaks = peak_data_with_area.loc[peak_data_with_area['date'] == date]

            # Find unique peaks in day
            unique_peak_times = daily_peaks.index.unique()

            # Filter to top two peaks by prominence, that way we exclude any noise from extra peaks, works with only one/zero peaks too
            peak_prominences = daily_peaks.groupby(daily_peaks.index)['prominence'].max()
            peak_prominences = peak_prominences.sort_values(ascending=False)
            top_two_prominence_peaks = peak_prominences[:2]

            # Pull duration requirement
            daily_duration = duration_requirements.loc[date,"daily_duration_requirement"]

            # Create matrix of all combinations of peak areas
            unique_peak_volumes = []

            for peak in top_two_prominence_peaks.index:
                unique_peak_volume = np.array(daily_peaks.loc[daily_peaks.index == peak]["peak_areas"])
                unique_peak_volumes.append(unique_peak_volume)

            #Get rid of negative artifacts might be a rounding issue
            sum_matrix = create_sum_matrix(unique_peak_volumes)

            # IF THERE ARE TWO PEAKS CONTINUE
            if len(top_two_prominence_peaks) == 2:

                #Select value that preserves prominence ratio the best - should change in the future so that it doesnt pick a small value just because it preserves ratio

                #Step 1: Create a matrix of ratios and find the prominence ratio
                divide_matrix = create_divide_matrix(unique_peak_volumes)
                prominence_ratio =  top_two_prominence_peaks[0]/top_two_prominence_peaks[1]

                # Step 2: Create a mask to filter divide_matrix where sum matrix exceeds the daily duration
                mask = sum_matrix < daily_duration
                filtered_sum_matrix = np.where(mask, sum_matrix, np.nan)
                filtered_divide_matrix = np.where(mask, divide_matrix, np.nan)

                # Step 3: Find the cell in the filtered_divide_matrix that is closest to the prominence_ratio
                # IF THERE ARE ANY PEAKS THAT MEET CRITERIA CONTINUE
                if np.any(~np.isnan(filtered_divide_matrix)):

                    # METHOD 1 strictly picks the highest value
                    #Calculate the value difference between the prominence ratio and divide matrix
                    closest_index = np.nanargmin(np.abs(filtered_divide_matrix - prominence_ratio))
                    # Convert the flat index to a 2D index
                    closest_position = np.unravel_index(closest_index, filtered_divide_matrix.shape)
                    # Get the value from matrix2 that is closest to prominence_ratio
                    closest_value = filtered_divide_matrix[closest_position]
                    # Step 4: Retrieve the values from arr1 and arr2 using the closest_position
                    value_from_arr1 = unique_peak_volumes[0][closest_position[0]]
                    value_from_arr2 = unique_peak_volumes[1][closest_position[1]]
                    peak_1_info = daily_peaks.loc[daily_peaks['peak_areas'] == value_from_arr1]
                    peak_2_info = daily_peaks.loc[daily_peaks['peak_areas'] == value_from_arr2]


                    """
                    #METHOD 2 balances ratio and size by taking top 3 closest matches and taking highest
                    n=3
                    combined_diff = np.abs(filtered_divide_matrix - prominence_ratio)
                    flat_combined_diff = combined_diff.flatten()
                    valid_indices = np.where(~np.isnan(flat_combined_diff))[0]
                    sorted_indices = valid_indices[np.argsort(flat_combined_diff[valid_indices])[:n]]
                    closest_positions = [np.unravel_index(index, filtered_divide_matrix.shape) for index in sorted_indices]
                    values = [filtered_sum_matrix[pos] for pos in closest_positions]
                    largest_value_index = np.argmax(values)
                    closest_position = closest_positions[largest_value_index]
                    value_from_arr1 = unique_peak_volumes[0][closest_position[0]]
                    value_from_arr2 = unique_peak_volumes[1][closest_position[1]]
                    peak_1_info = daily_peaks.loc[daily_peaks['peak_areas'] == value_from_arr1]
                    peak_2_info = daily_peaks.loc[daily_peaks['peak_areas'] == value_from_arr2]
                    """
                    peak_1_height = peak_1_info["width_height"][0]
                    peak_2_height = peak_2_info["width_height"][0]

                    peak_1_bess = thermal_call_data.iloc[:,0].loc[
                                  peak_1_info["left_ips"][0]:peak_1_info["right_ips"][0]]  # get thermal call shape for range
                    peak_2_bess = thermal_call_data.iloc[:,0].loc[
                                  peak_2_info["left_ips"][0]:peak_2_info["right_ips"][0]]  # get thermal call shape for range

                    peak_1_bess = peak_1_bess - peak_1_height  # adjust for height
                    peak_2_bess = peak_2_bess - peak_2_height  # adjust for height

                    battery_output = pd.concat([battery_output, peak_1_bess])
                    battery_output = pd.concat([battery_output, peak_2_bess])

            # IF THERE IS A SINGLE PEAK CONTINUE
            if len(top_two_prominence_peaks) == 1:
                filtered_array = sum_matrix[sum_matrix < daily_duration]
                # IF THERE ARE ANY PEAKS THAT MEET CRITERIA CONTINUE
                if np.any(~np.isnan(filtered_array)):
                    largest_value = np.max(filtered_array)
                    peak_1_info = daily_peaks.loc[daily_peaks['peak_areas'] == largest_value]
                    peak_1_bess = thermal_call_data.iloc[:,0].loc[
                                  peak_1_info["left_ips"][0]:peak_1_info["right_ips"][0]]  # get net load shape for range
                    peak_1_bess = peak_1_bess - peak_1_info["width_height"][0]  # adjust for height
                    battery_output = pd.concat([battery_output, peak_1_bess])

        ###### Clean up data and return ######
        battery_output.columns = ["battery_output"]

        #Fix a bit of artifacting at the edge points might want to see what is causing those in the future
        battery_output.loc[battery_output["battery_output"] < 0] = 0

        #Average to hourly and drop any NA values
        hourly_battery_output = battery_output.resample('H')['battery_output'].mean()
        hourly_battery_output = pd.DataFrame(hourly_battery_output).dropna()

        #Get original index, average it hourly
        full_index = thermal_call_data.resample('h').mean().index
        full_index = pd.DataFrame(index=full_index)

        #Merge battery data on and replace NAs with zeros
        clean_hourly_battery_output = full_index.join(hourly_battery_output)
        clean_hourly_battery_output = clean_hourly_battery_output.fillna(0)

        return clean_hourly_battery_output

    def runner(self):

        # Pull all input data
        self.Extract_Merge()

        # upscale to X minute so that we get better results from scipy peak widths fuction
        upscaled_data = self.input_data.resample(self.resolution).interpolate(method='linear')

        #Prepare the charging and discharging signals, ensuring no values are negative
        upscaled_data["inversed_thermal_call"] = upscaled_data["thermal_call"] * -1
        discharging_thermal_call_signal = self.ensure_no_negatives(upscaled_data[["thermal_call"]])
        charging_thermal_call_signal = self.ensure_no_negatives(upscaled_data[["inversed_thermal_call"]])

        ############# CALCULATE DISCHARGING DATA #############

        # Find peaks and return information about them
        peak_data = self.find_peaks(discharging_thermal_call_signal)
        logger.info(f'Found discharging peak possibilities')

        # Prepare max power requirements
        power_requirements = self.input_data.groupby("date")[["arb_bess_capacity"]].mean()

        # Prepare duration requirements
        discharge_duration_requirements = self.input_data.groupby("date")[["arb_bess_gwh"]].mean()
        discharge_duration_requirements.columns = ["daily_duration_requirement"]

        # Find area for each peak possibility
        peak_integrals = self.integrate_peaks(peak_data, discharging_thermal_call_signal, power_requirements, discharge_duration_requirements)
        logger.info(f'Found discharging peak possibility areas')

        # Select areas that fit requirements
        battery_discharge = self.select_peaks(peak_integrals, discharging_thermal_call_signal,
                                              discharge_duration_requirements)
        logger.info(f'Finished modeling BESS discharging for {self.ISO}')

        ############# CALCULATE CHARGING DATA #############

        # Find peaks and return information about them
        peak_data = self.find_peaks(charging_thermal_call_signal)
        logger.info(f'Found charging peak possibilities')

        # Prepare duration requirements for charging to match discharging + X round trip efficiency
        charge_duration_requirements = battery_discharge.copy()
        charge_duration_requirements["date"] = charge_duration_requirements.index.date
        charge_duration_requirements = charge_duration_requirements.groupby("date").sum()
        charge_duration_requirements.columns = ["daily_duration_requirement"]
        charge_duration_requirements["daily_duration_requirement"] = charge_duration_requirements[
                                                                         "daily_duration_requirement"] * (
                                                                                 1 + self.round_trip_losses_percentage)

        # Find area for each peak possibility, power requirements are the same
        peak_integrals = self.integrate_peaks(peak_data, charging_thermal_call_signal, power_requirements,charge_duration_requirements)
        logger.info(f'Found charging peak possibility areas')



        # Select areas that fit requirements, multiply by negative 1 so that it is the right sign
        battery_charge = -self.select_peaks(peak_integrals, charging_thermal_call_signal, charge_duration_requirements)
        logger.info(f'Finished modeling BESS charging for {self.ISO}')
        ############# CLEAN DATA AND RETURN #############

        # Net charging and discharging to one net dataseries
        net_battery_output = battery_charge + battery_discharge
        net_battery_output.columns = ["batteries"]

        # Add actual battery gen for comparison, can drop in the PROD version
        #net_battery_output = net_battery_output.join(self.input_data["actual_battery_gen"])

        #Save to self and load
        self.net_battery_output = net_battery_output[["batteries"]]
        self.load_results()

        logger.info(f'Finished modeling BESS for {self.ISO}')

        self.net_battery_output = net_battery_output

    #Write results to SIDs
    def load_results(self):
        data = self.net_battery_output
        for column in data:
            sid = rf"sid=users\{sj.shooju_api_user_name}\power\models\na_gen\{self.ISO}\generation_forecasts\{column}"
            static_fields = {}
            dynamic_fields = {
                "description": f" Hourly EA net battery output for {self.ISO} in GW"
            }
            fields = {**static_fields, **dynamic_fields}
            job_id = sj.shooju_write(sid=sid, series=data[column], metadata=fields, job=shooju_job,
                                     remove_others=None)
        logger.info(fr'just wrote with this job: {os.environ["SHOOJU_SERVER"]}#jobs/{job_id}')
        logger.info(fr'these sids: {os.environ["SHOOJU_SERVER"]}#explorer?query=meta.job%3D{job_id}')
        ############# PREPARE DATA #############

net_battery_output = battery_model().net_battery_output


########################## PLOT RESULTS ##########################################
plot_results = False

if plot_results:

    net_battery_output["model_state_of_charge"] = net_battery_output["batteries"].cumsum()*-1
    net_battery_output["model_daily_state_of_charge"] = net_battery_output.groupby(net_battery_output.index.date)['batteries'].cumsum()*-1
    #net_battery_output["residuals"] = net_battery_output["batteries"] - net_battery_output["actual_battery_gen"]

    # Specify the date range you want to display
    start_date = '2024-08-16'
    end_date = '2024-08-24'

    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the main line plot
    fig.add_trace(go.Scatter(x=net_battery_output.index, y=net_battery_output["batteries"], mode='lines', name='modeled_net_bess_output',line=dict(color='rgba(0,71,171,1)')), secondary_y=False)

    # Add the battery_cf line plot on the secondary y-axis
    #fig.add_trace(go.Scatter(x=net_battery_output.index, y=net_battery_output["actual_battery_gen"], mode='lines', name='actual_battery_gen',line=dict(color='rgba(5,150,5,1)')), secondary_y=False)

    # Add the SOC plot
    fig.add_trace(go.Scatter(x=net_battery_output.index, y=net_battery_output["model_state_of_charge"], mode='lines', name='model_state_of_charge',line=dict(color='rgba(150,5,5,1)')), secondary_y=False)

    # Add the SOC plot
    fig.add_trace(go.Scatter(x=net_battery_output.index, y=net_battery_output["model_daily_state_of_charge"], mode='lines', name='model_daily_state_of_charge',line=dict(color='rgba(150,5,5,1)')), secondary_y=False)


    # Add the battery_cf line plot on the secondary y-axis
    fig.add_trace(go.Scatter(x=net_battery_output.index, y=net_battery_output["residuals"], mode='lines', name='residuals',line=dict(color='rgba(0,0,0,1)')), secondary_y=False)

    # Update layout
    fig.update_layout(title='Battery model vs actuals',
                      xaxis_title='Datetime',
                      yaxis_title='GW',
                      #yaxis2_title='Battery Charge Factor',
                      xaxis=dict(
                            range=[start_date, end_date],  # Set the initial date range
                            type='date'  # Ensure the x-axis is treated as a date axis
        ),
                      )

    # Show the plot
    fig.show()


