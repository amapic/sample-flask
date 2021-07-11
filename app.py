from os import path
import os.path
from datetime import datetime as dt
import datetime
# import plotly.express as px
# from dash.dependencies import Input, Output, State
# import dash_html_components as html
# import dash_core_components as dcc
import json
import pandas as pd
import numpy as np
# from jupyter_dash import JupyterDash
# import plotly.graph_objs as go
# import dash
# import traceback
import sys
import os
import copy
import os
import glob
# import grasia_dash_components as gdc
import json
# import jsonify
# import ciso8601
old_env = os.environ['PATH']
df_return = []

# couleur(5,98,138)
# 72,145,118 #489176
# # 206,136,87 #ce8857
# 154,80,82 #9a5052
# 160,175,82 #a0af52
# 88,158,157 #589e9d
# 103,120,132 #677884
# 206,182,75 #ceb64b
# 40,72,101 #284865
# 166,135,103 #a68767


from flask import Flask, jsonify,render_template
# from flask_cors import CORS
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# CORS(app, support_credentials=True)
today = dt.today().strftime("%m-%d-%Y")
# csv_today_path = 'C:/Users/Utilisateur/PycharmProjects/montee_en_competence/csv_vaccin/' + today + ".csv"
csv_today_path = today + ".csv"
if path.exists(csv_today_path):
	df_vaccin_quotidien = pd.read_csv(csv_today_path)


@app.route('/somme/<reg>/<vaccin>')
def filter_data_somme_quotidien(reg,vaccin):
		# print(type(reg))
		reg=np.int64(reg)
		vaccin=np.int64(vaccin)
		return df_vaccin_quotidien.query('reg==@reg & vaccin==@vaccin').to_json()
		
@app.route('/detail/<reg>/<vaccin>')
def filter_data_detail(reg,vaccin):
		# print(type(reg))
		reg=np.int64(reg)
		vaccin=np.int64(vaccin)
		# response=df_vaccin_detail.query('reg==@reg & vaccin==@vaccin').to_json()
		# response.headers.add("Access-Control-Allow-Origin", "*")
		return df_vaccin_detail.query('reg==@reg & vaccin==@vaccin').reset_index().to_json()
		
@app.route("/")
def helloWorld():
  return "Hello, cross-origin-world!"
	
@app.route('/color')
def choose_color(i):
	color_list = ["#489176", "#ce8857", "#9a5052", "#a0af52", " #589e9d", "#677884", "#ceb64b", "#284865", "#a68767"]
	if i>=len(color_list):
		return color_list[len(color_list)-i]
	else:
		return color_list[i]
# liste_des_vaccins
df_liste_vaccin = ["Tous","COMIRNATY Pfizer/BioNTech", "Moderna", "AstraZeneka"]
print(df_liste_vaccin)
# CHargement liste des régions
src = 'C:/Users/Utilisateur/PycharmProjects/montee_en_competence/stat_pop.csv'

# df_population=pd.read_csv(src,sep=";")///////////////a remettre
# bilan des données actuelle
# Le fichier est mis à jour quotidiennement et il comprend la somme par région et par vaccin. Cependant pour certaines
# régions et vaccins,la valeur peut manquer à un jour donné. On créee donc un fichier à jour cumulant les données
# de la journée selon data-france et celles données par data-fr&ance il y a pluysieurs jours pour les valeurs manquantes

date_min=datetime.datetime.strptime("27/12/2020","%d/%m/%Y").timestamp()
date_max=datetime.datetime.timestamp(datetime.datetime.today()-datetime.timedelta(days=1))
print(date_min)

labels = ["A1", "A2", "A3", "A4", "A5", "B1", "B2"]
parents = ["", "", "", "", "", "", ""]
values = ["11", "12", "13", "14", "15", "20", "30"]

# fig = go.Figure(go.Treemap(
    # labels = labels,
	# values = values,
    # parents = parents,
	# marker_colors = [ choose_color(i) for i in range(len(labels))]
# ))
@app.route('/req/proutos')
def maj_data_complete():
	today = dt.today().strftime("%m-%d-%Y")
	csv_today_path = 'C:/Users/Utilisateur/PycharmProjects/montee_en_competence/csv_vaccin/' + today + ".csv"
	if path.exists(csv_today_path + "r"):
		df_vaccin_quotidien = pd.read_csv(csv_today_path)
	else:
		src = 'https://www.data.gouv.fr/fr/datasets/r/900da9b0-8987-4ba7-b117-7aea0e53f530'
		df_vaccin_quotidien = pd.read_csv(src, sep=";")
		dernier_jour = df_vaccin_quotidien.tail(1)
		dernier_jour = str(dernier_jour['jour'].values[0])

		df_vaccin_quotidien = df_vaccin_quotidien.query('jour==@dernier_jour')
		df_vaccin_quotidien.query('reg!=6', inplace=True)
		df_vaccin_quotidien.query('reg!=8', inplace=True)
		# df_vaccin_quotidien.to_csv('C:/Users/Utilisateur/PycharmProjects/montee_en_competence/csv_vaccin/01-03-2021.csv')
		df_vaccin_quotidien.query('reg!=1', inplace=True)

		reg_sans_data = [
			x for x in pd.unique(
				df_population["id"]) if x not in pd.unique(
				df_vaccin_quotidien["reg"])]

		# recherche de données dans des anciennes stats

		day_diff = 0
		# for reg in reg_sans_data:
		data_full = False
		# boucle sur les archives de données
		while not data_full:
			day_diff += 1
			today = dt.today().strftime("%m-%d-%Y")
			day_delta = dt.today() + datetime.timedelta(days=-day_diff)
			day_delta_str = day_delta.strftime("%m-%d-%Y")
			files = glob.glob(
				"C:/Users/Utilisateur/PycharmProjects/montee_en_competence/csv_vaccin/*-*-*.csv")
			df_reg_manquant = pd.read_csv(files[-1])

			df_reg_manquant = df_reg_manquant[df_reg_manquant.reg.isin(
				reg_sans_data)]
			if pd.unique(df_reg_manquant.reg) == len(reg_sans_data):
				data_full = True

		if not data_full:
			print("data pas complète")
			exit()

		df_reg_manquant.sort_values(['jour'], inplace=True)
		# on reprend la dernière ligne du fichier de données manquante
		dernier_jour = df_reg_manquant.tail(1)
		dernier_jour = str(dernier_jour['jour'].values[0])
		df_reg_manquant.query('jour==@dernier_jour', inplace=True)
		df_vaccin_quotidien = pd.concat([df_vaccin_quotidien, df_reg_manquant])

	# sauvegarde des données
		df_vaccin_quotidien.iloc[:-2, : 7].to_csv(csv_today_path)
		
	return df_vaccin_quotidien.iloc[:-2, : 7].to_json()
	# return "e"
	
# df_vaccin_quotidien=maj_data_complete()

# somme des datas par jour même incomplète pour pouvoir viusaliser les changements depuis lavaeille
#dans df_vaccin_detail on peut avoir des journées manquantes
@app.route('/b')
def make_vaccin_detail():
	today = dt.today().strftime("%m-%d-%Y")
	csv_today_path = 'C:/Users/Utilisateur/PycharmProjects/montee_en_competence/csv_vaccin_detail/' + today + ".csv"
	if path.exists(csv_today_path):
		return pd.read_csv(csv_today_path)
	else:
		src = 'https://www.data.gouv.fr/fr/datasets/r/900da9b0-8987-4ba7-b117-7aea0e53f530'
		df_vaccin_detail = pd.read_csv(src, sep=";")
		df_vaccin_detail['datetime']=pd.to_datetime(df_vaccin_detail['jour'], format='%Y-%m-%d')
		df_vaccin_detail['timestamp']=df_vaccin_detail['datetime'].apply(lambda x: datetime.datetime.timestamp(x))
		# df_vaccin_detail['somme_jour_vaccin'] = df_vaccin_detail.groupby(['jour', 'vaccin'])['n_cum_dose1'].transform(
		# 	sum)
		df_vaccin_detail.to_csv(csv_today_path)
		
		return df_vaccin_detail
	#le seul truc qui nous manque c'est le cumul par jour par vaccin sur toutes les régions

df_vaccin_detail=make_vaccin_detail() 


# month_options = [{'label': month, 'value': i}
#                  for (month,i) in enumerate(pd.unique(df_liste_vaccin))]

#somme_jour_vaccin tient compte du fait qu'il peut y  voir plusieurs ligns pour une meme journée et uneme meme re
#gion et un meme vaccin
# dataframe=pd.DataFrame([]);
# dataframe['somme_jour_vaccin'] = df_vaccin_detail.groupby(['jour', 'vaccin'])[
	# 'n_dose1'].transform(sum)
# dataframe.drop(['reg','n_dose1','n_dose2','n_cum_dose1','n_cum_dose2'],axis=1,inplace=True)
@app.route('/c')
def make_data(df_vaccin_detail=df_vaccin_detail,liste_vaccin=[0,1,2,3],date_range=[]):

	if liste_vaccin is None:
		liste_vaccin = [0, 1, 2, 3]

	dataframe = df_vaccin_detail.copy()
	dataframe.drop_duplicates()
	dataframe = dataframe.sort_values(by=['jour'])

	for vaccin in liste_vaccin:
		dataframe['cumsum_' + str(vaccin)] = dataframe.loc[dataframe.vaccin == vaccin, "n_dose1"].cumsum()

	dataframe.to_clipboard(sep=',', index=False)
	if date_range != []:
		date_min = date_range[0]
		date_max= date_range[1]
		dataframe = dataframe.loc[dataframe['timestamp']>date_min,]
		dataframe = dataframe.loc[dataframe['timestamp']<date_max,]
	dataframe.drop(['timestamp','datetime'],axis=1,inplace=True)
	data = []
	for vaccin in liste_vaccin:
		dataframe2 = dataframe.query('vaccin==@vaccin')

		trace = go.Scatter(x=dataframe2.jour, y=dataframe2['cumsum_' + str(vaccin)], name=df_liste_vaccin[vaccin])
		data.append(trace)
		
	return data
	
# data=make_data(df_vaccin_detail)

# layout
# layout = go.Layout(xaxis={'title': 'Time'},
				   # yaxis={'title': 'Produced Units'},
				   # margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
				   # hovermode='closest')

# figChart = go.Figure(data=data, layout=layout)


gg = 0
df = []

# ratio de pop vacciné
@app.route('/f')
def ratio():
	df_vaccin_quotidien['somme_jour'] = df_vaccin_quotidien.groupby(['reg', 'jour'])[
		'n_dose1'].transform(sum)

	population_totale = 65000000

	pop_totale_vacciné = population_totale / df_vaccin_quotidien["n_dose1"].sum()
	liste_vaccin = pd.unique(df_vaccin_quotidien["vaccin"])

	# df_vaccin_quotidien['reg'] = df_vaccin_quotidien['reg'].apply(lambda x: x.zfill(2))
	# df_vaccin_quotidien['reg']=df_vaccin_quotidien['reg'].astype("string")

	df_tot = df_population.merge(df_vaccin_quotidien, left_on='id', right_on='reg')

	df_tot = df_tot.query('vaccin==0')  # vaccin=0 somme des vaccins

	df_tot["ratio"] = 100 * (df_tot["n_cum_dose1"] / df_tot["pop"])
	df_tot["ratio_dose_2"] = 100 * \
							 (df_tot["n_cum_dose1"] + df_tot["n_cum_dose2"]) / df_tot["pop"]
							 
	return df_tot

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','./assets/yah.css']
# external_scripts = ['https://polyfill.io/v3/polyfill.min.js?features=default',
					# 'france_geojson.js', 'https://cdn.jsdelivr.net/npm/underscore@1.12.0/underscore-min.js',
					# 'load_gmap.js',
					# 'https://maps.googleapis.com/maps/api/js?key=AIzaSyBHNfjux'
					# 'MNcHVdkLgHctexkayh5tAMOWjA&callback=initMap&libraries=&v=weekly']
					

# app = dash.Dash(__name__, external_scripts=external_scripts)
@app.route('/g')
def load_geojson():
	with open('C:/Users/Utilisateur/PycharmProjects/montee_en_competence/france_geojson.json') as f:
		geojson = json.load(f)

	for i, x in enumerate(geojson["features"]):
		df.append([x["properties"]["code"], x["properties"]
		["nom"], "pink", x["properties"]["code"]])

	df = pd.DataFrame(df, columns=['code', 'nom', 'color', 'custom_data'])
	# df.astype({'id': 'int'}).dtypes
	df["code"] = df["code"].astype(np.int64)
	# df['code'] = df['code'].apply(lambda x: x.zfill(2))
	df_tot=ratio()
	df = df.merge(df_tot, left_on='code', right_on='reg')



# app.layout = html.Div([

	# html.Div([], id="map"),

	# dcc.Dropdown(id="dd_vaccin",
    # options=[
        # {'label': 'New York City', 'value': 1},
        # {'label': 'Montreal', 'value': 2},
        # {'label': 'San Francisco', 'value': 3},
		# {'label': 'tout', 'value': 0}
    # ],
	# multi=True
	# ) ,
	# html.Div([
		# dcc.Graph(id="figChart",
				  # config={'scrollZoom': False}, figure=figChart),
	# ], id="second_row"),
# html.Div([
		# dcc.Graph(id="treeChart2",
				  # config={'scrollZoom': False}, figure=fig),
	# ], id="treeChart"),
	# html.Div(['rrr'],id="aha"),
	# dcc.RangeSlider(
                    # id='time-rangeslider',
                    # min=date_min,
                    # max=date_max,
                    # value=[date_min, date_max]
                # )

# ])

# # @app.callback(
    # # dash.dependencies.Output('figChart', 'figure'),
    # # [dash.dependencies.Input('dd_vaccin', 'value'),
	# # dash.dependencies.Input('time-rangeslider', 'value')
	 # # ],prevent_initial_call=True)
# # def update_output(vaccin,date_range):
	# # print(date_range)
	# # data=make_data(liste_vaccin=vaccin,date_range=date_range)
	# # figChart = go.Figure(data=data, layout=layout)
	# # return figChart

# # @app.callback(
    # # dash.dependencies.Output('treeChart2', 'figure'),
    # # [dash.dependencies.Input('dd_vaccin', 'value'),
	# # dash.dependencies.Input('clickbutton', 'value')
	 # # ],prevent_initial_call=True)
# # def update_output(vaccin,date_range):
	# # print(date_range)
	# # data=make_data_tree_map(liste_vaccin=vaccin,date_range=date_range)
	# # # figChart = go.Figure(data=data, layout=layout)
	# # fig = go.Figure(go.Treemap(
		# # labels=labels,
		# # values=values,
		# # parents=parents,
		# # marker_colors=[choose_color(i) for i in range(len(labels))]
	# # ))
	# # return fig



