from operator import index
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import io
from PIL import Image
import base64
from utilities import logo_up, get_chart_49243206, logo_login, create_and_fill_symbols_info_table
from threading import Thread
from threading import Timer
from datetime import datetime, timedelta
import multiprocessing
from account_info import *
from zenit_CRUCEEMAS_strategy import BotZenitCRUCEEMAS
from zenit_TRENDEMASCLOUD_strategy import BotZenitTRENDEMASCLOUD 
from zenit_strategy_bot import BotZenitTrendMaster
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utilities import *
from api_interface import initialize_db
import platform
import time
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError) 

from os import getpid
import random 

from streamlit_calendar import calendar

create_and_fill_symbols_info_table('data/futures_symbols_v1.csv', output_csv_path="futures_symbol_info_v1.csv")

futures_symbols_info = pd.read_csv("data/futures_symbols_v1.csv")

initialize_db()
# Obtener todos los registros

# Función para actualizar los eventos del calendario

st.set_page_config(page_title='Reporte de trades - Zenit', page_icon='📊')



calendar_resources = [
    {"id": "a", "building": "Building A", "title": "Room A"},
    {"id": "b", "building": "Building A", "title": "Room B"},
    {"id": "c", "building": "Building B", "title": "Room C"},
    {"id": "d", "building": "Building B", "title": "Room D"},
    {"id": "e", "building": "Building C", "title": "Room E"},
    {"id": "f", "building": "Building C", "title": "Room F"},
    {"id": "gasz", "building": "Building sss", "title": "Today profit"},
]

calendar_options = {
    "editable": "true",
    "navLinks": "true",
    "resources": calendar_resources,
    "selectable": "true",
}

fecha_hoy = datetime.now()

# Formatear la fecha
fecha_formateada = fecha_hoy.strftime("%Y-%m-%d")


mode = "daygrid"

calendar_options = {
    **calendar_options,
    "headerToolbar": {
        "left": "today prev,next",
        "center": "title",
        "right": "dayGridDay,dayGridWeek,dayGridMonth",
    },
    "initialDate": f"{fecha_formateada}",
    "initialView": "dayGridMonth",
}
   






confi = logo_up()

st.markdown(confi,
        unsafe_allow_html=True,
    )


# Crear una lista de opciones para los checkboxes



# Diccionario para almacenar el estado de los checkboxes
checkbox_account = {}
checkbox_strategy = {}
checkbox_trade_type = {}
checkbox_symbol = {}
checkbox_interval = {}
checkbox_date = {}


with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Creating a login widget
try:
    if not st.session_state.get('authentication_status'):
        logo_login()
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    if st.button('Actualizar página'):
        st.rerun()
    # Page title
    st.title('📊 Reporte de trades de los bots - Zenit')
    with st.sidebar:
        st.write(f'**Usuario**: *{st.session_state["name"]}*')
        authenticator.logout()
        st.markdown("---")
    df = fetch_all_records()
    df['datetime'] = pd.to_datetime(list(map(lambda x: x.replace('+00:00', ''), df['time'])))
    
    if (st.session_state["username"] in ['jemirsonramirez', 'admin']):
        asig_accounts = accounts 
    elif st.session_state["username"] == 'zenittest':
        asig_accounts = accounts1 
       
       
    df = df[df['account'].isin(asig_accounts)]  
    options_account = asig_accounts
    options_strategy = df['strategy'].unique()
    options_trade_type = df['trade_type'].unique()
    options_symbol = df['symbol'].unique()
    options_interval = df['interval'].unique()
    options_date = pd.to_datetime(df['time']).dt.date.unique()  
    # Crear los checkboxes en el sidebar
    with st.sidebar:
        st.sidebar.header("Filtros")
        with st.expander('Filtrar por Cuenta'):
            st.write("### Cuentas")
            for option in options_account:
                checkbox_account[option] = st.checkbox(option)
            selected_options = [option for option, state in checkbox_account.items() if state]
            if len(selected_options) > 0:
                df = df[df['account'].isin(list(selected_options))]
                
        with st.expander('Filtrar por Estrategia'):
            st.write("### Estrategias")
            for option in options_strategy:
                checkbox_strategy[option] = st.checkbox(option)
            selected_options1 = [option for option, state in checkbox_strategy.items() if state]
            if len(selected_options1) > 0:
                df = df[df['strategy'].isin(list(selected_options1))]
        
        with st.expander('Filtrar por Dirección'):
            st.write("### Dirección")
            for option in options_trade_type:
                checkbox_trade_type[option] = st.checkbox(option)
            selected_options1 = [option for option, state in checkbox_trade_type.items() if state]
            if len(selected_options1) > 0:
                df = df[df['trade_type'].isin(list(selected_options1))]
        
        with st.expander('Filtrar por Simbolo'):
            st.write("### Simbolo")
            for option in options_symbol:
                checkbox_symbol[option] = st.checkbox(option)
            selected_options1 = [option for option, state in checkbox_symbol.items() if state]
            if len(selected_options1) > 0:
                df = df[df['symbol'].isin(list(selected_options1))]
        
        with st.expander('Filtrar por Intervalo'):
            st.write("### Intervalo")
            for option in options_interval:
                checkbox_interval[option] = st.checkbox(option)
            selected_options1 = [option for option, state in checkbox_interval.items() if state]
            if len(selected_options1) > 0:
                df = df[df['interval'].isin(list(selected_options1))]
        
        with st.expander('Filtrar por Fecha'):
            st.write("### Fecha")
            try:
                col5, col6 = st.columns(2)
                with col5:
                    start_date = st.date_input('Inicio', df['datetime'].dt.date.min())
                with col6:
                    end_date = st.date_input('Fin', df['datetime'].dt.date.max()+timedelta(1))
                    
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                if start_date != end_date:
                    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date+timedelta(1))]
                else:
                    df = df[(df['datetime'] == start_date)]
            except:
                pass
        
        with st.expander('Filtrar por Hora'):
            st.write("### Hora")
            try:
                col7, col8 = st.columns(2)# Entradas para seleccionar el rango de hora
                
                
                with col7:
                    start_time = st.time_input('Inicio', value=datetime.strptime('00:00:00', '%H:%M:%S').time())
                with col8:
                    end_time = st.time_input('Fin', value=datetime.strptime('23:59:59', '%H:%M:%S').time())
                # Convertir las fechas y horas seleccionadas a datetime
                start_datetime = pd.Timestamp(start_time.strftime('%H:%M:%S')).time()
                end_datetime = pd.Timestamp(end_time.strftime('%H:%M:%S')).time()
                               

                
                df = df[(df['datetime'].dt.time >= start_datetime) & (df['datetime'].dt.time <= end_datetime)]
            except:
                pass

    df_metrics = calculate_trade_metrics(df)
    trade_metrics = df_metrics.dropna(axis=0).reset_index(drop = True)

    if trade_metrics.shape[0] > 0:
        futures_symbols_info['symbol'] = futures_symbols_info['Symbol']
        # Combinar los dataframes usando merge en la columna 'symbol'
        trade_metrics = pd.merge(trade_metrics, futures_symbols_info, on='symbol', how='left')
        
        # Calcular saldo_usd
        trade_metrics['saldo_usd'] = ((trade_metrics['pnl'] / trade_metrics['Tick']) * trade_metrics['Value']) * trade_metrics['contracts']

        win_usd = round(trade_metrics[trade_metrics['saldo_usd'] > 0]['saldo_usd'].sum(), 2)
        loss_usd = round(trade_metrics[trade_metrics['saldo_usd'] < 0]['saldo_usd'].sum(), 2)
        total_usd = round(win_usd + loss_usd, 2)    
        
        st.title(f'Trades completados: {trade_metrics.shape[0]}')
        
        cola, colb, colc = st.columns(3)
        with cola:
            st.write("Rendimiento Total")
            if total_usd >= 0:
                st.write(f'<p style="color:green; font-size:30px;"> {total_usd} USD</p>', unsafe_allow_html=True)
            else:
                st.write(f'<p style="color:red; font-size:30px;"> {total_usd} USD</p>', unsafe_allow_html=True)
                
        with colb:
            st.write("Trades ganados")
            st.write(f'<p style="color:green; font-size:20px;"> +{win_usd} USD</p>', unsafe_allow_html=True)
        with colc:
            st.write("Trades perdidos")
            st.write(f'<p style="color:red; font-size:20px;">{loss_usd} USD</p>', unsafe_allow_html=True)
        
        with st.expander('Data de trades completados'):
            st.dataframe(trade_metrics, hide_index=True, height=450, use_container_width=True)
        ######################### Calcular el porcentaje de wins y losses ####################
        trade_result_counts = trade_metrics['trade_result'].value_counts()


        # Asignar colores personalizados para Win y Loss
        colors = ['green' if result == 'Win' else 'red' for result in trade_result_counts.index]

        # Crear gráfico de torta con Plotly
        fig = go.Figure(data=[go.Pie(labels=trade_result_counts.index, values=trade_result_counts.values, hole=0.3, marker=dict(colors=colors))])

        # Configurar el diseño del gráfico
        fig.update_layout(
            title_text="Distribución de Resultados de Trades",
            annotations=[dict(text='Trades', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        #####################################################################################
        
        win_strategy = trade_metrics[trade_metrics['trade_result'] == 'Win']['strategy'].value_counts()
        # Colores verdes para trades ganadores
        green_colors = [ '#32CD32', '#008000', '#006400', '#9ACD32']
        # Crear gráfico de torta con Plotly
        fig1 = go.Figure(data=[go.Pie(labels=win_strategy.index, values=win_strategy.values, hole=0.3, marker=dict(colors=green_colors))])

        # Configurar el diseño del gráfico
        fig1.update_layout(
            title_text="Trades ganadores por estrategias",
            annotations=[dict(text='Estrategias', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        ####################################################################################

        loss_strategy = trade_metrics[trade_metrics['trade_result'] == 'Loss']['strategy'].value_counts()
        # Colores rojos para trades perdedores
        red_colors = ['#FF0000', '#DC143C', '#8B0000', '#FF6347']
        # Crear gráfico de torta con Plotly
        fig2 = go.Figure(data=[go.Pie(labels=loss_strategy.index, values=loss_strategy.values, hole=0.3, marker=dict(colors=red_colors))])

        # Configurar el diseño del gráfico
        fig2.update_layout(
            title_text="Trades perdedores por estrategias",
            annotations=[dict(text='Estrategias', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

        ############################# Analisis por hora ####################################

        trade_metrics['time'] = pd.to_datetime(trade_metrics['time'])

        # Extraer la hora del día de la columna de tiempo
        trade_metrics['hour'] = trade_metrics['time'].dt.hour

        # Crear histogramas de wins y losses
        win_data = trade_metrics[trade_metrics['trade_result'] == 'Win']
        w_d = win_data['hour'].value_counts()
        loss_data = trade_metrics[trade_metrics['trade_result'] == 'Loss']
        l_d = loss_data['hour'].value_counts()

        # Crear el histograma para los wins con Plotly Go
        fig_win = go.Figure()
        fig_win.add_trace(go.Bar(
            x=w_d.index,
            y=w_d.values,
            name='Wins',
            marker_color='rgba(0, 128, 0, 0.6)'  # Verde opaco
        ))

        fig_win.update_layout(
            title='Trades ganados por hora del día',
            xaxis_title='Hora del día',
            yaxis_title='Número de trades',
            bargap=0.2
        )

        # Crear el histograma para los losses con Plotly Go
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Bar(
            x=l_d.index,
            y= l_d.values,
            name='Losses',
            marker_color='rgba(255, 0, 0, 0.6)'  # Rojo opaco
        ))

        fig_loss.update_layout(
            title='Trades perdidos por hora del día',
            xaxis_title='Hora del día',
            yaxis_title='Número de trades',
            bargap=0.2
        )


        ############################### Analisis por dia de la semana    ######################################################

        trade_metrics['weekday'] = trade_metrics['time'].dt.strftime('%A')  # Obtener el nombre del día de la semana en español

        # Contar los wins y losses por día de la semana
        trade_metrics1 = trade_metrics.groupby(['weekday', 'trade_result']).size().unstack(fill_value=0)
        trade_metrics1 = trade_metrics1.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        # Crear las figuras de Plotly
        fig_win_d = go.Figure()
        fig_loss_d = go.Figure()

        # Gráfico de Wins por día de la semana
        try:
            fig_win_d.add_trace(go.Bar(name='Win', x=trade_metrics1.index, y=trade_metrics1['Win'], marker_color='green'))
            fig_win_d.update_layout(
                title="Trades ganados por día de la semana",
                xaxis_title="Día de la semana",
                yaxis_title="Cantidad",
            )
        except:
            pass
        
        try:
            # Gráfico de Losses por día de la semana
            fig_loss_d.add_trace(go.Bar(name='Loss', x=trade_metrics1.index, y=trade_metrics1['Loss'], marker_color='red'))
            fig_loss_d.update_layout(
                title="Trades perdidos por día de la semana",
                xaxis_title="Día de la semana",
                yaxis_title="Cantidad",
            )
        except:
            pass

        
            
        with st.expander('Resultados de trades Win-Loss'):
            # Ejemplo de número
            # numero = -15.2

            # # Configuración con st.write()
            # if numero < 0:
            #     st.write(f'<p style="color:red; font-size:15px;">{numero}</p>', unsafe_allow_html=True)
            # else:
            #     st.write(f'<p style="color:green; font-size:15px;">{numero}</p>', unsafe_allow_html=True)
                        
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns([2,2])
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig2, use_container_width=True)

        with st.expander('Resultados de trades por hora'):
            # Mostrar el gráfico en Streamlit
            col11, col22 = st.columns([2,2])
            with col11:
                
                st.plotly_chart(fig_win, use_container_width=True)
                
            with col22:
            
                st.plotly_chart(fig_loss, use_container_width=True)
                
        with st.expander('Resultados de trades por día de la semana'):
            # Mostrar el gráfico en Streamlit
            col111, col222 = st.columns([2,2])
            with col111:
                
                st.plotly_chart(fig_win_d, use_container_width=True)
                
            with col222:
                st.plotly_chart(fig_loss_d, use_container_width=True)
        
        
        with st.expander('Calendario de trades'):
            daily_stats = prepare_data_for_plot(trade_metrics)
            # Crear el gráfico de barras
            # Crear el gráfico de barras para el total de trades
            fig_trades = go.Figure()
            fig_trades.add_trace(go.Bar(
                x=daily_stats['time_dt'],
                y=daily_stats['total_trades'],
                name='Total de Trades',
                marker_color='blue'
            ))
            fig_trades.update_layout(
                title='Total de Trades por Día',
                xaxis=dict(title='Fecha'),
                yaxis=dict(title='Total de Trades')
            )

            # Crear el gráfico de barras para el saldo
            fig_saldo = go.Figure()
            fig_saldo.add_trace(go.Bar(
                x=daily_stats['time_dt'],
                y=daily_stats['total_saldo'],
                name='Total Ganado/Perdido (USD)',
                marker_color=daily_stats['total_saldo'].apply(lambda x: 'green' if x >= 0 else 'red')
            ))
            fig_saldo.update_layout(
                title='Total Ganado/Perdido por Día',
                xaxis=dict(title='Fecha'),
                yaxis=dict(title='Total Ganado/Perdido (USD)')
            )

            col_s, col_t = st.columns(2)
            # Mostrar los gráficos en Streamlit
            with col_s:
                st.plotly_chart(fig_saldo, use_container_width=True)
            with col_t:
                st.plotly_chart(fig_trades, use_container_width=True)
            
            if st.button("📆 Ver Calendario"):
                calendar_events_create(trade_metrics, mode, calendar_options)
        
        
        with st.expander('Actividad almacenada en la base de datos'):
            # Configurar la aplicación de Streamlit
            st.title('Actividad general')
            st.write('Actividades de los bots almacenados en la base de datos:')


            # Mostrar el DataFrame en Streamlit
            st.dataframe(df[['account','trade_id', 'strategy', 'interval', 'trade_type', 'action','symbol', 'price', 'time', 'contracts']], 
                        hide_index=True, 
                        height=450, 
                        use_container_width=True)
        
        
        
        
        

    else:
        st.title("No hay data para mostrar")
elif st.session_state["authentication_status"] is False:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    st.error('Usuario o contraseña incorrectos')
elif st.session_state["authentication_status"] is None:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    st.warning('Ingrese usuario y contraseña asignados')




