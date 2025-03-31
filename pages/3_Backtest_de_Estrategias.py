import streamlit as st
import time
import pandas as pd
import numpy as np
from utilities import *
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError) 
from advance_future_strategy import AdvanceFutureStrategy

# Definir las opciones del selector y sus descripciones
options = {
    '1m': '1 minute',
    '2m': '2 minutes',
    '5m': '5 minutes',
    '15m': '15 minutes',
    '30m': '30 minutes',
    '90m': '1 hour and 30 minutes',
    '1h': '1 hour',
    '1d': '1 day',
    '1wk': '1 week',
    '1mo': '1 month',
    '3mo': '3 months'
}

advance_strategy = AdvanceFutureStrategy()

st.set_page_config(page_title="AgentTrader Micro App", page_icon="📈")

confi = logo_up()

st.markdown(confi,
        unsafe_allow_html=True,
    )


with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Creating a login widget
try:
    if not st.session_state.get('authentication_status'):
        logo_login()
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    with st.sidebar:
        st.write(f'**Usuario**: *{st.session_state["name"]}*')
        authenticator.logout()
        st.markdown("---")
        
    st.markdown("# Cálculo de posición")
    st.sidebar.header("Cálculo de posiciones")
    st.subheader('Selecciona tus instrumentos y el tamaño de inversión en cada uno')

    df_symbols = pd.read_csv('data/futures_symbols_v1.csv', sep=',')

    df_symbols['large_name'] = df_symbols['Name'] + ' (' + df_symbols['Symbol'] + ')'
    symbol_list = df_symbols['large_name'].unique()

    # Checkbutton para seleccionar todos los instrumentos
    select_all = st.checkbox('Seleccionar todos')

    if select_all:
        symbol_selection = st.multiselect('Seleccionar instrumento', symbol_list, symbol_list)
    else:
        symbol_selection = st.multiselect('Seleccionar instrumento', symbol_list, ['E-mini S&P 500 (ES)', 
                                                                    'Crude Oil (CL)', 
                                                                    'E-mini Nasdaq-100 (NQ)'])
        
    # fig_candle_with_EMAS(self, symbol_data, symbol, emas_list)
    # Definir el valor inicial del slider

    # Crear el botón tipo slider
    #slider_value = st.toggle("Cambiar a capital deseado")

    # Diccionario para almacenar los valores de capital inicial
    capital_inicial = {}

    with st.expander('Configuración de capital inicial'):
        # Crear un input para cada selección de símbolo
        for symbol in symbol_selection:
            capital_inicial[symbol] = st.number_input(f'Capital inicial para {symbol}', value=100000.0, step=100.0, format="%.2f")
        
        total_capital = sum(list(capital_inicial.values()))
    list_symbols_filter = list(df_symbols[df_symbols['large_name'].isin(symbol_selection)]['Symbol'])
    list_symbols_load = list(df_symbols[df_symbols['large_name'].isin(symbol_selection)]['Symbol_data'])
    st.text("💹 Modifica los parametros para cálculo")

    # Dividir la página en dos columnas
    left_column, right_column = st.columns(2)
    # Obtener la opción seleccionada
    # Input para solicitar el porcentaje de riesgo

    with left_column:
        interval= st.selectbox('Selecciona el Time Frame:', options.keys(), format_func=lambda x: options[x], index=list(options.keys()).index('1wk'))
        risk_percentage = st.number_input("Porcentaje de riesgo (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%f")
        

    # Input para solicitar el capital inicial
    with right_column:
        initial_capital = st.number_input("Capital Total", value=total_capital, disabled=True)
    
            

        #
        # Selector de opciones
        capped_forecast = st.toggle("Capped Forecast")
    

    left_column1, right_column1 = st.columns(2)

    if capped_forecast:
        with left_column1:
            ema_1 = st.number_input("EWMA 1", min_value=1.0, max_value=1000.0, value=64.0, step=1.0)
            #st.write('Lambda value:', round(2 / (ema_1 + 1), 3))

        # Input para solicitar el capital inicial
        with right_column1:
        
            ema_2 = st.number_input("EWMA 2", min_value=0.0, max_value=1000.0,value=256.0, step=1.0)
            #st.write('Lambda value:', round(2 / (ema_2 + 1), 3))


    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    # Botón para calcular
if st.button("Start"):
    st.write(f"Cálculo de posiciones, con actualización a la última vela del time frame")
    advance_strategy.interval = interval
    advance_strategy.main(list_symbols_load)
    tau = risk_percentage /100
    
    for i in range(len(list_symbols_filter)):
        pro = round(((i+1)/ len(list_symbols_filter)), 2)
        status_text.text(f"{round(float(pro)*100)}% Complete")
        progress_bar.progress(pro)
        sym = list_symbols_filter[i]
        #-------------- Informacion del activo ------------------
        df_sym = df_symbols[df_symbols['Symbol'] == sym].reset_index(drop=True)
        #-------------------------------------------------------
        symbol_data = df_sym['Symbol_data'].iloc[0]
        multiplier = df_sym['Multiplier'].iloc[0]
        advance_strategy.capital = capital_inicial[df_sym['large_name'][0]]
        try:
            advance_strategy.data['returns', symbol_data] = advance_strategy.data['Close', symbol_data].pct_change()
                
                
            ew_st_dev = advance_strategy.data['returns', symbol_data].ewm(span=32, adjust=False).std()
            advance_strategy.data['standard_deviation', symbol_data] = 16*(0.3*ew_st_dev.rolling(window='2560D').mean()   #.ewm(span=10*256).mean() 
                                                + 0.7*ew_st_dev)  # See appendix B Carver
            
            if capped_forecast:
                slow_span = ema_2
                fast_span = ema_1
                advance_strategy.data[f'EMA{fast_span}', symbol_data] = advance_strategy.calcular_ema(symbol_data, fast_span)
                advance_strategy.data[f'EMA{slow_span}', symbol_data] = advance_strategy.calcular_ema(symbol_data, slow_span)
                    
                advance_strategy.data[f'EWMAC_{fast_span}_{slow_span}', symbol_data] = (advance_strategy.data[f'EMA{fast_span}', symbol_data] - 
                                                                            advance_strategy.data[f'EMA{slow_span}', symbol_data])
                advance_strategy.data['St_dev_daily_price_units', symbol_data] = advance_strategy.data['Close', symbol_data]*advance_strategy.data['standard_deviation', symbol_data]/16
                advance_strategy.data['capped_forecast', symbol_data] = advance_strategy.capped_forecast(advance_strategy.data[f"EWMAC_{fast_span}_{slow_span}", symbol_data].to_numpy(), 
                                                                                                    advance_strategy.data['St_dev_daily_price_units', symbol_data].to_numpy())

                advance_strategy.data['position_size', symbol_data] = advance_strategy.data[[("Close",symbol_data), 
                                                                    ("standard_deviation", symbol_data),
                                                                    ("capped_forecast", symbol_data)]].apply(lambda row: advance_strategy.calculate_position_size_with_forecast(
                                                                        row['capped_forecast', symbol_data],
                                                                        1, 1, tau, multiplier, 
                                                                        row['Close', symbol_data], 
                                                                        1, row['standard_deviation', symbol_data]), 
                                                                    axis=1)
                
            else:
                advance_strategy.data['position_size', symbol_data] = advance_strategy.data[[("Close",symbol_data), 
                                                                    ("standard_deviation", symbol_data)]].apply(lambda row: advance_strategy.calculate_position_size(
                                                                        1, 1, tau, multiplier, 
                                                                        row['Close', symbol_data], 
                                                                        1, row['standard_deviation', symbol_data],
                                                                        1), 
                                                                    axis=1)
                                                                    
            st.write(f"Cantidad de contratos a adquirir de {sym} :", round(advance_strategy.data['position_size', symbol_data].iloc[-1],2))
        except:
            pass
        

        
        
    
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
    


