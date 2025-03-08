import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import io
from PIL import Image
import base64
from utilities import *
from threading import Thread
from threading import Timer
from datetime import datetime
import multiprocessing
from account_info import *
from zenit_CRUCEEMAS_strategy import BotZenitCRUCEEMAS
from zenit_TRENDEMASCLOUD_strategy import BotZenitTRENDEMASCLOUD 
from zenit_strategy_bot import BotZenitTrendMaster
import streamlit.components.v1 as components
from utilities import *
import platform


from os import getpid
import random 
import yaml
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError) 

create_and_fill_symbols_info_table('data/futures_symbols_v1.csv', output_csv_path="data/futures_symbol_info_v1.csv")
df_symbols_editable = pd.read_csv('data/futures_symbol_info_v1.csv')

directorio = os.getcwd()

checkbox_account = {}
checkbox_strategy = {}
checkbox_trade_type = {}
checkbox_symbol = {}
checkbox_interval = {}
checkbox_date = {}


st.set_page_config(page_title='Bot Ejecutables - Zenit', page_icon='📊')


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
    with st.sidebar:
        st.write(f'**Usuario**: *{st.session_state["name"]}*')
        authenticator.logout()
        st.markdown("---")
    
    if (st.session_state["username"] in ['jemirsonramirez', 'admin']):
        asig_accounts = accounts 
        t_type = ['long', 'short', 'smart']
    elif st.session_state["username"] == 'zenittest':
        remove_last_li_script = """
            <script>
                const sidebarNav = document.querySelectorAll('ul[data-testid="stSidebarNavItems"] li');
                if (sidebarNav.length > 0) {
                    sidebarNav[sidebarNav.length - 2].style.display = 'none';
                }
            </script>
        """

        # Ejecutar el código JavaScript en la aplicación
        st.markdown(remove_last_li_script, unsafe_allow_html=True)

        asig_accounts = accounts1 
        t_type = ['long', 'smart']
    
    if st.button('Actualizar página'):
        st.rerun()
    # Page title
    st.title('📊 Ejecutar Bots - Zenit')
    st.sidebar.header("Crear Bot")
    
    with st.expander('Sobre este microservicio'):
        st.markdown('**Que puede hacer este microservicio?**')
        st.info('Esta aplicación está diseñada para gestionar y ejecutar los bots de trading de Zenit Capital.')
        st.markdown('**Como usar este microservicio?**')
        st.warning('Para usar este servicio solo debe seleccionar el activo que desee tradear, la cuenta de IB permitida, el tipo de estrategia y oprimir crear.')
    
    st.subheader('Modifica los parámetros del bot')

    df_symbols = pd.read_csv('data/futures_symbols_v1.csv', sep=',')

    df_symbols['large_name'] = df_symbols['Name'] + ' (' + df_symbols['Symbol'] + ')'
    symbol_list = df_symbols['large_name'].unique()


    st.text("☑️ Configura tu bot")

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Inputs en la primera columna
    with col1:

        ib_account = st.selectbox(
            'Selecciona una cuenta de IB autorizada',
            options=asig_accounts,
            index=0
        )
        ticker = st.selectbox(
            'Tickers autorizados',
            options=account_info[ib_account]['symbol'],
            index=0
        )
        df_sym_info = df_symbols_editable[df_symbols_editable['symbol'] == ticker].reset_index(drop=True).to_dict(orient='records')[0]
        # Input tipo fecha
        contract_date = st.text_input('Fecha de vencimiento de contrato', value=df_sym_info['ContractMonth'], disabled=True)
        
    
        ip = st.text_input('IP de TWS en local', value=account_info[ib_account]['ip'])
        
        port = st.text_input('Puerto de TWS o Gateway', value=account_info[ib_account]['port'])

    # Inputs en la segunda columna
    with col2:
        strategy = st.selectbox(
            'Seleccionar estrategia',
            options=account_info[ib_account]['info_by_symbol'][ticker]['strategies'],
            index=0
        )
        # if strategy != 'TREND_MASTER':
        #     op_list = strategies_info[strategy]['accept_trade_list']
        #     op_list.append('smart')
        # else:
        #     op_list = strategies_info[strategy]['accept_trade_list']
        
        trade_type = st.selectbox(
            'Dirección de trade',
            options=t_type,
            index=0
        )

        
        if (st.session_state["username"] in ['jemirsonramirez', 'admin']):
            intervals = strategies_info[strategy]["interval"]
            quantity = st.number_input('Cantidad de contratos', min_value=1, step=1, value=1)
        elif st.session_state["username"] == 'zenittest':
            intervals = ['1m', '5m', '15m', '1h']
            quantity = st.number_input('Cantidad de contratos', min_value=1,max_value=10, step=1, value=1)
            
        interval = st.selectbox(
            'Timeframe autorizado',
            options=intervals,
            index=0
        )
        
        if trade_type == 'smart':
            smart_interval = st.selectbox(
                        'Timeframe para la capa smart',
                        options=['auto', '10m', '1h', '1d'],
                        index=0
                    )
        else:
            smart_interval = 'auto'
        

    #### Bots arguments
        # --symbol MESM4 
        # --exchange CME 
        # --secType FUT 
        # --client 2 
        # --trading_class MES 
        # --multiplier 5 
        # --lastTradeDateOrContractMonth 20240621 
        # --is_paper False 
        # --interval 1m 
        # --quantity 60 
        # --account DU7186454 
        # --accept_trade long 

    
    symbol = df_sym_info['symbol_ib']
    exchange = df_sym_info['exchange']
    secType = df_sym_info['secType']
    client = random.randint(20, 900)
    trading_class = ticker 
    multiplier = df_sym_info['Multiplier']
    #lastTradeDateOrContractMonth = symbols_info[ticker]['ContractMonth']
    is_paper = account_info[ib_account]['is_paper']
    account = ib_account
    #port = account_info[ib_account]['port']

    if strategy == 'TREND_MASTER':
        accept_trade = 'ab'
    else:
        accept_trade = trade_type

    hora_ejecucion = datetime.now().strftime('%Y%m%d_%H%M%S')

    if accept_trade == 'smart':
        with_trend_study = True
    else:
        with_trend_study = False

    args = {
        'symbol': symbol,
        'exchange': exchange,
        'secType': secType,
        'client': client,
        'trading_class': trading_class,
        'multiplier': multiplier,
        'lastTradeDateOrContractMonth': contract_date,
        'is_paper': is_paper,
        'interval': interval,
        'quantity': quantity,
        'account': account,
        'accept_trade': accept_trade,
        'port': port,
        'ip':'127.0.0.1',
        'currency':'USD',
        'strategy': strategy,
        "hora_ejecucion": hora_ejecucion,
        "ip" : ip,
        "with_trend_study": with_trend_study,
        "smart_interval" : smart_interval
    }

    if strategy != 'TREND_MASTER':
        commands = [
            f'''python {strategies_info[strategy]['file_name']} --ip {ip} --symbol {symbol} --exchange {exchange} --secType {secType} --client {client} --trading_class {trading_class} --multiplier {multiplier} --lastTradeDateOrContractMonth {contract_date} --is_paper {is_paper} --interval {interval} --quantity {quantity} --account {account} --accept_trade {accept_trade} --port {port} --hora_ejecucion {hora_ejecucion} --with_trend_study {with_trend_study} --smart_interval {smart_interval}'''
        ]
    else:
        commands = [
            f'''python {strategies_info[strategy]['file_name']} --ip {ip} --symbol {symbol} --exchange {exchange} --secType {secType} --client {client} --trading_class {trading_class} --multiplier {multiplier} --lastTradeDateOrContractMonth {contract_date} --is_paper {is_paper} --interval {interval} --quantity {quantity} --account {account} --accept_trade {accept_trade} --port {port} --hora_ejecucion {hora_ejecucion} --smart_interval {smart_interval}'''        
            ]

    # Botón para enviar el formulario
    if st.button('Crear Bot'):
        try:
            
            run_custom_commands(commands)
            st.success('Bot iniciado con exito!')

            st.write('Ticker:', trading_class)
            st.write('Cuenta de IB:', ib_account)
            st.write('Estrategia:', strategy)
            st.write('Cantidad de contratos:', quantity)
            st.write('Tipo de trade:', trade_type)

        except Exception as e:
            st.write(f'Error: {e}')
            st.error("OOPS!!, ha ocurrido un error: Asegúrate de encender el Trader Workstation y que la cuenta seleccionada sea la correcta")



    processes = load_processes()
    if len(processes) > 0:

        df_ = pd.DataFrame(processes)

        df = df_[df_['account'] == ib_account].sort_values('symbol').reset_index(drop=True)
        
        
        with st.sidebar:
            
            options_account = asig_accounts
            options_strategy = df['strategy'].unique()
            options_trade_type = df['trade_type'].unique()
            options_symbol = df['symbol'].unique()
            options_interval = df['interval'].unique() 
            
            # Crear los checkboxes en el sidebar
            #df = df[df['account'].isin(asig_accounts)]  
            
            st.sidebar.header("Filtros")
            
            with st.expander('Filtrar por Simbolo'):
                st.write("### Simbolo")
                for option in options_symbol:
                    checkbox_symbol[option] = st.checkbox(option)
                selected_options1 = [option for option, state in checkbox_symbol.items() if state]
                if len(selected_options1) > 0:
                    df = df[df['symbol'].isin(list(selected_options1))]
                    
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
            
            with st.expander('Filtrar por Intervalo'):
                st.write("### Intervalo")
                for option in options_interval:
                    checkbox_interval[option] = st.checkbox(option)
                selected_options1 = [option for option, state in checkbox_interval.items() if state]
                if len(selected_options1) > 0:
                    df = df[df['interval'].isin(list(selected_options1))]
            
        
        
        st.subheader(f'Bots Activos: {df.shape[0]}')
        
        cola, colb = st.columns([8,4])
        # with cola:
        #     with st.expander('Agregar filtros'):
        #         # Lista de elementos para los checkboxes
        #         items = [
        #                 # ("Cuenta", 'account'), 
        #                 ("Intervalo", 'interval'), 
        #                 ("Ticker", "symbol"), 
        #                 ("Estrategia", 'strategy'), 
        #                 ("Dirección", 'trade_type')
        #                 ]

        #         # Diccionario para almacenar el estado de los checkboxes
        #         checkbox_states = {}
        #         filter_values = {}
                            
        #         # Crear los checkboxes
        #         for item in items:
        #             checkbox_states[item[1]] = st.checkbox(item[0])
        #             if checkbox_states[item[1]]:
        #                 filter_values[item[1]] = st.selectbox(
        #                                             f'Selecciona un(a) {item[0]}',
        #                                             options=df[item[1]].unique(),
        #                                             index=0
        #                                         )
        #                 #st.write("Select:", checkbox_states[item[1]])
                    

        #         # # Mostrar el estado de los checkboxes seleccionados
        #         # st.write("Checkbox States:", checkbox_states)
        #         # st.write("Filter States:", filter_values)

        #         # # Mostrar los elementos seleccionados
        #         # selected_items = [item for item, checked in checkbox_states.items() if checked]
        #         # st.write("Selected Items:", selected_items)
        #         colaa, colbb = st.columns(2)
        #         with colaa:
        #             if st.button("🗄️ Filtrar"):
        #                 # Crear una condición inicial como True
        #                 mask = pd.Series([True] * len(df))
                        
        #                 # Aplicar cada condición usando &
        #                 for col, val in filter_values.items():
        #                     if col in df.columns:
        #                         mask = mask & (df[col] == val)
                        
        #                 # Filtrar el DataFrame usando la máscara
        #                 df = df[mask]
                            
        #         with colbb:
        #             if st.button("🧹 Limpiar filtro"):
        #                 st.rerun()
        with colb:
            if st.button("🟥 Detener Todos"):
                for i in df.index:
                    kill_custom_process(df['pid'][i])
                
                df_new = df_[df_['account'] != ib_account].reset_index(drop=True)
                if len(df_new) == 0:
                    processes.clear()
                    save_processes(processes)
                else:
                    df_new.to_csv(PROCESS_FILE, index=False)
                    
                st.success(f'Bots detenidos con éxito!')
                st.rerun()

        # Mostrar los nombres de las columnas
        col0,col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1.5,2.5, 2,2,1,2,2,2,2])
        
        with col0:
            st.markdown("<h1 style='font-size:15px;'>Id</h1>", unsafe_allow_html=True)
        with col1:
            st.markdown("<h1 style='font-size:15px;'>Account</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h1 style='font-size:15px;'>Interval", unsafe_allow_html=True)
        with col3:
            st.markdown("<h1 style='font-size:15px;'>Ticker", unsafe_allow_html=True)
        with col4:
            st.markdown("<h1 style='font-size:15px;'>Size", unsafe_allow_html=True)
        with col5:
            st.markdown("<h1 style='font-size:15px;'>Trade", unsafe_allow_html=True)
        with col6:
            st.markdown("<h1 style='font-size:15px;'>Strategy", unsafe_allow_html=True)
        with col7:
            st.markdown("<h1 style='font-size:15px;'>Activity", unsafe_allow_html=True)
        with col8:
            st.markdown("<h1 style='font-size:15px;'>Action", unsafe_allow_html=True)

        # Mostrar la tabla y los botones
        for i, row in df.iterrows():
            col0, col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1.5,2.5, 2,2,1,2,2,2,2])
            with col0:
                st.write(f"{row['client']}")
            with col1:
                st.write(f"{row['account']}")
            with col2:
                st.write(f"{row['interval']}")
            with col3:
                st.write(f"{row['symbol']}")
            with col4:
                st.write(f"{row['size_contracts']}")
            with col5:
                st.write(f"{row['trade_type']}")
            with col6:
                st.write(f"{row['strategy']}")
            with col7:
                # Escribe el enlace HTML con el atributo target="_blank"
                html_link = f"{directorio}/bot_activity/{row['strategy']}_{row['trade_type']}_{row['interval']}_{row['symbol']}_{row['hora_ejecucion']}.html"
                #f"<a href='file://{directorio}/bot_activity/{row['strategy']}_{row['trade_type']}_{row['interval']}_{row['symbol']}_{row['hora_ejecucion']}.html' target='_blank'>Abrir enlace en una nueva pestaña</a>"
                
                if st.button("Show", key=f"show_{row['pid']}"):
                    bot_activity(html_link)
                #components.html(source_code)
                # Mostrar el enlace HTML en la aplicación
                #st.write(html_link, unsafe_allow_html=True)
                #st.write(f"[Fig](./bot_activity/{row['strategy']}_{row['trade_type']}_{row['interval']}_{row['symbol']}_{row['hora_ejecucion']}.html)")
            with col8:
                if st.button(f'Stop', key=f'kill_custom_{i}'):
                    kill_custom_process(row['pid'])
                    st.success(f'Bot detenido con éxito!')
                    df_1 = df_[df_['pid'] != row['pid']]
                    df_1.to_csv(PROCESS_FILE, index=False)
                    st.rerun()
                
                    
    else:
        st.write(f"No existen procesos abiertos")
    


#st.info(f"Sistema operativo: {os_platform}")

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
