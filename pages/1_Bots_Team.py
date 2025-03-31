import streamlit as st
import time
import pandas as pd
import numpy as np
from utilities import *
from account_info import *
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError) 
import time
# Definir las opciones del selector y sus descripciones

#create_and_fill_symbols_info_table('zenit_oms.db', 'data/futures_symbols_v1.csv', table_name="symbols_info")

# Conectar a la base de datos

# conn = sqlite3.connect('zenit_oms.db')
# # Realizar la consulta SQL
# query = f"SELECT * FROM symbols_info"
# df_symbols_editable = pd.read_sql_query(query, conn)
# df_symbols_editable.to_csv('data/futures_symbol.csv')
# # Cerrar la conexión
# conn.close()
create_and_fill_symbols_info_table('data/futures_symbols_v1.csv', output_csv_path="data/futures_symbol_info_v1.csv")


df_symbols_editable = pd.read_csv('data/futures_symbol_info_v1.csv')

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



checkbox_account = {}
checkbox_strategy = {}
checkbox_trade_type = {}
checkbox_symbol = {}
checkbox_interval = {}
checkbox_date = {}

directorio = os.getcwd()

st.set_page_config(page_title='Bot Ejecutables', page_icon='📊')



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
    if st.button('Actualizar página'):
        st.rerun()
    with st.sidebar:
        st.write(f'**Usuario**: *{st.session_state["name"]}*')
        authenticator.logout()
        st.markdown("---")
        
    if (st.session_state["username"] in ['jemirsonramirez', 'admin']):
        asig_accounts = accounts 
        st.write('**Información general sobre los equipos:**')
        with st.expander('**Equipo DTC Beta Live**'):
            # Crear dos columnastea
            col11, col22 = st.columns(2)
            with col11:
                st.info('- Long: MES')
                st.info('- Temporalidades:')
                st.info('- Estrategias:')
            with col22:
                st.info('- Short: ES')
                st.info('1m, 5m, 15m')
                st.info('TA, Trend EMAS CLoud, Trend Master')
                
            col111, col222, col333, col444, col555 = st.columns(5)
            with col111:
                st.markdown('**Simbolo**')
                st.info('- MES')
                st.info('- MES')
                st.info('- MES')
                st.info('- ES')
                st.info('- ES')
            with col222:
                st.markdown('**Estrategia**')
                st.info('TA')
                st.info('TrendEMAS')
                st.info('Trend Master')
                st.info('TA')
                st.info('TrendEMAS')
            with col333:
                st.markdown('**TimeFrame**')
                st.info('1m y 5m')
                st.info('5m')
                st.info('5m')
                st.info('1m y 5m')
                st.info('5m')
            with col555:
                st.markdown('**Size**')
                st.info('30')
                st.info('40')
                st.info('40')
                st.info('3')
                st.info('4')
            with col444:
                st.markdown('**Trade**')
                st.info('Long')
                st.info('Long')
                st.info('Long')
                st.info('Short')
                st.info('Short')
        with st.expander('**Equipo DTC Beta Paper**'):
            # Crear dos columnas
            col11, col22 = st.columns(2)
            with col11:
                st.info('Long: MES, MNQ, M2K, MGC, MCL')
            with col22:
                st.info('Short: ES, NQ, RTY, GC, CL')
            st.info('- Temporalidades: 1m, 5m, 15m')
            st.info('- Estrategias: TA, Trend EMAS CLoud, Trend Master')
            st.markdown('**Sizes - Long**')
            st.info('- TA 1m, 5m: 30')
            st.info('- Trend EMAS Cloud 5m: 40')
            st.info('- Trend Master  5m: 40')
            st.markdown('**Sizes - Short**')
            st.info('- TA 1m, 5m: 3')
            st.info('- Trend EMAS Cloud 5m: 4') 
    elif st.session_state["username"] == 'zenittest':
        asig_accounts = accounts1 
        
    st.markdown("# Ejecutar Bots en equipo")
    st.sidebar.header("Ejecutando equipo")
        

    st.subheader('Selecciona la cuenta y el equipo a ejecutar')

    st.text("☑️ Configura tu equipo de bots")

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Inputs en la primera columna
    with col1:

        ib_account = st.selectbox(
            'Selecciona una cuenta de IB autorizada',
            options=asig_accounts,
            index=0
        )
    # Inputs en la segunda columna
    with col2:
        ip = st.text_input('IP de TWS en local', value=account_info[ib_account]['ip'])
        try:
            team = st.selectbox(
                'Equipos asignado a la cuenta',
                options= teams_info[ib_account]['teams'],
                index=0
            )
        except:
            team = st.selectbox(
                'Equipos asignado a la cuenta',
                options= [],
                index=0
            )
            st.warning('Esta cuenta no tiene un equipo asignado')
        
    hora_ejecucion= datetime.now().strftime('%Y%m%d_%H%M%S')
    args = {
        "account": ib_account,
        "port":account_info[ib_account]['port'],
        "hora_ejecucion": hora_ejecucion,
        "ip": ip

    }
    try:
        commands = teams_commands(args, team,df_symbols_editable)
    except:
        pass
    # Botón para enviar el formulario
    if st.button('🫂 Ejecutar equipo'):
        try:
            
            run_custom_commands(commands)
            st.success(f'Equipo {team} iniciado con exito!')

            st.write('Cuenta de IB:', ib_account)

        except Exception as e:
            #st.write(f'Error: {e}')
            st.error("OOPS!!, ha ocurrido un error: Asegúrate de encender el Trader Workstation y que la cuenta seleccionada tenga equipos asignados")



    def initialize_checkboxes(options, key):
        if key not in st.session_state:
            st.session_state[key] = {option: False for option in options}
    
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

