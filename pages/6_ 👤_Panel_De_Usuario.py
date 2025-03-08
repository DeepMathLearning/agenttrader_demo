
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
from utils_functions import (create_portafolio_table, 
                             get_user_portfolios,
                             delete_portfolio,
                             show_portfolio_symbols)

create_and_fill_symbols_info_table('data/futures_symbols_v1.csv', output_csv_path="data/futures_symbol_info_v1.csv")

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

contract_month_options = [
    'ENE', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
    'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DEC'
]

symbols = []
teams = []

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
        
    elif st.session_state["username"] == 'zenittest':
        asig_accounts = accounts1 
    
    for acc in asig_accounts:
        symbols = symbols + account_info[acc]['symbol']
    
    for te in asig_accounts:    
        try:
            teams = teams + teams_info[te]['teams']
        except:
            pass
    symbols = list(set(symbols))
    teams = list(set(teams))
    
    st.markdown('## Información de la cuenta')  
    st.markdown(f'- **Username**: {st.session_state["username"]}')
    st.markdown(f'- **Nombre**: {st.session_state["name"]}')
    st.markdown("---")
    
    with st.expander('Configuración de portafolio'):
        
        # Obtener el username dinámicamente (simulación)
        username = st.session_state["username"]  
        
        # Mostrar los portafolios existentes
        st.subheader("Portafolios existentes")
        user_portfolios = get_user_portfolios(username)
        if user_portfolios:
            st.write("Tus portafolios creados:")
            for portfolio in user_portfolios:
                portfolio_name = portfolio[0]
                symbols_ = portfolio[1].strip("'").split("','")  # Convertir la cadena de símbolos en una lista

                col1, col2 = st.columns([4, 1])  # Crear columnas para organizar el contenido

                # Mostrar el nombre del portafolio con un botón para ver los símbolos
                with col1:
                    if st.button(portfolio_name, key=f"view-{portfolio_name}"):
                        show_portfolio_symbols(portfolio_name, symbols_)
                # Mostrar ícono para eliminar el portafolio
                with col2:
                    if st.button("🗑️", key=f"delete-{portfolio_name}"):
                        delete_portfolio(portfolio_name)
                        # Recargar la página para reflejar los cambios
                        st.rerun()
        else:
            st.write("No tienes portafolios configurados.")
            
        # Interfaz en Streamlit
        st.title("Crear portafolio")

        # Input para el nombre del portafolio
        portfolio_name = st.text_input("Nombre del portafolio", placeholder="Escribe el nombre del portafolio")

        # Input para la lista de símbolos
        symbols_input = st.text_area(
            "Lista de símbolos (separados por comas)",
            placeholder="Ingresa los símbolos separados por comas, por ejemplo: EOE,M6A,BSBY,AIGCI,MCD"
        )

        # Botón para crear la tabla e insertar datos
        if st.button("Insertar datos"):
            if portfolio_name and symbols_input:
                # Convertir el texto ingresado a una lista
                symbols_ = [symbol.strip() for symbol in symbols_input.split(",") if symbol.strip()]
                create_portafolio_table(username, portfolio_name, symbols_)
                 # Input para el nombre del portafolio
                # Limpiar los campos de entrada
                st.session_state["portfolio_name"] = ""
                st.session_state["symbols_input"] = ""
                st.rerun()
            else:
                st.warning("Por favor, proporciona un nombre de portafolio y una lista de símbolos.")
    
    
    
    with st.expander('Recursos asignados'):
        col1, col2, col3 = st.columns([0.3,0.3,0.5])
        with col1:
            st.markdown('### Cuentas asignadas')
            for i in asig_accounts:
                st.markdown(f'- {i}')
        with col2:
            st.markdown('### Simbolos asignados')
            for i in symbols:
                st.markdown(f'- {i}')
        with col3:
            st.markdown('### Equipos asignados')
            for i in teams:
                st.markdown(f'- {i}')
    
            
        
    # Configurar el DataFrame para que solo la columna 'ContractMonth' sea editable
    def make_editable(df, editable_columns):
        for col in df.columns:
            if col not in editable_columns:
                df[col] = df[col].astype(str)
        return df
    
    
    if (st.session_state["username"] in ['admin']):
        
        # editable_columns = ['ContractMonth']
        df_symbols_editable = get_filtered_data_by_symbols('data/futures_symbol_info_v1.csv', symbols)
        df_symbols_editable['ContractMonth'] = df_symbols_editable['ContractMonth'].astype(int).astype(str)
        # archie = df_symbols_editable[df_symbols_editable['symbol'] == 'ES'].reset_index(drop=True).to_dict(orient='records')[0]
        # st.write(archie['Multiplier'])
        #make_editable(df_symbols.copy(), editable_columns)

        # # Mostrar el DataFrame editable en Streamlit
        edited_df = st.data_editor(df_symbols_editable[['symbol', 'exchange', 'secType', 'ContractMonth', 'str_contract_month']], column_config={
            'symbol': st.column_config.TextColumn(disabled=True),
            'exchange': st.column_config.TextColumn(disabled=True),
            'secType': st.column_config.TextColumn(disabled=True),
            'str_contract_month': st.column_config.SelectboxColumn(options=contract_month_options),
            'ContractMonth': st.column_config.TextColumn()
        }, hide_index=True,use_container_width=True)

        if st.button("🟢 Actualizar data"):
            for index, row in edited_df.iterrows():
                new_symbol = generar_simbolo_futuro(row['symbol'], row['str_contract_month'],row['ContractMonth'])
                update_contract_month('data/futures_symbol_info_v1.csv', row['symbol'], new_symbol, row['str_contract_month'],row['ContractMonth'])
            
            st.success("Los datos fueron actualizados satisfactoriamente")
        # Mostrar el DataFrame editado
        # st.write("DataFrame Editado:")
        # st.dataframe(edited_df, hide_index=True, height=450, use_container_width=True)

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