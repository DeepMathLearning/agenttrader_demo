import streamlit as st
import subprocess
import os
import signal
import psutil
import pandas as pd
from utilities import logo_up, logo_login
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError) 

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
    #st.set_page_config(page_title='Calendario economico - Zenit', page_icon='📆')
    st.title('📆 Calendario Económico - Zenit')

    # Incluir el iframe de Investing.com
    html_code = """
    <iframe src="https://sslecal2.investing.com?ecoDayBackground=%23000000&columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&category=_employment,_economicActivity,_inflation,_credit,_centralBanks,_confidenceIndex,_balance,_Bonds&importance=1,2,3&features=datepicker,timezone,timeselector,filters&countries=5&calType=day&timeZone=9&lang=4" style="border-radius: 10px; overflow: hidden;" width="100%" height="700"  frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0">
    </iframe>
    """

    # Mostrar el HTML en Streamlit
    st.markdown(html_code, unsafe_allow_html=True)

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