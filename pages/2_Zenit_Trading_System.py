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
from utils_functions import *
import threading
from datetime import datetime

today_str_date = datetime.now().date().strftime("%Y%m%d")
symbol_info = load_data_from_db(db_name="data/zenit_future_instrument.db",
                               table_name="general_futures_info_carver")

st.set_page_config(page_title='Zenit Trading System', page_icon='💸')

if "show_table" not in st.session_state:
    st.session_state["show_table"] = False  # Inicializar show_table

if "today_generate" not in st.session_state:
    st.session_state["today_generate"] = False  # Inicializar show_table

if "update_position" not in st.session_state:
    st.session_state["update_position"] = False  # Inicializar show_table

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
username = st.session_state["username"]
port = get_user_portfolios(username)
port_dict = [{'portfolio': x[0], 'symbols': x[1]} for x in port]

print(port_dict)

if st.session_state["authentication_status"]:
    # if st.button('Actualizar página'):
    #     st.rerun()
    with st.sidebar:
        st.write(f'**Usuario**: *{st.session_state["name"]}*')
        authenticator.logout()
        st.markdown("---")
    
    # if (st.session_state["username"] in ['jemirsonramirez', 'admin']):
    # elif st.session_state["username"] == 'zenittest':
    
    st.markdown("# Gestión Dinámica de Posiciones de Portafolio")
    st.sidebar.header("Ejecutando")
    
    with st.expander("📊 Secuencia del Sistema de Trading basado en Carver"):
        st.markdown("### Descripción General")
        st.write(
            """
            Este sistema aplica estrategias basadas en medias móviles exponenciales (EWMAs) y estudios de *carry* 
            para generar decisiones de trading sistemático. Utiliza parámetros optimizados como volatilidad objetivo 
            y correlaciones para determinar el tamaño de posiciones y gestionar un portafolio diversificado.
            """
        )
        
        st.markdown("### 1. Selección Inicial de Símbolos")
        st.write(
            """
            - Se define una lista inicial de activos que representa los instrumentos a evaluar.
            - Filtra los símbolos en función de datos de expiración y disponibilidad desde la base de datos.
            """
        )

        st.markdown("### 2. Parámetros del Sistema")
        st.write(
            """
            - **Capital Inicial**
            - **Volatilidad Objetivo**
            - **Período de Retroceso**: 36 días para el cálculo de indicadores.
            - **Pesos de EWMAs y Carry**: 60% EWMAs y 40% Carry.
            - **Pesos Relativos**: Calculados con correlaciones entre EWMAs.
            """
        )

        st.markdown("### 3. Cálculos Derivados")
        st.write(
            """
            - **Volatilidad Diaria Objetivo**: Ajustada por raíz cuadrada de días de trading (256 días/año).
            - **Pesos de EWMAs**: Generados en función de correlaciones.
            - **Forecast Scalars**: Calculados con `ewmac` pares para optimizar señales.
            """
        )

        st.markdown("### 4. Generación de Contratos")
        st.write(
            """
            - Cada símbolo se asocia a datos relevantes como:
            - Descripción, región, clase de activo, moneda, y tamaño de punto.
            - Información adicional como *carry symbols* y fechas de expiración también se incluye.
            """
        )

        st.markdown("### 5. Clasificación por Sectores")
        st.write(
            """
            - Los símbolos se agrupan según su clase de activo (ej. *commodities*, índices, monedas).
            - Esto permite una diversificación y manejo del riesgo eficiente.
            """
        )

        st.markdown("### 6. Ejecución del Sistema")
        st.write(
            """
            - **Función `process_multiple_assets`**:
            - Integra todos los datos calculados (contratos, sectores, volatilidad objetivo, etc.).
            - Ejecuta estrategias basadas en EWMAs y *carry* para generar señales de trading.
            """
        )
    
    
    
    
    # Extraer la lista de nombres de portafolios
    portfolio_names = [portfolio['portfolio'] for portfolio in port_dict]

    # Crear un selectbox con la lista de portafolios
    selected_portfolio = st.selectbox("Selecciona un portafolio", options=portfolio_names)

    # Mostrar los símbolos del portafolio seleccionado
    if selected_portfolio:
        selected_symbols = next(
            (portfolio['symbols'] for portfolio in port_dict if portfolio['portfolio'] == selected_portfolio),
            None
        )
        symbols_list = [symbol.strip().strip("'") for symbol in selected_symbols.split(',')]
        st.text_area(f"Símbolos del portafolio **{selected_portfolio}**", ', '.join(symbols_list), disabled=True)
        with st.expander("⚙️ Configuraciones Iniciales"):
            
            account = st.text_input(
                    "Cuenta de IB", 
                    value="DU7186453"
                )
            
            cola, colb = st.columns(2)
            
            with cola:
                ip = st.text_input(
                    "Conexión IP con IB", 
                    value="127.0.0.1"
                )
            with colb:   
                port = st.number_input(
                    "Puerto de conexión con IB", 
                    value=7497
                )
            
            st.markdown("---")
            
            # Crear dos columnas
            col1, col2 = st.columns(2)

            # Columna 1: Inputs principales
            with col1:
                capital = st.number_input(
                    "Capital inicial", 
                    value=1_000_000, 
                    step=100_000, 
                    min_value=0, 
                    format="%d"
                )
                ewmacs = st.text_input(
                    "EWMACs (separados por comas)", 
                    value="2,4,8,16,32,64"
                )
                look_back_period = st.number_input(
                    "Look Back Period", 
                    value=36, 
                    step=1, 
                    min_value=0, 
                    format="%d"
                )
                volatility = st.number_input(
                    "Volatilidad esperada", 
                    value=0.25, 
                    step=0.01, 
                    min_value=0.0, 
                    max_value=1.0, 
                    format="%.2f"
                )
                ewmacs_final_weight = st.number_input(
                    "Peso final de EWMACs", 
                    value=0.6, 
                    step=0.1, 
                    min_value=0.0, 
                    max_value=1.0, 
                    format="%.2f"
                )

            # Columna 2: Inputs adicionales
            with col2:
                carry_final_weight = st.number_input(
                    "Carry Final Weight", 
                    value=0.4, 
                    step=0.1, 
                    min_value=0.0, 
                    max_value=1.0, 
                    format="%.2f"
                )
                carry_value = st.number_input(
                    "Valor de carry", 
                    value=1,  
                    disabled=True
                )

                
                num_ewmacs = len([int(x.strip()) for x in ewmacs.split(",")])
                st.number_input(
                    "Número de EWMACs (calculado)", 
                    value=num_ewmacs, 
                    disabled=True, 
                    format="%d"
                )

                annualised_cash_vol_target = capital * volatility
                st.number_input(
                    "Volatilidad anualizada del efectivo (calculada)", 
                    value=annualised_cash_vol_target, 
                    disabled=True, 
                    format="%.2f"
                )

                annual_to_days = np.sqrt(256)
                st.number_input(
                    "Factor anual a diario (calculado)", 
                    value=annual_to_days, 
                    disabled=True, 
                    format="%.2f"
                )

                daily_cash_vol_target = annualised_cash_vol_target / annual_to_days
                st.number_input(
                    "Volatilidad diaria del efectivo (calculada)", 
                    value=daily_cash_vol_target, 
                    disabled=True, 
                    format="%.2f"
                )


        # Valores calculados más complejos a partir de funciones personalizadas
        forecast_scalars = get_forecast_scalars_from_ewma_pairs([int(x.strip()) for x in ewmacs.split(",")])
        ewmacs_correlations = get_ewmac_correlations([int(x.strip()) for x in ewmacs.split(",")])
        ewma_weights = get_ewmac_weights(num_ewmacs, ewmacs_correlations)
        fdm_value = generate_fdm_ewmac_and_carry(
            [int(x.strip()) for x in ewmacs.split(",")],
            ewma_weights,
            ewmacs_final_weight,
            carry_final_weight,
            carry_value
        )
        
        req_symbol_list=list(symbol_info[(symbol_info['broker_symbol'].isin(symbols_list)) & (symbol_info['expiration_actual'] != '0') ]['broker_symbol'])


        idm_value = idm_size_calculator(len(req_symbol_list))

        contracts, multipliers, sectors = generate_dicts(req_symbol_list,
                                                         symbol_info)
        
        ewmacs_list = [int(x) for x in ewmacs.split(',')]
        # Botón para crear la tabla e insertar datos
        if st.button("Realizar Análisis"):
            st.session_state["show_table"] = True  # Mostrar la tabla
            
            if st.session_state["today_generate"]:
                csv_data = load_portfolio_csv(f"{account}_{selected_portfolio}", username)
                merged_df = load_portfolio_csv(f"{account}_{selected_portfolio}", f'{username}_compare', False)
            
                
            if not st.session_state["today_generate"]:
                with st.spinner("Actualizando base de datos... Esto puede tardar unos segundos"):
                    run_update_script(ip, 
                                    port,
                                    req_symbol_list)
                
                with st.spinner("Calculando cantidad de contratos para ajustar posiciones..."):
                    run_update_position_script(ip,
                                    port,
                                contracts,
                                sectors,
                                look_back_period,
                                forecast_scalars,
                                ewma_weights,
                                fdm_value,
                                multipliers,
                                daily_cash_vol_target,
                                f"{selected_portfolio}_{username}",
                                ewmacs_list,
                                ewmacs_final_weight,
                                carry_final_weight,
                                account
                                ) 
                    st.session_state["today_generate"] = True
                    
        if st.session_state["today_generate"]:
            if st.button("Realizar Análisis de nuevo"):
                st.session_state["today_generate"] = False
                st.session_state["update_position"] = False
                st.rerun() 
        else:
            st.warning("El día de hoy no se ha generado el cálculo de las posiciones")       
                    
                
        if st.session_state["show_table"]:
            try:
                if st.session_state["today_generate"]:
                    st.success(f"Posiciones generadas el día de hoy {datetime.now().date().strftime('%Y-%m-%d')}")
                    #if csv_data is None:
                    csv_data = load_portfolio_csv(f"{account}_{selected_portfolio}", username)
                    merged_df = load_portfolio_csv(f"{account}_{selected_portfolio}", f'{username}_compare', False)
                    
                # Calcular el buffer y las operaciones necesarias
                # Group by 'Asset Class' and calculate the sum of 'Weights'
                csv_data.set_index("Metric",inplace=True)
                csv_data_t = csv_data.T
                csv_data_t['Weights'] = csv_data_t['Weights'].astype(float)
                # Agrupación por Asset Class y cálculo de suma de pesos y conteo de instrumentos
                weights_by_asset_class = csv_data_t.groupby('Asset Class').agg(
                    Weights=('Weights', 'sum'),
                    Instruments=('Asset Class', 'count')
                ).reset_index()

                # Renombrar columnas para mayor claridad
                weights_by_asset_class.columns = ['Asset Class', 'Weights', 'Instrument Count']

                # Convertir los pesos a porcentaje con dos decimales
                weights_by_asset_class['Weights'] = weights_by_asset_class['Weights'].apply(lambda x: f"{round(x * 100, 2)}%")
            
                
                merged_df["Contracts to Operate"] = merged_df.apply(calculate_operations, axis=1)
                # Mostrar la tabla en Streamlit
                def style_contracts_to_operate(value):
                    """Estilo para la columna 'Contracts to Operate'."""
                    if value > 0:
                        return 'background-color: lightgreen; color: black;'
                    elif value < 0:
                        return 'background-color: lightcoral; color: black;'
                    return ''
                merged_df["Position"] = merged_df["Position"].astype(int)
                # Crear el DataFrame para mostrar
                edited_df = merged_df[['Instrument', 
                                    'Portfolio Instrument Position',
                                    'Position', 
                                    'Contracts to Operate']][merged_df['Contracts to Operate'] != 0].reset_index(drop=True)

                # Aplicar estilo condicional a la columna
                edited_df = edited_df.style.applymap(style_contracts_to_operate, subset=['Contracts to Operate'])

                
                info_df = merged_df[['Instrument', 
                                    'Portfolio Instrument Position',
                                    'Position', 
                                    'Contracts to Operate']][merged_df['Contracts to Operate'] != 0].reset_index(drop=True)
                
                # Mostrar la tabla en Streamlit
                st.dataframe(edited_df, use_container_width=True)
                
                with st.expander("🔽 Data de posiciones completa"):
                    edited_df_2 = st.data_editor(
                    merged_df[['Instrument', 
                                'Portfolio Instrument Position',
                                'Position','Contracts to Operate', 
                                ]],  # Incluye la nueva columna "index"
                    column_config={
                        #"is_widget": "Approved",
                        "Position": "Actual Position",
                        "Portfolio Instrument Position":"Instrument Position"
                        # Cambia el nombre del índice si lo deseas
                    },
                    disabled=["Instrument", "Portfolio Instrument Position", 
                            "Position", "Contracts to Operate"],
                    hide_index=True,  # Oculta el índice visual original de Streamlit
                    use_container_width=True  # Desactiva el ancho automático
                )
                with st.expander("🔽 Data descargable y Análisis"):
                    st.dataframe(csv_data.T)
                    st.write('Distribución de pesos por tipo de activos')
                    st.dataframe(weights_by_asset_class.sort_values('Weights', ascending=True).reset_index(drop=True))
                    st.markdown(f'Quantity of Transaction **{len(info_df)}**')
                    st.markdown(f'Quantity of Lots **{info_df["Contracts to Operate"].abs().sum()}**')
                    
                
            except Exception as e:
                print(e)
                st.session_state["show_table"] = False
                pass
                #st.warning("Parace que hubo un error al general la tabla, revisa la configuración de conexión con IB")
          
        if st.session_state["today_generate"]:
            if not st.session_state["update_position"]:
                if st.button("Ejecutar ordenes aprobadas"):
                    with st.spinner("Ejecutando ordenes"):
                        st.session_state["show_table"] = False
                        symbol_list = list(merged_df[merged_df['Contracts to Operate'] != 0]["Instrument"])
                        for sym in symbol_list:
                            contracts[sym]["contracts_to_operate"] = merged_df[merged_df["Instrument"] == sym]["Contracts to Operate"].iloc[0]
                        run_buy_sell_position_script(ip, 
                                port, 
                                str(contracts), 
                                str(symbol_list), 
                                account)   
                        st.session_state["update_position"] = True
            else:
                st.warning("Hoy ya han sido actualizadas las posiciones, intenta mañana")    
        
    else:
        st.warning("No tienes ningún portafolio configurado. Haz clic en el enlace a continuación para configurarlo:")
        st.markdown('<a href="/Panel_De_Usuario" target="_self">Ir al Panel de Usuario</a>', unsafe_allow_html=True)
        