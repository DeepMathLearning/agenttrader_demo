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


# CSV file to store process information
PROCESS_FILE = "custom_processes.csv"

# Function to load processes from CSV
def load_processes():
    if os.path.exists(PROCESS_FILE):
        try:
            df = pd.read_csv(PROCESS_FILE).to_dict(orient='records')
            return df
        except:
             return [] 
    else:
        return []

# Function to save processes to CSV
def save_processes(processes):
    df = pd.DataFrame(processes)
    df.to_csv(PROCESS_FILE, index=False)

# Function to run custom commands and store process information
def run_custom_commands():
    commands = [
        "python zenit-CRUCEEMAS-strategy.py --symbol MESM4 --exchange CME --secType FUT --client 2 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 1m --quantity 6 --account DU7186454 --accept_trade long",
        "python zenit-CRUCEEMAS-strategy.py --symbol MESM4 --exchange CME --secType FUT --client 3 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 5m --quantity 6 --account DU7186454 --accept_trade long",
        "python zenit-CRUCEEMAS-strategy.py --symbol MESM4 --exchange CME --secType FUT --client 4 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 15m --quantity 6 --account DU7186454 --accept_trade long",
        "python zenit-CRUCEEMAS-strategy.py --symbol ESM4 --exchange CME --secType FUT --client 5 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 1m --quantity 6 --account DU7186454 --accept_trade short",
        "python zenit-CRUCEEMAS-strategy.py --symbol ESM4 --exchange CME --secType FUT --client 6 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 5m --quantity 6 --account DU7186454 --accept_trade short",
        "python zenit-CRUCEEMAS-strategy.py --symbol ESM4 --exchange CME --secType FUT --client 7 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 15m --quantity 6 --account DU7186454 --accept_trade short",
        "python zenit-TRENDEMASCLOUD-strategy.py --symbol MESM4 --exchange CME --secType FUT --client 8 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 1m --quantity 12 --account DU7186454 --accept_trade long",
        "python zenit-TRENDEMASCLOUD-strategy.py --symbol MESM4 --exchange CME --secType FUT --client 9 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 5m --quantity 12 --account DU7186454 --accept_trade long",
        "python zenit-TRENDEMASCLOUD-strategy.py --symbol MESM4 --exchange CME --secType FUT --client 10 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 15m --quantity 12 --account DU7186454 --accept_trade long",
        "python zenit-TRENDEMASCLOUD-strategy.py --symbol ESM4 --exchange CME --secType FUT --client 11 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 1m --quantity 12 --account DU7186454 --accept_trade short",
        "python zenit-TRENDEMASCLOUD-strategy.py --symbol ESM4 --exchange CME --secType FUT --client 12 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 5m --quantity 12 --account DU7186454 --accept_trade short",
        "python zenit-TRENDEMASCLOUD-strategy.py --symbol ESM4 --exchange CME --secType FUT --client 13 --trading_class ES --multiplier 50 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 15m --quantity 12 --account DU7186454 --accept_trade short",
        "python zenit-strategy-bot.py --port 7497 --accept_trade ab --symbol MESM4 --exchange CME --secType FUT --client 14 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 1m --quantity 120 --account DU7186454",
        "python zenit-strategy-bot.py --port 7497 --accept_trade ab --symbol MESM4 --exchange CME --secType FUT --client 15 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 5m --quantity 120 --account DU7186454",
        "python zenit-strategy-bot.py --port 7497 --accept_trade ab --symbol MESM4 --exchange CME --secType FUT --client 16 --trading_class MES --multiplier 5 --lastTradeDateOrContractMonth 20240621 --is_paper False --interval 15m --quantity 120 --account DU7186454"
    ]

    new_processes = []
    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        new_processes.append({
            "command": cmd,
            "pid": process.pid
        })
    
    processes = load_processes() + new_processes
    save_processes(processes)

# Function to kill a custom process
def kill_custom_process(pid):
    os.killpg(os.getpgid(pid), signal.SIGTERM)

# Function to get system processes information
def get_system_processes_info():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline']):
        try:
            process_info = proc.info
            processes.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

# Function to kill a system process
def kill_system_process(pid):
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=3)  # Wait for the process to terminate
        return f"Process {pid} terminated successfully."
    except psutil.NoSuchProcess:
        return f"Process {pid} does not exist."
    except psutil.AccessDenied:
        return f"Access denied to terminate process {pid}."
    except psutil.TimeoutExpired:
        return f"Process {pid} termination timed out."


# tab1, tab2 = st.tabs(["Custom Trading Strategies", "System Processes"])

# with tab1:
#     st.header('Custom Trading Strategies')

#     if st.button('Run Strategies'):
#         run_custom_commands()
#         st.success('Strategies started!')

#     st.subheader('Running Custom Strategies:')
#     processes = load_processes()
#     if len(processes) > 0:
#         for i, proc_info in enumerate(processes):
#             st.write(f"Command: {proc_info['command']}, PID: {proc_info['pid']}")
#             if st.button(f'Kill Custom Process {i+1}', key=f'kill_custom_{i}'):
#                 kill_custom_process(proc_info['pid'])
#                 st.success(f'Custom Process {i+1} terminated!')
#                 processes.pop(i)
#                 save_processes(processes)
#     else:
#         st.write(f"No existen procesos abiertos")

# with tab2:

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
    st.title('Process Management Application')
    with st.sidebar:
        st.write(f'**Usuario**: *{st.session_state["name"]}*')
        authenticator.logout()
        st.markdown("---")
    
    st.header('System Processes')
    system_processes = get_system_processes_info()
    target_strategies = ["zenit_CRUCEEMAS_strategy", "zenit_TRENDEMASCLOUD_strategy", "zenit_EMASCLOUD_strategies", "zenit_strategy_bot"]
    if system_processes:
        for proc in system_processes:
            if proc['name'] in ['Python', 'python']:
                try:
                    if (True not in ["streamlit" in x for x in proc['cmdline']]):
                        st.write(f"PID: {proc['pid']}, Name: {proc['name']}, User: {proc['username']}, Command: {proc['cmdline']}")
                        if st.button(f'Kill PID {proc["pid"]}', key=f'kill_system_{proc["pid"]}'):
                            message = kill_system_process(proc['pid'])
                            st.success(message)
                            st.experimental_rerun()
                except:
                    st.write("No existen procesos abiertos.")
    else:
        st.write("No existen procesos abiertos.")

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
    
    
    
    
