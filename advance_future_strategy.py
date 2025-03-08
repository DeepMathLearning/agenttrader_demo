import pandas as pd
import yfinance as yf
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ta
from tqdm import tqdm
import time
from time import sleep
import base64
import plotly.offline as pyo
import logging
import argparse
import numpy as np
from datetime import datetime
import math

logger = logging.getLogger()
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO)


class AdvanceFutureStrategy():
    def __init__(
                self
                ):
        
        
        self.period = None
        # Inicializar variables
        self.capital = None  # Capital inicial en dólares
        self.risk_percentage = None
        self.data = None
        self.contract = None
        self.interval = None
        self.symbol_ = None
        self.strategy_number = None
        self.sf = pd.read_csv('data/futures_symbols_v1.csv', sep=",")
        print(self.sf.columns)
        self.sym_info = self.sf.to_dict(orient='records')

    def main(self,
                list_symbol):
       
        # if self.interval is None:
        #     self.data = yf.download(list_symbol, period='max',keepna=False)
        # else:
        #     self.data = yf.download(list_symbol,interval=self.interval, period='max',keepna=False)
        #     if len(self.data) == 0:
        #         self.data = yf.download(list_symbol, interval=self.interval, period='3mo',keepna=False)
        #         if len(self.data) == 0:
        #             self.data = yf.download(list_symbol, interval=self.interval, period='60d',keepna=False)
        
        if self.interval is None:
            self.data = yf.download(list_symbol, period='max',keepna=False)
        elif self.interval in ['5m', '15m', '90m']:
            self.data = yf.download(list_symbol,interval=self.interval, period='60d',keepna=False)
        elif self.interval in ['1h']:
            self.data = yf.download(list_symbol, interval=self.interval, period='730d',keepna=False)
            
        else:
            self.data = yf.download(list_symbol,interval=self.interval, period='max',keepna=False)
            
       
        
        if len(list_symbol) > 1:
            tickers = self.data.columns.levels[1]
            # Iterar sobre los tickers y asignar ceros a las columnas 'n_position' para cada uno
            for ticker in tickers:
                try:
                    self.data['daily_returns', ticker] = self.data['Close', ticker].pct_change()
                    self.data['n_position', ticker] = 0
                    self.data['n_position_variable', ticker] = 0
                    self.data['capital_series', ticker] = 0
                except:
                    pass
        else:
            ticker = list_symbol[0]
            for col in self.data.columns:
                self.data[col, ticker] = self.data[col]
            try:
                self.data['daily_returns', ticker] = self.data['Close', ticker].pct_change()
                self.data['n_position', ticker] = 0
                self.data['n_position_variable', ticker] = 0
                self.data['capital_series', ticker] = 0
            except:
                pass

    
    def ewmac(self, fast_span, slow_span):
        """
        Computes EWMAC
        Inputs: dataframe with closing prices, fast ewm span, and slow ewm span
        Output: dataframe with EWMAC column
        """
        self.data[f"EWMA_{fast_span}"] = self.data["Close"].ewm(span=fast_span, adjust=False).mean()
        self.data[f"EWMA_{slow_span}"] = self.data["Close"].ewm(span=slow_span, adjust=False).mean()
        self.data[f"EWMAC_{fast_span}_{slow_span}"] = (
            self.data[f"EWMA_{fast_span}"] - self.data[f"EWMA_{slow_span}"]
        )
    
    def capped_forecast(self, ewma_series, st_dev_daily):
        raw_forecast = ewma_series / st_dev_daily
        forecast_scalar = 10 / np.nanmean(np.abs(raw_forecast))
        scaled_forecast = raw_forecast * forecast_scalar #See page 180 Carver
        capped_f = np.clip(scaled_forecast, -20, 20) #Equivalent to data['scaled_forecast'].apply(lambda x: max(min(x, 20), -20))
        return capped_f
    
    
    def trading_decision(self, C, B_l, B_u):
        if C < B_l:
            return (B_l-C), 1
        elif C > B_u:
            return -(C-B_u), -1
        else:
            return 0, 0
    
    def calculate_position_size_buffer(self,
        Capped_forecast,  
        IDM, 
        weight, tau, 
        Multiplier, 
        Price, 
        FX_rate, 
        standard_deviation
        ):
        """
        Calculates amount of contracts for a given date
        Inputs: Capped forecast, Capital, IDM, weight of the instrument, predefined risk, futures contract multiplier, 
        foreign exchange rate, volatility of the instrument, ewmac signal
        Output: Contracts for a given date (N)
        """
        N = (Capped_forecast * self.capital * IDM * weight * tau) / (10 * Multiplier * Price * FX_rate * standard_deviation)
        # if signal>0:
        B = (0.1 * self.capital * IDM * weight * tau) / (Multiplier * Price * FX_rate * standard_deviation)
        # else:
        #     B = -(0.1 * Capital * IDM * weight * tau) / (Multiplier * Price * FX_rate * standard_deviation)


        return N, B
    
    def calculate_position_size_with_forecast(self,
            Capped_forecast, 
            IDM, 
            weight, tau, 
            Multiplier, Price, 
            FX_rate, standard_deviation
        ):
        """
        Calculates amount of contracts for a given date
        Inputs: Capped forecast, Capital, IDM, weight of the instrument, predefined risk, futures contract multiplier, 
        foreign exchange rate, volatility of the instrument, ewmac signal
        Output: Contracts for a given date (N)
        """

        return ((Capped_forecast * self.capital * IDM * weight * tau) / 
                (10 * Multiplier * Price * FX_rate * standard_deviation))
    
    
    def calculate_indicators(self, symbol, multiplier, is_with_capital=False):

        close_prices = self.data['Close', symbol].dropna()
        
        # Calcular el Notional Exposure
        notional_exposure = close_prices * multiplier
        
        # Calcular los rendimientos diarios
        daily_returns = self.data['daily_returns', symbol]
        
        # Calcular la desviación estándar anualizada
        annualized_sd = np.std(daily_returns) * np.sqrt(252)
        
        # Calcular el rendimiento diario promedio
        average_daily_return = daily_returns.mean()
        
        # Calcular la desviación estándar diaria promedio
        average_daily_sd = daily_returns.std()

        # Calcula la skewness mensual
        skew = np.mean(daily_returns) / np.std(daily_returns)
        
        # Calcular el Sharp Ratio
        sharp_ratio = (average_daily_return / average_daily_sd) * np.sqrt(252)

        # Calcula la cola inferior
        lower = np.percentile(daily_returns.dropna(), 5)
        
        # Calcula la cola superior
        upper = np.percentile(daily_returns.dropna(), 95)
        if not is_with_capital:
            cap_min = (self.contract * multiplier * self.data['Close', symbol][-1] * annualized_sd) / self.risk_percentage
        else:
            self.contract = (self.capital * self.risk_percentage) / (multiplier * self.data['Close', symbol][-1] * annualized_sd)
            cap_min = self.capital
        
        ewsd = self.data['Close', symbol].ewm(span=32, min_periods=32).std()
        
        # Calcular el número de días en la data
        N = len(self.data['Close', symbol].dropna())

        # Calcular el factor de suavizado exponencial (lambda)
        lmbda = 2 / (N + 1)

        # Calcular los pesos exponenciales
        weights = np.array([(1 - lmbda) ** i for i in range(N)])

        # Normalizar los pesos
        weights /= weights.sum()

        # Calcular la media exponencialmente ponderada
        ew_mean = np.average(self.data['Close', symbol].dropna(), weights=weights)

        # Calcular la varianza exponencialmente ponderada
        ew_variance = np.average((self.data['Close', symbol].dropna() - ew_mean) ** 2, weights=weights)

        # Calcular la desviación estándar exponencialmente ponderada
        ew_std = np.sqrt(ew_variance)

        year_data = len(pd.to_datetime(self.data['Close', symbol].dropna().index).year.unique())
        # Calcular la "Fat Tail Risk" (Riesgo de cola gorda)
        #fat_tail_risk = np.percentile(daily_returns.dropna(), 5)
        
        
        # Calcular el rendimiento anual promedio (Mean annual return)
        mean_annual_return = self.data['daily_returns', symbol].mean() * 252

        # Calcular el drawdown
        self.data['Cumulative Return', symbol] = (1 + self.data['daily_returns', symbol]).cumprod()
        self.data['Cumulative High', symbol] = self.data['Cumulative Return', symbol].cummax()
        self.data['Drawdown', symbol] = self.data['Cumulative Return', symbol] / self.data['Cumulative High', symbol] - 1
        
        # Calcular el drawdown promedio (Average drawdown)
        average_drawdown = self.data['Drawdown', symbol].mean()
        
        metrics = {
            "years_of_data" : year_data,
            "last_price": self.data['Close', symbol][-1],
            "notional_exposure_per_contract": round(notional_exposure[-1], 2),
            f"minimun_capital_per_contract": round(cap_min/self.contract, 2),
            "mean_annual_return": round(mean_annual_return,5),
            "average_drawdown": round(average_drawdown,5),
            f"notional_exposure_per_{round(self.contract)}_contracts": round(notional_exposure[-1] * round(self.contract+1), 2),
            "annualized_standard_deviation": round(annualized_sd, 3),
            "average_daily_returns": round(average_daily_return, 3),
            "average_daily_standard_deviation": round(average_daily_sd, 3),
            "sharpe_ratio": round(sharp_ratio, 4),
            "monthly_skew": round(skew, 4),
            "lower_tail": round(lower, 4),
            "upper_tail": round(upper, 4),
            f"minimun_capital_for_{round(self.contract)}_contracts": round(cap_min, 2),
            "annualized_standard_deviation_usd": round(annualized_sd*notional_exposure[-1], 3)
            #"Fat Tail Risk": round(fat_tail_risk, 4)
        }
        return metrics

    def format_number(self, number):
        return '{:,.0f}'.format(number)

    def plot_daily_returns(self, symbol_data, symbol):
                
        daily_returns = self.data['daily_returns', symbol_data]      
        # Crear la figura de Plotly
        fig = go.Figure()
        
        # Agregar la serie de datos de los retornos diarios
        fig.add_trace(go.Scatter(x=daily_returns.dropna().index, 
                                 y=daily_returns.dropna().values, 
                                 mode='lines', 
                                 line=dict(color='#8E895A')))
        
        # Configurar título y etiquetas
        fig.update_layout(title='Daily Returns of {}'.format(symbol), xaxis_title='Date', yaxis_title='Daily Returns')
        
        # Configurar el fondo del gráfico y del área de trazado como transparentes
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))
        
        
        # Mostrar la gráfica
        return fig
    
    def plot_cumulative_returns(self, symbol_data, symbol):
        
        # Calcular los retornos porcentuales
        daily_returns = self.data['daily_returns', symbol_data]      
        
        # Calcular la suma acumulada de los retornos porcentuales
        cumulative_returns = daily_returns.cumsum()
        
        # Crear la figura de Plotly
        fig = go.Figure()
        
        # Agregar la serie de datos de la suma acumulada de los retornos porcentuales
        fig.add_trace(go.Scatter(x=cumulative_returns.dropna().index, 
                                 y=cumulative_returns.dropna().values, 
                                 mode='lines', 
                                 line=dict(color='#8E895A')))
        
        # Configurar título y etiquetas
        fig.update_layout(title='Cumulative Returns of {}'.format(symbol), xaxis_title='Date', yaxis_title='Cumulative Returns')
        
        # Configurar el fondo del gráfico y del área de trazado como transparentes
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))
        
        # Mostrar la gráfica
        return fig
    
    def fig_buy_and_hold_with_fixed_risk_target(self, symbol_data, symbol, name):
        # Crear el gráfico
        fig = go.Figure()

        # Agregar la serie de tiempo del capital
        fig.add_trace(go.Scatter(x=self.data['capital_series', 
                                             symbol_data].dropna().index, 
                                             y=self.data['capital_series', symbol_data].dropna().values, 
                                             mode='lines', 
                                             name='Capital'))

        # Estilizar el gráfico
        fig.update_layout(title=f'Account curve for {name} ({symbol}), buy and hold with fixed risk target',
                        xaxis_title='Date',
                        yaxis_title='Capital Value ($)',
                        )
        
        # Configurar el fondo del gráfico y del área de trazado como transparentes
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))

        # Mostrar el gráfico
        return fig
    
    def fig_position_in_the_time(self, symbol_data, symbol, name):
        # Crear el gráfico
        fig = go.Figure()

        # Agregar la serie de tiempo del capital
        fig.add_trace(go.Scatter(x=self.data['n_position', 
                                             symbol_data].dropna().index, 
                                             y=self.data['n_position', symbol_data].dropna().values, 
                                             mode='lines',line=dict(color='red'), 
                                             name='Number of positions')
                                             )

        # Estilizar el gráfico
        fig.update_layout(title=f'Position in contracts over time given risk scaling of {name} ({symbol})',
                        xaxis_title='Date',
                        yaxis_title='Number of Contracts',
                        )
        
        # Configurar el fondo del gráfico y del área de trazado como transparentes
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))

        # Mostrar el gráfico
        return fig
    
    def calcular_ema(self, symbol_data, window):
        return self.data['Close', symbol_data].ewm(span=window, adjust=False).mean()
    
    def interval_to_numeric(self, interval):
        start, end = interval.split(" to ")
        return float(start), float(end)

    
    def calculate_metrics(self, data, symbol_data, cap_ini, cap_final, ret_column, multiplier, trades):
        # Calcula las métricas deseadas

        # Rendimientos diarios
        returns = data[ret_column, symbol_data].dropna()

        # Sharpe Ratio
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # 252 días de trading al año

        # Max Drawdown
        cumulative_returns = (1 + returns).cumprod()
        #print(f'PRINT {returns}')
        

        # Min Drawdown
        min_drawdown = ((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).min()

        # Rendimiento total
        total_return = cumulative_returns.iloc[-1] - 1

        # Métricas de operaciones
        num_trades = len(trades)
        wins = trades[(trades['PnL'] > 0)]
        losses = trades[(trades['PnL'] <= 0)]
        
        win_change = trades[(trades['PnL'] > 0) & (trades['type'] == 'change_in_position')]
        loss_change = trades[(trades['PnL'] <= 0) & (trades['type'] == 'change_in_position')]
        
        max_drawdown = (losses['PnL'].min() / cap_ini) * 100
        #((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max()
        
        win_streaks = wins.groupby((wins['PnL'] <= 0).cumsum()).size()
        loss_streaks = losses.groupby((losses['PnL'] > 0).cumsum()).size()

        avg_win_streak = win_streaks.mean() if not win_streaks.empty else 0
        max_win_streak = win_streaks.max() if not win_streaks.empty else 0

        avg_loss_streak = loss_streaks.mean() if not loss_streaks.empty else 0
        max_loss_streak = loss_streaks.max() if not loss_streaks.empty else 0

        avg_win = wins['PnL'].mean() if not wins.empty else 0
        avg_loss = losses['PnL'].mean() if not losses.empty else 0

        avg_return = trades['PnL'].mean()
        reward_to_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        avg_length = trades['Length'].mean() if num_trades > 0 else 0

        # Trades per day/month
        trades_per_day = num_trades / data['Close', symbol_data].index.nunique()
        trades_per_month = num_trades / (data['Close', symbol_data].index.nunique() / 21)

        return {
            #"Net Performance": f"{round(total_return*100,2)}%",
            "Final Capital": round(cap_final,2),
            "Initial Capital": cap_ini,
            "Positions": num_trades,
            "Wins": len(wins),
            "Win Streak, avg": avg_win_streak,
            "Win Streak, max": max_win_streak,
            "Losses": len(losses),
            "Loss Streak, avg": avg_loss_streak,
            "Loss Streak, max": max_loss_streak,
            "Win in change position": len(win_change),
            "Loss in change position": len(loss_change),
            "Max DD (Max Drawdown)": f'{round(max_drawdown,2)}%',
            "Max PnL Win": f"{self.format_number(trades['PnL'].max())}",
            "Max PnL Loss": f"{self.format_number(trades['PnL'].min())}",
            "Average Win": self.format_number(round(avg_win,2)),
            "Average Loss": self.format_number(round(avg_loss, 2)),
            "Average Return": self.format_number(round(avg_return)),
            "Rew/Risk Ratio (Reward-to-Risk Ratio)": round(reward_to_risk_ratio, 2),
            "Avg. Length (Average Length)": round(avg_length, 2),
            "Trades/Day": round(trades_per_day, 4),
            "Trades/Month": round(trades_per_month, 4),
            "Sharpe Ratio": round(sharpe_ratio, 2)
        }
    
    def strategy_futures_carver(self, 
                                symbol_data, 
                                emas_list, 
                                trade_type, 
                                multiplier,
                                is_with_capital = True,
                                is_carver = True):
        # Calcular los rendimientos diarios
        daily_returns = self.data['daily_returns', symbol_data]
        if self.interval in ['1m', '2m', '5m', '15m', '30m', '90m']:
            annualized_sd = np.std(daily_returns) * np.sqrt(252)
        else:
            annualized_sd = np.std(daily_returns) * np.sqrt(252)
            
        print(f'ANNUALIZED SD {annualized_sd}')
        cont_buy = 0
        cont_sell = 0
        out_position = False
        self.data['Open_position', symbol_data] = 0
        self.data['Close_real_price', symbol_data] = 0
        self.data['Short_Exit', symbol_data] = 0
        open_position = False
        for j in emas_list:
            self.data[f'EMA{j}', symbol_data] = self.calcular_ema(symbol_data, j)
            self.data[f'EMA{j}', symbol_data] = self.calcular_ema(symbol_data, j)
            
        self.data['MACD', symbol_data] = self.data[f'EMA{emas_list[0]}', symbol_data] - self.data[f'EMA{emas_list[1]}', symbol_data]
        mid_point = emas_list[1] #(emas_list[0]+emas_list[1])/2
        self.data['Signal', symbol_data] = self.data['MACD', symbol_data].ewm(span=mid_point, adjust=False).mean()

        
        
        self.data['Long_Signal', symbol_data] = np.where(
                                                        (self.data[f'EMA{emas_list[0]}', symbol_data] > self.data[f'EMA{emas_list[1]}', symbol_data])&
                                                        (self.data['MACD', symbol_data] > self.data['Signal', symbol_data])
                                                        ,1,0
                                                    )
        self.data['Short_Signal', symbol_data] = np.where(
                                                        (self.data[f'EMA{emas_list[1]}', symbol_data] > self.data[f'EMA{emas_list[0]}', symbol_data])&
                                                        (self.data['MACD', symbol_data] < self.data['Signal', symbol_data])
                                                        ,1,0
                                                    )
        # Calcular el factor de escala
        scale_factor = 16

        # Calcular el Raw Forecast en función del riesgo anualizado
        Raw_Forecast = (self.data[f'EMA{emas_list[0]}', symbol_data][-1] - self.data[f'EMA{emas_list[1]}', symbol_data][-1]) / (self.data['Close', symbol_data][-1] * annualized_sd / scale_factor)
        
        Scaled_Forecast = Raw_Forecast * 4.1
        # Calcula el Capped Forecast
        Capped_Forecast = round(np.maximum(np.minimum(Scaled_Forecast, 20), -20), 2)
       

        if is_carver:
            if not is_with_capital:
                cap_ini = (self.contract * multiplier * self.data['Close', symbol_data][-1] * self.risk_percentage) / annualized_sd
            else:
                self.contract = round((Capped_Forecast * self.capital * self.risk_percentage) / (10 * multiplier * self.data['Close', symbol_data].dropna().iloc[-1] * annualized_sd) + 1)
                buffer = round((0.1 * self.capital * self.risk_percentage) / (multiplier * self.data['Close', symbol_data][-1] * annualized_sd)+1)
                if (self.contract < 0):
                    self.contract = (-1) * self.contract 
                lower_buffer = round(self.contract - buffer)
                upper_buffer = round(self.contract + buffer)
                cap_ini = self.capital
        else:
            cap_ini = self.capital
        
        print(f'CAPPED FORECAST {Capped_Forecast}')
        print(f'CONTRATOS  ----- {self.contract}')
        print(f'IS {is_with_capital}')

        cap_final = cap_ini
        
        # Inicialización
        trades = []

        # Simulación del bucle principal del backtest
        for index, row in self.data.iterrows():
            current_price = row['Close', symbol_data]
            if trade_type == 'long':
                if ((not open_position) and 
                    (row['Long_Signal', symbol_data] == 1) and
                    (not out_position)
                    ):
                    open_trade_price = current_price
                    self.data['Open_position', symbol_data].loc[index] = 1
                    self.data['Close_real_price', symbol_data].loc[index] = current_price
                    open_position = True
                    cont_buy += self.contract
                    cap_final = cap_final - (open_trade_price * self.contract * multiplier)
                    entry_index = index
                    
                elif ((open_position) and 
                    (row['MACD', symbol_data] <= row['Signal', symbol_data])
                    ):
                    self.data['Open_position', symbol_data].loc[index] = -1
                    self.data['Close_real_price', symbol_data].loc[index] = current_price
                    open_position = False
                    cont_sell += self.contract
                    cap_final = cap_final + (current_price * self.contract * multiplier)
                    out_position = True
                    
                    # Registrar la operación
                    trade_length = (index - entry_index).days
                    trades.append({'PnL': (current_price - open_trade_price) * self.contract * multiplier, 'Length': trade_length})
                    
                if ((out_position) and 
                    (row[f'EMA{emas_list[1]}', symbol_data] > row[f'EMA{emas_list[0]}', symbol_data])):
                    out_position = False
                
            elif trade_type == 'short':
                if (not open_position) and (row['Short_Signal', symbol_data]):
                    open_trade_price = current_price
                    self.data['Open_position', symbol_data].loc[index] = -1
                    self.data['Close_real_price', symbol_data].loc[index] = current_price
                    open_position = True
                    cont_sell += self.contract
                    cap_final = cap_final + (open_trade_price * self.contract * multiplier)
                    entry_index = index
                    
                elif ((open_position) and 
                    (row['MACD', symbol_data] >= row['Signal', symbol_data])
                    ):
                    self.data['Open_position', symbol_data].loc[index] = 1
                    self.data['Close_real_price', symbol_data].loc[index] = current_price
                    open_position = False
                    cont_buy += self.contract
                    cap_final = cap_final - (current_price * self.contract * multiplier)
                    out_position = True
                    
                    # Registrar la operación
                    trade_length = (index - entry_index).days
                    trades.append({'PnL': (open_trade_price - current_price) * self.contract * multiplier, 'Length': trade_length})
                    
                if ((out_position) and 
                    (row[f'EMA{emas_list[1]}', symbol_data] < row[f'EMA{emas_list[0]}', symbol_data])):
                    out_position = False

        
        # Añadir retornos y estrategias a los datos
        self.data['retornos', symbol_data] = self.data['Close', symbol_data].pct_change()
        self.data['posicion', symbol_data] = np.where(self.data[f'EMA{emas_list[0]}', symbol_data] > self.data[f'EMA{emas_list[1]}', symbol_data], 1, -1)
        self.data['estrategia', symbol_data] = self.data['posicion', symbol_data].shift(1) * self.data['retornos', symbol_data]
        #cap_final = cap_ini * (1 + self.data['estrategia', symbol_data].dropna()).cumprod().iloc[-1]

        # Convertir trades a DataFrame
        trades_df = pd.DataFrame(trades)
        cap_final = trades_df['PnL'].sum() + cap_ini
        # Calcular métricas

        # Agregar otras métricas al diccionario
        metrics = {
            "PNL": f'{self.format_number(round(cap_final - cap_ini,2))} ({round(((cap_final - cap_ini)/cap_ini)*100, 2)}%)',
            "Buy contracts": cont_buy,
            "Sell contracts": cont_sell
        }
        
        if is_with_capital and is_carver:
            metrics["Contracts per position"] = self.contract

        metrics.update(self.calculate_metrics(self.data, symbol_data, cap_ini, cap_final, 'estrategia', multiplier, trades_df))
        

        return metrics, trades_df

    
    def idm_size_calculator(self, idm_size):
        idm_mapping = {
            1: 1,
            2: 1.20,
            3: 1.48,
            4: 1.56,
            5: 1.70,
            6: 1.90,
            7: 2.10
        }
        
        # Asignar 2.20 para idm_size entre 8 y 14
        if idm_size in range(8, 15):
            return 2.20
        elif idm_size in range(15, 25):
            return 2.20
        elif idm_size in range(25, 30):
            return 2.40
        elif idm_size >= 30:
            return 2.50
        
        
        # Obtener el valor de IDM del diccionario o devolver None si no se encuentra
        return idm_mapping.get(idm_size, None)
            
            
    
    def strategies_futures_carver(self, 
                                symbol_data, 
                                emas_list, 
                                trade_type, 
                                multiplier,
                                weight,
                                strategy_number,
                                idm_size=1):
        
        # Calcular los rendimientos diarios
        daily_returns = self.data['daily_returns', symbol_data]
        self.strategy_number = strategy_number
        
        IDM = self.idm_size_calculator(idm_size)  # Check table 16 Carver page 135 
        tau = self.risk_percentage  # Risk
        FX_rate = 1  # FX: Exchange rate (if applicable)
        
        annualized_sd = np.std(daily_returns) * np.sqrt(252)
            
        
        metrics = self.calculate_indicators(symbol_data,multiplier, is_with_capital=True)
        initial_cap = self.capital * weight
        cap_final = initial_cap
        cont_buy = 0
        cont_sell = 0
        out_position = False
        cant_position = 0
        self.data['Open_position', symbol_data] = 0
        self.data['Close_real_price', symbol_data] = 0
        self.data['Short_Exit', symbol_data] = 0
        open_position = False
        fast_span = emas_list[0]
        slow_span = emas_list[1]
        
        self.data[f'EMA{fast_span}', symbol_data] = self.calcular_ema(symbol_data, fast_span)
        self.data[f'EMA{slow_span}', symbol_data] = self.calcular_ema(symbol_data, slow_span)
            
        self.data[f'EWMAC_{fast_span}_{slow_span}', symbol_data] = (self.data[f'EMA{fast_span}', symbol_data] - 
                                                                    self.data[f'EMA{slow_span}', symbol_data])
            
        self.data['MACD', symbol_data] = self.data[f'EMA{emas_list[0]}', symbol_data] - self.data[f'EMA{emas_list[1]}', symbol_data]
        mid_point = emas_list[1] #(emas_list[0]+emas_list[1])/2
        self.data['Signal', symbol_data] = self.data['MACD', symbol_data].ewm(span=mid_point, adjust=False).mean()

        
        
        self.data['Long_Signal', symbol_data] = np.where(
                                                        (self.data[f'EMA{fast_span}', symbol_data] > self.data[f'EMA{slow_span}', symbol_data])
                                                        #&
                                                        #(self.data['MACD', symbol_data] > self.data['Signal', symbol_data])
                                                        ,1,0
                                                    )
        self.data['Short_Signal', symbol_data] = np.where(
                                                        (self.data[f'EMA{slow_span}', symbol_data] > self.data[f'EMA{fast_span}', symbol_data])
                                                        #&
                                                        #(self.data['MACD', symbol_data] < self.data['Signal', symbol_data])
                                                        ,1,0
                                                    )
        
        self.data['returns', symbol_data] = self.data['Close', symbol_data].pct_change()
            
           
        ew_st_dev = self.data['returns', symbol_data].ewm(span=32, adjust=False).std()
        self.data['standard_deviation', symbol_data] = 16*(0.3*ew_st_dev.rolling(window='2560D').mean()   #.ewm(span=10*256).mean() 
                                            + 0.7*ew_st_dev)  # See appendix B Carver
        
    
        self.data['St_dev_daily_price_units', symbol_data] = self.data['Close', symbol_data]*self.data['standard_deviation', symbol_data]/16
        self.data['capped_forecast', symbol_data] = self.capped_forecast(self.data[f"EWMAC_{fast_span}_{slow_span}", symbol_data].to_numpy(), self.data['St_dev_daily_price_units', symbol_data].to_numpy())
        
        if strategy_number == 5:
            # Apply the function row-wise to the DataFrame
            self.data['position_size', symbol_data] = self.data[[("Close",symbol_data), 
                                                                ("standard_deviation", symbol_data),
                                                                (f'EWMAC_{fast_span}_{slow_span}', symbol_data),
                                                                ("Long_Signal", symbol_data)]].apply(lambda row: self.calculate_position_size(
                                                                    IDM, weight, tau, multiplier, 
                                                                    row['Close', symbol_data], 
                                                                    FX_rate, row['standard_deviation', symbol_data],
                                                                    row[f'EWMAC_{fast_span}_{slow_span}', symbol_data]) 
                                                                    if row['Long_Signal', symbol_data] == 1 else 0, 
                                                                axis=1)
            
        elif strategy_number == 6:
            # Apply the function row-wise to the DataFrame
            self.data['position_size', symbol_data] = self.data[[("Close",symbol_data), 
                                                                ("standard_deviation", symbol_data),
                                                                (f'EWMAC_{fast_span}_{slow_span}', symbol_data)]].apply(lambda row: self.calculate_position_size(
                                                                    IDM, weight, tau, multiplier, 
                                                                    row['Close', symbol_data], 
                                                                    FX_rate, row['standard_deviation', symbol_data],
                                                                    row[f'EWMAC_{fast_span}_{slow_span}', symbol_data]), 
                                                                    #if row['Long_Signal', symbol_data] == 1 else 0, 
                                                                axis=1)
        
            # Optimización del cálculo de 'position_size' basado en la señal Long/Short
            self.data['position_size', symbol_data] = self.data.apply(
                lambda row: row['position_size', symbol_data] 
                            if row['Long_Signal', symbol_data] == 1 
                            else -row['position_size', symbol_data], 
                    axis=1
                )

        
        
        # calculate_position_size_with_forecast(
        #     Capped_forecast, 
        #     IDM, 
        #     weight, tau, 
        #     Multiplier, Price, 
        #     FX_rate, standard_deviation
        # )
        elif strategy_number == 7:
            # Apply the function row-wise to the DataFrame
            self.data['position_size', symbol_data] = self.data[[("Close",symbol_data), 
                                                                ("standard_deviation", symbol_data),
                                                                ("capped_forecast", symbol_data)]].apply(lambda row: self.calculate_position_size_with_forecast(
                                                                    row['capped_forecast', symbol_data],
                                                                    IDM, weight, tau, multiplier, 
                                                                    row['Close', symbol_data], 
                                                                    FX_rate, row['standard_deviation', symbol_data]), 
                                                                axis=1)
            for _, row in self.data.iterrows(): 
                if row['Long_Signal', symbol_data] == 0: 
                    
                    row['position_size', symbol_data] = -row['position_size', symbol_data]
        elif strategy_number == 8:
            position_size_buffer = self.data.apply(
                lambda row: self.calculate_position_size_buffer(
                    row['capped_forecast', symbol_data], IDM, weight, tau, multiplier, 
                    row['Close', symbol_data], FX_rate, 
                    row['standard_deviation', symbol_data]
                ), axis=1, result_type='expand'
            )
            self.data[[('position_size', symbol_data), ('Buffer', symbol_data)]] = position_size_buffer

            self.data['Buffer_lower', symbol_data] = (self.data['position_size', symbol_data] - self.data['Buffer', symbol_data]).round(0)
            self.data['Buffer_upper', symbol_data] = (self.data['position_size', symbol_data] + self.data['Buffer', symbol_data]).round(0)

                
        self.data['position_size', symbol_data] = self.data['position_size', symbol_data].fillna(0).round(0)
        
        #if metrics['notional_exposure_per_contract'] < self.capital:
        
        if strategy_number == 5 : 
            
            # Inicialización
            trades = []

            # Simulación del bucle principal del backtest
            for index, row in self.data.iterrows():
                current_price = row['Close', symbol_data]
                standard_deviation = row['standard_deviation', symbol_data]
                
                N = row['position_size', symbol_data]
                
                if trade_type == 'long':
                    if ((N > 0) and (not open_position) and 
                        (row['Long_Signal', symbol_data] == 1) 
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = 1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        cont_buy += N #self.contract
                        cant_position += N 
                        cap_final = cap_final - (open_trade_price * N * multiplier)
                        entry_index = index
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((N > 0) and (N > cant_position) 
                            and (open_position) and 
                        (row['Long_Signal', symbol_data] == 1) 
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = 1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = N - cant_position
                        cont_buy += diferential #self.contract
                        cant_position += diferential 
                        cap_final = cap_final - (open_trade_price * diferential * multiplier)
                        entry_index = index
                        #trades.append({'PnL': (current_price - open_trade_price) * diferential * multiplier, 'Length': trade_length})
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((N > 0) and (N < cant_position) 
                            and (open_position) and 
                        (row['Long_Signal', symbol_data] == 1) 
                        ):
                        #open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = -1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = cant_position - N
                        cont_sell += diferential #self.contract
                        cant_position -= diferential 
                        cap_final = cap_final + (current_price * diferential * multiplier)
                        trade_length = (index - entry_index).days
                        trades.append({'PnL': (current_price - open_trade_price) * diferential * multiplier, 
                                       'Length': trade_length,
                                       'type': 'change_in_position',
                                       'open_price': open_trade_price,
                                       'close_price': current_price,
                                       'date': index})                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                        
                    if ((open_position) and 
                        (row['Long_Signal', symbol_data] == 0)
                        ):
                        self.data['Open_position', symbol_data].loc[index] = -1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = False
                        cont_sell += cant_position
                        cap_final = cap_final + (current_price * cant_position * multiplier)
                        
                        # Registrar la operación
                        trade_length = (index - entry_index).days
                        trades.append({'PnL': (current_price - open_trade_price) * cant_position * multiplier, 
                                       'Length': trade_length,
                                       'type': 'close_position',
                                       'open_price': open_trade_price,
                                       'close_price': current_price,
                                       'date': index})
                        cant_position = 0
                        
        if strategy_number in [6,7]: 
            
            # Inicialización
            trades = []

            # Simulación del bucle principal del backtest
            for index, row in self.data.iterrows():
                current_price = row['Close', symbol_data]
                standard_deviation = row['standard_deviation', symbol_data]
                
                N = row['position_size', symbol_data]
                
                if row['Long_Signal', symbol_data] == 1:
                    trade_type = 'long'
                else:
                    trade_type = 'short'
                
                if trade_type == 'long':
                        
                    if ((N > 0) and (not open_position)
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = 1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        cont_buy += N #self.contract
                        cant_position += N 
                        cap_final = cap_final - (open_trade_price * N * multiplier)
                        entry_index = index
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((N > 0) and (N > cant_position) 
                            and (open_position) 
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = 1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = N - cant_position
                        cont_buy += diferential #self.contract
                        cant_position += diferential 
                        cap_final = cap_final - (open_trade_price * diferential * multiplier)
                        entry_index = index
                        #trades.append({'PnL': (current_price - open_trade_price) * diferential * multiplier, 'Length': trade_length})
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((N > 0) and (N < cant_position) 
                            and (open_position)  
                        ):
                        #open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = -1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = cant_position - N
                        cont_sell += diferential #self.contract
                        cant_position -= diferential 
                        cap_final = cap_final + (current_price * diferential * multiplier)
                        trade_length = (index - entry_index).days
                        trades.append({'PnL': (current_price - open_trade_price) * diferential * multiplier, 
                                       'Length': trade_length,
                                       'type': 'change_in_position',
                                       'open_price': open_trade_price,
                                       'close_price': current_price,
                                       'date': index})                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                        
                
                
                if trade_type == 'short':
                    
                    if ((-N > 0) and (not open_position)
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = -1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        cont_sell -= N #self.contract
                        cant_position -= N 
                        cap_final = cap_final - (open_trade_price * (-N) * multiplier)
                        entry_index = index
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((-N > 0) and (N < cant_position) 
                            and (open_position) 
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = -1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = cant_position - N
                        cont_sell -= diferential #self.contract
                        cant_position -= diferential 
                        cap_final = cap_final + (open_trade_price * (diferential) * multiplier)
                        entry_index = index
                        #trades.append({'PnL': (current_price - open_trade_price) * diferential * multiplier, 'Length': trade_length})
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((-N > 0) and (N > cant_position) 
                            and (open_position)  
                        ):
                        #open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = 1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = N + cant_position
                        cont_buy -= diferential #self.contract
                        cant_position -= diferential 
                        cap_final = cap_final + (current_price * (diferential) * multiplier)
                        trade_length = (index - entry_index).days
                        trades.append({'PnL': (open_trade_price - current_price) * diferential * multiplier, 
                                       'Length': trade_length,
                                       'type': 'change_in_position',
                                       'open_price': open_trade_price,
                                       'close_price': current_price,
                                       'date': index})                        #print(f"POSICION ACTUALIZADA: {cant_position}")
            
        elif strategy_number == 8:
            trades = []
            self.data['current_position', symbol_data] = self.data['position_size', symbol_data].shift(1)
            trading_decision_results = self.data.apply(
                lambda row: self.trading_decision(row['current_position', symbol_data], 
                                                  row['Buffer_lower', symbol_data], 
                                                  row['Buffer_upper', symbol_data]),
                axis=1, result_type='expand'
            )
            self.data[[('Amount_traded', symbol_data), ('Signal', symbol_data)]] = trading_decision_results

            self.data['Current_plus_Amount_traded', symbol_data] = (self.data['current_position', symbol_data] + 
                                                                    self.data['Amount_traded', symbol_data])

            # Compute Buffered_Position with vectorized approach
            self.data['Buffered_Position', symbol_data] = self.data['Current_plus_Amount_traded', symbol_data].where(self.data['Amount_traded', symbol_data] != 0)
            self.data['Buffered_Position', symbol_data].ffill(inplace=True)  # Forward fill NaN values
            # Antes del bucle, creas una columna con los valores desplazados
            self.data['position_size_shifted', symbol_data] = self.data['position_size', symbol_data].shift(1)

            # Simulación del bucle principal del backtest
            for index, row in self.data.iterrows():
                current_price = row['Close', symbol_data]
                
                N = row['position_size', symbol_data]
                
                if row['Signal', symbol_data] == 1:
                    trade_type = 'long'
                elif row['Signal', symbol_data] == -1:
                    trade_type = 'short'
                
                if row['Signal', symbol_data] == 1:
                        
                    if ((not open_position)
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = 1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        cont_buy += N #self.contract
                        cant_position += N 
                        cap_final = cap_final - (open_trade_price * N * multiplier)
                        entry_index = index
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((N > cant_position) 
                            and (open_position) 
                        ):
                        #open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = 1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = N - cant_position
                        cont_buy += abs(diferential) #self.contract
                        cant_position += abs(diferential) 
                        cap_final = cap_final + (open_trade_price * abs(diferential) * multiplier)
                        trade_length = (index - entry_index).days
                        trades.append({'PnL': (current_price - open_trade_price) * abs(diferential) * multiplier, 
                                       'Length': trade_length,
                                       'type': 'change_in_position',
                                       'open_price': open_trade_price,
                                       'close_price': current_price,
                                       'date': index})
                        #trades.append({'PnL': (current_price - open_trade_price) * diferential * multiplier, 'Length': trade_length})
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                        
                if row['Signal', symbol_data] == -1:
                    
                    if ((not open_position)
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = -1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        cont_sell -= N #self.contract
                        cant_position -= N 
                        cap_final = cap_final - (open_trade_price * N * multiplier)
                        entry_index = index
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
                    elif ((N < cant_position) 
                            and (open_position) 
                        ):
                        open_trade_price = current_price
                        self.data['Open_position', symbol_data].loc[index] = -1
                        self.data['Close_real_price', symbol_data].loc[index] = current_price
                        open_position = True
                        diferential = cant_position - N
                        cont_sell -= abs(diferential) #self.contract
                        cant_position -= abs(diferential) 
                        cap_final = cap_final + (current_price * abs(diferential) * multiplier)
                        trade_length = (index - entry_index).days
                        trades.append({'PnL': (open_trade_price - current_price) * abs(diferential) * multiplier, 
                                       'Length': trade_length,
                                       'type': 'change_in_position',
                                       'open_price': open_trade_price,
                                       'close_price': current_price,
                                       'date': index})  
                        #trades.append({'PnL': (current_price - open_trade_price) * diferential * multiplier, 'Length': trade_length})
                        #print(f"POSICION ACTUALIZADA: {cant_position}")
                    
        # Añadir retornos y estrategias a los datos
        self.data['retornos', symbol_data] = self.data['Close', symbol_data].pct_change()
        self.data['posicion', symbol_data] = np.where(self.data[f'EMA{emas_list[0]}', symbol_data] > self.data[f'EMA{emas_list[1]}', symbol_data], 1, -1)
        self.data['estrategia', symbol_data] = self.data['posicion', symbol_data].shift(1) * self.data['retornos', symbol_data]
        #cap_final = cap_ini * (1 + self.data['estrategia', symbol_data].dropna()).cumprod().iloc[-1]

        # Convertir trades a DataFrame
        trades_df = pd.DataFrame(trades)
        cap_final = trades_df['PnL'].sum()
        # Calcular métricas

        # Agregar otras métricas al diccionario
        metrics = {
            "PnL_usd": round(cap_final ,2),
            "PNL": f'{self.format_number(round(cap_final - initial_cap,2))} ({round(((cap_final - initial_cap)/initial_cap)*100, 2)}%)',
            "Buy contracts": cont_buy,
            "Sell contracts": cont_sell
        }
        
       
        metrics.update(self.calculate_metrics(self.data, symbol_data, initial_cap, cap_final, 'estrategia', multiplier, trades_df))
        

        return metrics, trades_df


            
    def calculate_position_size(self, IDM, weight, tau, multiplier, Price, FX_rate, standard_deviation, signal):
        '''
        Calculates amount of contracts for a given date
        Inputs: Capital, IDM, weight of the instrument, predefined risk, futures contract multiplier,
        foreign exchange rate, volatility of the instrument, ewmac signal
        Output: Contracts for a given date (N)
        '''

        if signal != 0:

            N = ((self.capital * IDM * weight * tau) / 
                (multiplier * Price * FX_rate * standard_deviation))

        else:

            N = 0

        return N                

    def run_strategy(self, smas):
        data = self.data.copy()
        data['retornos', self.symbol_] = data['Close', self.symbol_].pct_change()
        data['sma_short', self.symbol_] = self.calcular_ema(self.symbol_, int(smas[0]))
        data['sma_long', self.symbol_] = self.calcular_ema(self.symbol_, int(smas[1]))
        data['posicion', self.symbol_] = np.where(data['sma_short', self.symbol_] > data['sma_long', self.symbol_], 1, -1)
        data['estrategia', self.symbol_] = data['posicion', self.symbol_].shift(1) * data['retornos', self.symbol_]
    
        #data.dropna(inplace = True)
    
        return -data[[('retornos', self.symbol_), ('estrategia',self.symbol_)]].sum().apply(np.exp)['estrategia', self.symbol_]
    
    def fig_candle_with_EMAS(self, symbol_data, symbol, emas_list, name, trade_type):
                   

        # Crear el gráfico
        fig=make_subplots( 
                        rows = 2,
                        cols=1,
                        shared_xaxes = True,
                        row_heights=[0.5, 0.5],
                        vertical_spacing = 0.06,
        )
        
        #fig = go.Figure()
        fig.add_trace(go.Candlestick(
                    x = self.data['Close', symbol_data].dropna().index,
                    open = self.data['Open', symbol_data].dropna(),
                    high = self.data['High', symbol_data].dropna(),
                    low = self.data['Low', symbol_data].dropna(),
                    close = self.data['Close', symbol_data].dropna(),
                    name=f'Precio de {symbol}'),
                    col=1,
                    row=1
                     )
        
        if ((self.strategy_number is not None) and (self.strategy_number != 8)):
            
            for j in emas_list:
                fig.add_trace(go.Scatter(x=self.data.index, 
                                        y=self.data[f'EMA{j}', symbol_data], 
                                        mode='lines', 
                                        name=f'EWMA {j}', 
                                        line_shape='spline'),
                                        col=1,
                                        row=1
                                        )
        else:
           
            fig.add_trace(go.Scatter(x=self.data.index, 
                                     y=self.data['position_size',symbol_data], 
                                     mode='lines', name='Position Size'),
                          col=1,
                          row=2
                          )

            # Add Buy Signal Markers
            fig.add_trace(go.Scatter(
                x=self.data[self.data['Signal',symbol_data] == 1].index,
                y=self.data[self.data['Signal',symbol_data] == 1]['position_size',symbol_data],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
                          col=1,
                          row=2
                          )

            # Add Sell Signal Markers
            fig.add_trace(go.Scatter(
                x=self.data[self.data['Signal',symbol_data] == -1].index,
                y=self.data[self.data['Signal',symbol_data] == -1]['position_size',symbol_data],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
                          col=1,
                          row=2
                          )
            
            fig.add_trace(go.Scatter(
                x=self.data[self.data['Signal',symbol_data] == -1].index,
                y=self.data[self.data['Signal',symbol_data] == -1]['current_position',symbol_data],
                mode='lines',
                name='Contracts in position',
                marker=dict(color='yellow')
            ),
                          col=1,
                          row=2
                          )
           
            # Add Buffer Upper line
            fig.add_trace(go.Scatter(x=self.data['Buffer_upper', symbol_data].index, 
                                     y=self.data['Buffer_upper', symbol_data], 
                                     mode='lines', name='Buffer Upper', 
                                     line=dict(dash='dash'), 
                                     marker=dict(color='white')),
                          col=1,
                          row=2
                          )

            # Add Buffer Lower line
            fig.add_trace(go.Scatter(x=self.data['Buffer_lower', symbol_data].index, 
                                     y=self.data['Buffer_lower', symbol_data], 
                                     mode='lines', name='Buffer Lower', line=dict(dash='dash')),
                          col=1,
                          row=2)

        
        fig.add_trace(go.Scatter(
                x=self.data[self.data['Open_position', symbol_data] == 1].index,
                y=self.data['Close', symbol_data][self.data['Open_position', symbol_data] == 1],
                mode= 'markers',
                name = 'BUY',
                marker=dict(
                    size=15,
                    color='blue',
                    symbol='star-triangle-up'
                ) ) ,
                    
                    col=1,
                    row=1
                    )
        
               
        # Ploteando Señales de VENTA
        fig.add_trace(go.Scatter(
            x=self.data[self.data['Open_position', symbol_data] == -1].index,
            y=self.data['Close', symbol_data][self.data['Open_position', symbol_data] == -1],
            mode= 'markers',
            name = 'SELL',
            marker=dict(
                size=15,
                color='cyan',
                symbol='star-triangle-down'
            )
                                ),
                    col=1,
                    row=1
                    )
       
       # Agregar la línea de posición_size
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Close', symbol_data],
            mode='lines+markers+text',
            name='Position Size',
            text=self.data['position_size', symbol_data],
            textposition='top center',
            line=dict(color='yellow', width=2),
            marker=dict(size=8, color='orange')
        ),
                    col=1,
                    row=1
                    ) 
        # fig.add_trace(go.Scatter(x=self.data.index, 
        #                         y=self.data['MACD', symbol_data], 
        #                         mode='lines', 
        #                         name='MACD', 
        #                         line=dict(color='cyan')
        #                             ),
        #                         col=1,
        #                         row=2
        #                         )
        # fig.add_trace(go.Scatter(x=self.data.index, 
        #                             y=self.data['Signal', symbol_data], 
        #                             mode='lines', 
        #                             name='Signal', 
        #                             line=dict(color='yellow', dash='dash')
        #                                 ),
        #                             col=1,
        #                             row=2)
        if trade_type is not None:
            if self.interval is None:
                # Estilizar el gráfico
                fig.update_layout(title=f'Strategy with EWMAS in {trade_type} of {name} ({symbol}) in 1d',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                )
            else:
                # Estilizar el gráfico
                fig.update_layout(title=f'Strategy with EWMAS in {trade_type} of {name} ({symbol}) in {self.interval}',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                )
        else:
            if self.interval is None:
                # Estilizar el gráfico
                fig.update_layout(title=f'Strategy with EWMAS of {name} ({symbol}) in 1d',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                )
            else:
                # Estilizar el gráfico
                fig.update_layout(title=f'Strategy with EWMAS  of {name} ({symbol}) in {self.interval}',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                )
        
        # Configurar el fondo del gráfico y del área de trazado como transparentes
        #fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), 
                          yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                          template='plotly_dark')
        fig.update_layout(xaxis_rangeslider_visible=False,hovermode='x unified')
        return fig

    
    def buy_and_hold_with_fixed_risk_target(self, symbol, multiplier):
       
        # Calcular los rendimientos diarios
        self.data['daily_returns', symbol] = self.data['Close', symbol].pct_change()

        self.data['daily_returns', symbol] =self.data['daily_returns', symbol].dropna()
        # Calcular la desviación estándar diaria
        daily_std_dev = self.data['daily_returns', symbol].std()

        # Calcular la desviación estándar anual
        annual_std_dev = daily_std_dev * np.sqrt(252)  # 252 días de negociación en un año
        asd = annual_std_dev #round(calculate_annual_std_dev(spx), 2) # desviación standar anual
        ewsd = self.data['Close'].ewm(span=32, min_periods=32).std()
        fx = 1 # Tasa de cambio
        
        self.data['n_position', symbol] = (self.capital * self.risk_percentage) / (multiplier * self.data['Close', symbol] * fx * asd)#.astype(int)
        self.data['n_position', symbol] = self.data['n_position', symbol].dropna().astype(int)

        self.data['n_position_variable', symbol] = (self.capital * self.risk_percentage) / (multiplier * self.data['Close', symbol] * fx * ewsd)#.astype(int)
        self.data['n_position_variable', symbol] = self.data['n_position_variable', symbol].dropna().astype(int)
        # Calcular el porcentaje máximo de pérdida
        max_loss_percentage = -self.risk_percentage
        max_loss_value = self.capital * max_loss_percentage  # Valor máximo de pérdida aceptable
    
        # Calcular la serie de capital
        capital_series = []
        position = int(self.data['n_position', symbol].dropna()[:1])
        # Iterar sobre los datos
        for _, row in self.data.iterrows():
            #print(position)
            if row['Close', symbol] * position <= max_loss_value:
                # Vender la posición si el valor alcanza la pérdida máxima
                capital_ = row['Close', symbol] * position
                position = 0
            else:
                # Mantener la posición comprada
                capital_ = row['Close', symbol] * position

            #max_loss_value = capital * max_loss_percentage  # Valor máximo de pérdida aceptable
            capital_series.append(self.capital+capital_)
        self.data['capital_series', symbol] = capital_series
        

    def html_generate_strategy_1(self):
   
        # Lee la imagen en formato binario
        with open('zenit_logo_dor.png', 'rb') as img_file:
            imagen_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
        s_type = self.sf['Definition'].unique()
        options = ''
        op = ''
        ya_opt = ''
        for t in s_type:
            options += f'''<div class="dropdown">
                            <button class="btn btn-secondary dropdown-toggle" width="100%">{t}</button>
                            <div class="dropdown-content" height="40%">
                            '''
            for k,s,sd in zip(self.sf[self.sf['Definition'] == t]['Name'], self.sf[self.sf['Definition'] == t]['Symbol'], self.sf[self.sf['Definition'] == t]['Symbol_data']):
                op += f'''
                <a href="#{s}-{sd}">{k} ({s})</a>
                '''
            opt_final = options + op + '''</div>
                                </div>
            
                                <br>'''
            ya_opt += opt_final
            options = ''
            op = ''
        today_date = str(datetime.now().date())
        style = '''
                body * {
                    box-sizing: border-box;
                }
                header {
                    display: block;
                }
                #main-header{
                            position: fixed;
                            background-color: #373a36ff;
                            top: 0;
                            width: 100%;
                            z-index: 1000; /* Asegura que el header esté por encima de otros elementos */
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Sombra del header */
                            }
                #main-header .inwrap {
                            width: 100%;
                            max-width: 80em;
                            margin: 0 auto;
                            padding: 1.5em 0;
                            display: -webkit-box;
                            display: -ms-flexbox;
                            display: flex;
                            -webkit-box-pack: justify;
                            -ms-flex-pack: justify;
                            justify-content: space-between;
                            -webkit-box-align: center;
                            -ms-flex-align: center;
                            align-items: center;
                            }
                .dropbtn {
            background-color: #4CAF50;
            color: white;
            padding: 16px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            }
            
            .dropdown {
            position: relative;
            display: inline-block;
            }
            
            .dropdown-content {
                display: none;
                position: absolute;
                background-color: #f9f9f9;
                min-width: 200px;
                max-height: 300px; /* Set max-height to enable scrolling */
                overflow-y: auto; /* Enable vertical scrolling */
                box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
                z-index: 1;
            }
            
            .dropdown-content a {
            color: black;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
            }
            
            .dropdown-content a:hover {
                        background-color: #f1f1f1;
                        }
            
            .dropdown:hover .dropdown-content {
            display: block;
            }
            
            .dropdown:hover .dropbtn {
            background-color: #3e8e41;
            }
        
        '''
        content1 = ''
        for info in self.sym_info:
            try:
                # 'Instrument', 'Name', 'Group', 'Sub-Group', 'Exchange', 'Months', 'Tick Size', 'Tick Value', 'Currency', 'Multiplier'
                symbol = info['Instrument']
                multiplier = info['Multiplier']
                tick_value = info['Tick Value']
                symbol_data = info['symbol_data']
                indicators = self.calculate_indicators(symbol_data, multiplier)
                return_fig = self.plot_daily_returns(symbol_data, symbol)
                acum_fig = self.plot_cumulative_returns(symbol_data, symbol)
                div_return_fig = pyo.plot(return_fig, output_type='div', include_plotlyjs='cdn', image_width= 600)
                div_acum_fig = pyo.plot(acum_fig, output_type='div', include_plotlyjs='cdn', image_width= 600)
                
                # String HTML de la tabla
                content = f'''
                            <table class="table">
                            <thead class="thead-dark">
                                <tr>
                                <th scope="col">Metric</th>
                                <th scope="col">Value</th>
                                </tr>
                            </thead>
                            <tbody>
                            '''
                
                # Agregar filas a la tabla con los datos del diccionario
                for key, value in indicators.items():
                    content += f'''
                                    <tr>
                                    <td>{key}</td>
                                    <td>{value}</td>
                                    </tr>'''
                
                # Cerrar la tabla HTML
                content += '''
                                </tbody>
                                </table>'''
            
                # String HTML de la tabla
                content2 = f'''
                            <table class="table">
                            
                            <tbody>
                            '''
                
                # Agregar filas a la tabla con los datos del diccionario
                for key, value in info.items():
                    content2 += f'''
                                    <tr>
                                    <td>{key}</td>
                                    <td>{value}</td>
                                    </tr>'''
                
                # Cerrar la tabla HTML
                content2 += '''
                                </tbody>
                                </table>'''
            
                content1 += f'''
                <div class="row" id="{symbol_data}">
                    <center>
                        <h1>Strategy: Buy and hold, single contract {symbol}</h1>
                    </center>
                    <br>
                    <div class="container" style="width:100%">
                        <div class="row">
                            <div class="col-sm-6">
                                {div_return_fig}
                            </div>
                            <div class="col-sm-6">
                                {div_acum_fig}
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-sm">
                                <div id="plotly-div" style="width:100%">{content2}</div>
                            </div>
                            <div class="col-sm">
                                <div id="plotly-div" style="width:100%">{content}</div>
                            </div>
                        </div>
                    </div>   
                </div>    
                '''
            except Exception as e:
                print(f'ERROR {e}')
        # Crear el archivo HTML y escribir el código de la gráfica en él
        with open(f"advance_futures_strategies_strategy_1.html", "w") as html_file:
            html_file.write(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
                <title>Gráfica Plotly</title>
                <!-- Incluir la biblioteca Plotly de CDN -->
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    {style}
                </style>
            </head>
            <body>
                <div class="container-fluid">
                    <div class="row flex-nowrap">
                        <div class="col-auto col-md-3 col-xl-2 px-sm-2 px-0" style="position: fixed; background-color: #373a36ff; z-index: 1000;">
                            <center>
                                <img src="data:image/png;base64,{imagen_base64}" class="img-fluid" alt="Imagen" max-width: 100%>
                            </center>
                            <div class="d-flex flex-column align-items-center align-items-sm-start px-3 pt-2 text-white min-vh-100">
                                
                                <div style="margin-top:10%"></div>
                                {ya_opt}
                                
                                <hr>
                                
                            </div>
                        </div>
                        <div class="col py-3" style="margin-left: 20%">
                            <h2>Last Update: {today_date}</h2>
                            {content1}
                        </div>
                    </div>
                </div>
                          
            </body>
            </html>
            """)

        logger.info(f"Archivo HTML generado y almacenado como advance_futures_strategies_strategy_1.html")
    
    def html_generate_strategy_2(self):
   
        # Lee la imagen en formato binario
        with open('zenit_logo_dor.png', 'rb') as img_file:
            imagen_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
        s_type = self.sf['Definition'].unique()
        options = ''
        op = ''
        ya_opt = ''
        for t in s_type:
            options += f'''<div class="dropdown">
                            <button class="btn btn-secondary dropdown-toggle" width="100%">{t}</button>
                            <div class="dropdown-content" height="40%">
                            '''
            for k,s,sd in zip(self.sf[self.sf['Definition'] == t]['Name'], self.sf[self.sf['Definition'] == t]['Symbol'], self.sf[self.sf['Definition'] == t]['Symbol_data']):
                op += f'''
                <a href="#{sd}">{k} ({s})</a>
                '''
            opt_final = options + op + '''</div>
                                </div>
            
                                <br>'''
            ya_opt += opt_final
            options = ''
            op = ''
        today_date = str(datetime.now().date())
        style = '''
                body * {
                    box-sizing: border-box;
                }
                header {
                    display: block;
                }
                #main-header{
                            position: fixed;
                            background-color: #373a36ff;
                            top: 0;
                            width: 100%;
                            z-index: 1000; /* Asegura que el header esté por encima de otros elementos */
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Sombra del header */
                            }
                #main-header .inwrap {
                            width: 100%;
                            max-width: 80em;
                            margin: 0 auto;
                            padding: 1.5em 0;
                            display: -webkit-box;
                            display: -ms-flexbox;
                            display: flex;
                            -webkit-box-pack: justify;
                            -ms-flex-pack: justify;
                            justify-content: space-between;
                            -webkit-box-align: center;
                            -ms-flex-align: center;
                            align-items: center;
                            }
                .dropbtn {
            background-color: #4CAF50;
            color: white;
            padding: 16px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            }
            
            .dropdown {
            position: relative;
            display: inline-block;
            }
            
            .dropdown-content {
                display: none;
                position: absolute;
                background-color: #f9f9f9;
                min-width: 200px;
                max-height: 300px; /* Set max-height to enable scrolling */
                overflow-y: auto; /* Enable vertical scrolling */
                box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
                z-index: 1;
            }
            
            .dropdown-content a {
            color: black;
            padding: 12px 16px;
            text-decoration: none;
            display: block;
            }
            
            .dropdown-content a:hover {
                        background-color: #f1f1f1;
                        }
            
            .dropdown:hover .dropdown-content {
            display: block;
            }
            
            .dropdown:hover .dropbtn {
            background-color: #3e8e41;
            }
        
        '''
        content1 = ''
        for info in self.sym_info:
            try:
                # 'Instrument', 'Name', 'Group', 'Sub-Group', 'Exchange', 'Months', 'Tick Size', 'Tick Value', 'Currency', 'Multiplier'
                symbol = info['Instrument']
                multiplier = info['Multiplier']
                tick_value = info['Tick Value']
                symbol_data = info['symbol_data']
                name = info['Name']
                data = self.buy_and_hold_with_fixed_risk_target(symbol_data, multiplier)
                return_strategy_fig = self.fig_buy_and_hold_with_fixed_risk_target(symbol_data, symbol, name)
                div_return_strategy_fig = pyo.plot(return_strategy_fig, output_type='div', include_plotlyjs='cdn', image_width= 600)
                position_time = self.fig_position_in_the_time(symbol_data, symbol, name)
                div_position_time = pyo.plot(position_time, output_type='div', include_plotlyjs='cdn', image_width= 600)
                #indicators, data = self.calculate_indicators(symbol_data, multiplier)
                return_fig = self.plot_daily_returns(symbol_data, symbol)
                acum_fig = self.plot_cumulative_returns(symbol_data, symbol)
                div_return_fig = pyo.plot(return_fig, output_type='div', include_plotlyjs='cdn', image_width= 600)
                div_acum_fig = pyo.plot(acum_fig, output_type='div', include_plotlyjs='cdn', image_width= 600)
                
                
            
                # String HTML de la tabla
                content2 = f'''
                            <table class="table">
                            
                            <tbody>
                            '''
                
                # Agregar filas a la tabla con los datos del diccionario
                for key, value in info.items():
                    content2 += f'''
                                    <tr>
                                    <td>{key}</td>
                                    <td>{value}</td>
                                    </tr>'''
                
                # Cerrar la tabla HTML
                content2 += '''
                                </tbody>
                                </table>'''
            
                content1 += f'''
                <div class="row" id="{symbol_data}">
                    <center>
                        <h1>Strategy: Buy and hold with fixed risk target in {name} ({symbol})</h1>
                    </center>
                    <br>
                    <div class="container" style="width:100%">
                        <div class="row">
                            <div class="col-12">
                                <div id="plotly-div" style="width:100%">{content2}</div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                {div_return_strategy_fig}
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                {div_position_time}
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-6">
                                {div_return_fig}
                            </div>
                            <div class="col-6">
                                {div_acum_fig}
                            </div>
                        </div>
                        
                    </div>   
                </div>    
                '''
            except Exception as e:
                print(f'ERROR {e} IN SYMBOL {symbol}')
        # Crear el archivo HTML y escribir el código de la gráfica en él
        with open(f"advance_futures_strategies_strategy_2.html", "w") as html_file:
            html_file.write(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
                <title>Gráfica Plotly</title>
                <!-- Incluir la biblioteca Plotly de CDN -->
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    {style}
                </style>
            </head>
            <body>
                <div class="container-fluid">
                    <div class="row flex-nowrap">
                        <div class="col-auto col-md-3 col-xl-2 px-sm-2 px-0" style="position: fixed; background-color: #373a36ff; z-index: 1000;">
                            <center>
                                <img src="data:image/png;base64,{imagen_base64}" class="img-fluid" alt="Imagen" max-width: 100%>
                            </center>
                            <div class="d-flex flex-column align-items-center align-items-sm-start px-3 pt-2 text-white min-vh-100">
                                
                                <div style="margin-top:10%"></div>
                                {ya_opt}
                                
                                <hr>
                                <ul>
                                    <li>Last Update: {today_date}</li>
                                    <li>Initial Capital: ${self.format_number(self.capital)}</li>
                                    <li>Risk Target: {self.risk_percentage*100}%</li>
                                </ul>
                            </div>
                        </div>
                        <div class="col py-3" style="margin-left: 20%">
                            
                            {content1}
                        </div>
                        <br>
                    </div>
                </div>
                          
            </body>
            </html>
            """)

        logger.info(f"Archivo HTML generado y almacenado como advance_futures_strategies_strategy_2.html")
    
    def rolling_call_strategy(self, symbol, s1, s2, s3):
        position = False
        print(self.data)
        self.data['Entry', symbol] = 0
        self.data['Exit', symbol] = 0
        # Calcular la variación porcentual semanal del VIX
        self.data['Weekly_Return', symbol] = ((self.data['Close', symbol] - self.data['Open', symbol]) / self.data['Open', symbol]) * 100
        #self.data['Close', symbol].pct_change(periods=1) * 100
        final_capital = self.capital
        semana = 0
        buy_contracts = 0
        sell_contracts = 0
        activity = {'Trades':[]}

        for index, row in self.data.iterrows():
            return_perc = row['Weekly_Return', symbol]
            current_price = row['Close', symbol]
            if (not position) and (return_perc <= s1):
                open_trade_price = current_price
                position = True
                self.data['Entry', symbol][index] = 1
                self.capital -= current_price * self.contract 
                buy_contracts += 1
                activity['Trades'].append(f'{index} --> BUY {self.contract} at price {current_price}')
       
            if position:
                semana += 1
                if (semana== 1) and (return_perc >= s2):
                    self.data['Exit', symbol][index] = 1
                    position = False
                    self.capital += current_price * self.contract
                    semana = 0
                    sell_contracts += 1
                    activity['Trades'].append(f'{index} --> SELL {self.contract} at price {current_price}')
                elif (semana== 2) and return_perc >= s3:
                    self.data['Exit', symbol][index] = 1
                    position = False
                    semana = 0
                    sell_contracts += 1
                    self.capital += current_price * self.contract
                    activity['Trades'].append(f'{index} --> SELL {self.contract} at price {current_price}')
                elif (semana== 3):
                    self.data['Exit', symbol][index] = 1
                    position = False
                    semana = 0
                    sell_contracts += 1
                    self.capital += current_price * self.contract
                    activity['Trades'].append(f'{index} --> SELL {self.contract} at price {current_price}')

        final_capital = (self.capital - final_capital) / final_capital
        metrics = {'Final Capital': [f'${self.format_number(self.capital)}'],
                   'Profit factor': [f'{round(final_capital*100,2)}%'],
                   'Buy contracts': [buy_contracts],
                   'Sell contracts': [sell_contracts]
                   }
        
        return metrics, activity
    
    
    def plot_rolling_call(self, symbol):
        # Crear la figura de Plotly
        fig = go.Figure()

        # Agregar el gráfico de velas (candlestick)
        fig.add_trace(go.Candlestick(x=self.data.index,
                        open=self.data['Open', symbol],
                        high=self.data['High', symbol],
                        low=self.data['Low', symbol],
                        close=self.data['Close', symbol],
                        name=f'{symbol} Price'))

        # Marcar las semanas de entrada y salida de posición con símbolos
        buy_entries = self.data[self.data['Entry', symbol] == 1]
        sell_exits = self.data[self.data['Exit', symbol] == 1]

        fig.add_trace(go.Scatter(x=buy_entries.index, y=buy_entries['Close', symbol], 
                                 mode='markers', name='BUY', 
                                 marker=dict(symbol='triangle-up', size=10, color='orange')))
        fig.add_trace(go.Scatter(x=sell_exits.index, y=sell_exits['Close', symbol], 
                                 mode='markers', name='SELL', 
                                 marker=dict(symbol='triangle-down', size=10, color='cyan')))

        # Configurar el diseño del gráfico
        fig.update_layout(title=f'Trading strategy apply to {symbol}',
                        xaxis_title='Date',
                        yaxis_title=f'{symbol} Price',
                        xaxis_rangeslider_visible=False)  # Ocultar slider de rango en eje x

        # Configurar el fondo del gráfico y del área de trazado como transparentes
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        return fig
    
    # def plot_rolling_call(self, symbol):
    #     # Crear la figura de Plotly
    #     #fig = go.Figure()
        
    #     fig=make_subplots( 
    #                     rows = 2,
    #                     cols=1,
    #                     shared_xaxes = True,
    #                     row_heights=[0.7, 0.3],
    #                     vertical_spacing = 0.06,
    #     #specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"colspan": 2}, None]]
    #     )

    #     # Agregar el gráfico de velas (candlestick)
    #     fig.add_trace(go.Candlestick(x=self.data.index,
    #                     open=self.data['Open', symbol],
    #                     high=self.data['High', symbol],
    #                     low=self.data['Low', symbol],
    #                     close=self.data['Close', symbol],
    #                     name=f'{symbol} Price'),
    #                     col=1,
    #                     row=1)

    #     # Marcar las semanas de entrada y salida de posición con símbolos
    #     buy_entries = self.data[self.data['Entry', symbol] == 1]
    #     sell_exits = self.data[self.data['Exit', symbol] == 1]

    #     fig.add_trace(go.Scatter(x=buy_entries.index, y=buy_entries['Close', symbol], 
    #                              mode='markers', name='BUY', 
    #                              marker=dict(symbol='triangle-up', size=10, color='orange')),
    #                             col=1,
    #                             row=1)
    #     fig.add_trace(go.Scatter(x=sell_exits.index, y=sell_exits['Close', symbol], 
    #                              mode='markers', name='SELL', 
    #                              marker=dict(symbol='triangle-down', size=10, color='cyan')),
    #                             col=1,
    #                             row=1)
    #     # Aqui si
        # fig.add_trace(go.Scatter(x=self.data.index, 
        #                         y=self.data[f'MACD'], 
        #                         mode='lines', 
        #                         name='MACD', 
        #                         line=dict(color='blue', dash='dash')
        #                             ),
        #                         col=1,
        #                         row=2
        #                         )
        # fig.add_trace(go.Scatter(x=self.data.index, 
        #                             y=self.data['Signal'], 
        #                             mode='lines', 
        #                             name='Signal', 
        #                             line=dict(color='green')
        #                                 ),
        #                             col=1,
        #                             row=2)
        
    #     # Configurar el diseño del gráfico
    #     fig.update_layout(title=f'Trading strategy apply to {symbol}',
    #                     xaxis_title='Date',
    #                     yaxis_title=f'{symbol} Price',
    #                     xaxis_rangeslider_visible=False)  # Ocultar slider de rango en eje x

    #     # Configurar el fondo del gráfico y del área de trazado como transparentes
    #     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    #     # Configurar color gris para las líneas de cuadrícula
    #     fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))
    #     fig.update_layout(xaxis_rangeslider_visible=False)
    #     return fig
    
    def plot_data_analysis_general(self, symbol):
        self.data['Candle_Return', symbol] = ((self.data['Close', symbol] - self.data['Open', symbol]) / self.data['Open', symbol]) * 100
        # Datos de las variaciones porcentuales semana a semana (ejemplo)
        grupos = [(i/10, (i+2)/10) for i in range(-50, 50, 2)]  # Define los intervalos de agrupación
        
        df_t = pd.concat([
                    self.data[(self.data['Candle_Return', symbol] >= interval[0]) & (self.data['Candle_Return', symbol] <= interval[1])]
                    .assign(group_histogram=f"{interval[0]} to {interval[1]}")
                    for interval in grupos
                ], ignore_index=True)
        
        
        
        var = df_t.value_counts('group_histogram')
        # Agrupar los datos de variaciones porcentuales de dos en dos
        
        # Crear un DataFrame con los datos del histograma
        data = {
            'group_histogram': list(var.index),
            'count': list(var)
        }

        df = pd.DataFrame(data)

        # Ordenar el DataFrame según los valores numéricos de los intervalos
        df['interval_numeric'] = df['group_histogram'].apply(self.interval_to_numeric)
        df = df.sort_values(by='interval_numeric').drop(columns='interval_numeric')

        # Calcular el histograma
        
        # Crear la figura de Plotly
        fig = go.Figure()

        # Añadir el histograma
        fig.add_trace(go.Bar(x=df['group_histogram'], y=df['count'], 
                            marker_color='skyblue', opacity=0.7))

        # Configurar diseño del gráfico
        fig.update_layout(title='Distribution of Percentage Variations',
                        xaxis_title='Percentage Variation Range',
                        yaxis_title='Frecuency',
                        bargap=0.1)

        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))
        # Mostrar la gráfica
        return fig
    
    def plot_data_analysis_month(self, symbol, months, date_name):
        #self.data['Weekly_Return', symbol] = ((self.data['Close', symbol] - self.data['Open', symbol]) / self.data['Open', symbol]) * 100
        # Datos de las variaciones porcentuales semana a semana (ejemplo)
        variaciones = self.data[self.data.index.month == months]

        grupos = [(i/10, (i+2)/10) for i in range(-50, 50, 2)]  # Define los intervalos de agrupación
        
        df_t = pd.concat([
                    variaciones[(variaciones['Candle_Return', symbol] >= interval[0]) & (variaciones['Candle_Return', symbol] <= interval[1])]
                    .assign(group_histogram=f"{interval[0]} to {interval[1]}")
                    for interval in grupos
                ], ignore_index=True)
        
        var = df_t.value_counts('group_histogram')
        # Agrupar los datos de variaciones porcentuales de dos en dos
        
        # Crear un DataFrame con los datos del histograma
        data = {
            'group_histogram': list(var.index),
            'count': list(var)
        }

        df = pd.DataFrame(data)

        # Ordenar el DataFrame según los valores numéricos de los intervalos
        df['interval_numeric'] = df['group_histogram'].apply(self.interval_to_numeric)
        df = df.sort_values(by='interval_numeric').drop(columns='interval_numeric')

        # Crear la figura de Plotly
        fig = go.Figure()

        # Añadir el histograma
        fig.add_trace(go.Bar(x=df['group_histogram'], y=df['count'], 
                            marker_color='cyan', opacity=0.7))

        # Configurar diseño del gráfico
        fig.update_layout(title=f'Distribution of Percentage Variations in month {date_name}',
                        xaxis_title='Percentage Variation Range',
                        yaxis_title='Frecuency',
                        bargap=0.1)
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Configurar color gris para las líneas de cuadrícula
        fig.update_layout(xaxis=dict(showgrid=True, gridcolor='lightgrey'), yaxis=dict(showgrid=True, gridcolor='lightgrey'))
        # Mostrar la gráfica
        return fig

    def financial_analysis(self, symbol):
        # Download data from yfinance
        returns = self.data['Close', symbol].pct_change().dropna()
        if self.interval in ['1m', '2m', '5m', '15m', '30m', '90m']:
            annualized_sd = np.std(returns)
        else:
            annualized_sd = np.std(returns) * np.sqrt(252)
        # Calculate statistics
        results = {
            'mean': [f'{round(returns.mean() * 100, 4)}%'],
            'median': [f'{round(returns.median()*100, 4)}%'],
            'variance': [f'{round(returns.var()*100,4)}%'],
            'standard_deviation': [f'{round(returns.std()*100, 4)}%'],
            'coefficient_of_variation': [round(returns.std() / returns.mean(), 3)],
            'skewness': [round(returns.skew(),4)],
            'kurtosis': [round(returns.kurtosis(), 4)],
            #'total_return': [f'{round((returns[-1] / returns[0])*100, 4)}%'],
            'annualized_return': [f'{round(annualized_sd*100, 4)}%'],
            'var': [f'{round(np.percentile(returns, 5),4)}%'],
            'cvar': [f'{round(returns[returns <= np.percentile(returns, 5)].mean()*100, 4)}%'],
            # 'pearson_correlation': [returns.corr(method='pearson')],
            # 'spearman_correlation': [returns.corr(method='spearman')],
            #'covariance': [returns.cov()]
        }

        return results

    def portfolio_sharpe_ratio(self, df):
        
        # Calcular el rendimiento de cada instrumento
        df['Return'] = df['PnL_usd'] / df['Initial Capital']

        # Calcular el peso de cada instrumento en el portafolio
        df['Weight'] = df['Initial Capital'] / self.capital

        # Calcular el rendimiento ponderado de cada instrumento
        df['Weighted_Return'] = df['Return'] * df['Weight']

        # Calcular el rendimiento total del portafolio
        portfolio_return = df['Weighted_Return'].sum()

        # Calcular la volatilidad del portafolio (desviación estándar de los rendimientos ponderados)
        portfolio_volatility = np.sqrt(np.dot(df['Weight'], df['Return'] ** 2) - portfolio_return ** 2)

        # Tasa libre de riesgo (suponemos 0 para este cálculo)
        risk_free_rate = 0

        # Calcular el Sharpe Ratio del portafolio
        portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        return portfolio_sharpe_ratio


if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='max', help='Data Period')
    parser.add_argument('--capital', type=float, default=1000000, help='Initial capital')
    parser.add_argument('--risk_percentage', type=float, default=0.20, help='Risk Target')
   
   
    
    
    args = parser.parse_args()
    logger.info(f"args {args}")

    bot = AdvanceFutureStrategy(
            args.period,
            args.capital,
            args.risk_percentage
              )
    try:
        bot.main()
    except KeyboardInterrupt:
        logger.info('Chao :D')

