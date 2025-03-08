import argparse
import sqlite3
import pandas as pd
import asyncio
from utils_functions import (load_data_from_db, 
                             update_column_value,
                             convert_bars_to_dataframe_db,
                             calculate_date_difference,
                             insert_dataframe_to_table)
from ib_insync import IB, Future
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

def update_expiration(ib, row):
    try:
        base_contract = Future(symbol=row["broker_symbol"], exchange=row["broker_exchange"])
        contracts = ib.reqContractDetails(base_contract)

        today = datetime.today()
        min_date = today + timedelta(days=20)  # Fecha mínima permitida (15 días desde hoy)

        # Filtrar fechas de expiración válidas
        expiration_dates = [
            detail.contract.lastTradeDateOrContractMonth
            for detail in contracts
            if datetime.strptime(detail.contract.lastTradeDateOrContractMonth, '%Y%m%d') >= min_date
        ]

        if expiration_dates:
            # Ordenar fechas de expiración
            sorted_expiration_dates = sorted(
                expiration_dates,
                key=lambda date: datetime.strptime(date, '%Y%m%d')
            )

            # Seleccionar fechas de expiración válidas
            actual_expiration = sorted_expiration_dates[0]
            next_expiration = None

            # Encontrar la siguiente fecha de expiración válida que sea diferente a la actual
            for date in sorted_expiration_dates[1:]:
                if date != actual_expiration:
                    next_expiration = date
                    break

            # Validar que haya una segunda fecha de expiración
            if next_expiration:
                update_column_value("general_futures_info_carver", 
                                    "expiration_actual", 
                                    actual_expiration, 
                                    f"instrument = '{row['instrument']}'")
                update_column_value("general_futures_info_carver", 
                                    "expiration_next", 
                                    next_expiration, 
                                    f"instrument = '{row['instrument']}'")
                print(f"Actualizado {row['broker_symbol']}: Actual {actual_expiration}, Siguiente {next_expiration}")
            else:
                print(f"No se encontró una segunda fecha válida para {row['broker_symbol']}")

    except Exception as e:
        print(f"Error actualizando contrato {row['broker_symbol']}: {e}")
    finally:
        ib.disconnect()


def update_prices(ib, row):
    try:
               
        prices = load_data_from_db(table_name=row["instrument"], last_row=True)
        start_date = prices["DATE"][0]
        contract = Future(symbol=row["broker_symbol"], exchange=row["broker_exchange"])
        contract.secType = "CONTFUT"
        end_date = datetime.today().strftime("%Y-%m-%d")
        
        duration = calculate_date_difference(start_date, end_date)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        if len(bars) > 0:
            df = convert_bars_to_dataframe_db(bars)
            df = df[(df["DATE"] > start_date) & (df["DATE"] < end_date)]
            df["DATE"] = df["DATE"].astype(str)
            print(f"Datos descargados: {df.head()}")
            insert_dataframe_to_table(row["instrument"], df)
      
    except Exception as e:
        print(f"Error actualizando precios para {row['broker_symbol']}: {e}")
    finally:
        ib.disconnect()


def update_futures_data(ip, 
                        port, 
                        symbols):
    port = int(port)
    symbols = eval(symbols)  # Convert the symbols from string to list
    general_info = load_data_from_db(table_name="general_futures_info_carver")
    symbols = symbols + ['EUR','AUD','CAD','JPY','HKD','CHF','CNH','GBP','INR','MXP','KRWUSD','SGD','SEK']
    filtered_info = general_info[general_info["broker_symbol"].isin(symbols)]

    def thread_worker(row):
        """Worker function to handle expiration and price updates within a thread."""
        # Create an event loop for the thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()

        # Create a new IB connection for this thread
        ib = IB()
        
        try:
            ib.connect(ip, port, clientId=datetime.now().microsecond)
            # Perform both expiration and price updates
            update_expiration(ib, row)
            
            ib.connect(ip, port, clientId=datetime.now().microsecond)
            update_prices(ib, row)
        except Exception as e:
            print(f"Error processing {row['broker_symbol']}: {e}")

    # Process all rows concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(thread_worker, [row for _, row in filtered_info.iterrows()])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actualizar datos de futuros en la base de datos")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Dirección IP del servidor TWS/IB Gateway")
    parser.add_argument("--port", type=int, default=7496, help="Puerto del servidor TWS/IB Gateway")
    parser.add_argument("--symbols", type=str, default="['ES','NQ']", help="Portfolio Symbol list")
    args = parser.parse_args()

    update_futures_data(args.ip, args.port, args.symbols)
