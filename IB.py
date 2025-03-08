
from utils_functions import *
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actualizar datos de futuros en la base de datos")
    parser.add_argument("--contracts", type=str, default="{'ES':{'multiplier':50, 'region':'US', 'asset_class':'index'}}", help="Contract information")
    parser.add_argument("--sectors", type=str, default="['metals']", help="Asset class sector")
    parser.add_argument("--look_back_period", type=int, default=36, help="Look back period")
    parser.add_argument("--forecast_scalars", type=str, default="[0.5, 1.0, 1.5]", help="Forecast scalars")
    parser.add_argument("--ewma_weights", type=str, default="[0.4, 0.6]", help="EWMA weights")
    parser.add_argument("--fdm", type=float, default=1.0, help="FDM value")
    parser.add_argument("--multipliers", type=str, default="{'ES': 50}", help="Asset multipliers")
    parser.add_argument("--daily_cash_vol_target", type=float, default=250000, help="Daily cash volatility target")
    parser.add_argument("--portfolio_name", type=str, default="default_portfolio", help="Portfolio name")
    parser.add_argument("--ewmacs", type=str, default="[2,4,8,16,32,64]", help="EWMAC pairs")
    parser.add_argument("--ewmacs_final_weight", type=float, default=0.6, help="EWMAC final weight")
    parser.add_argument("--carry_final_weight", type=float, default=0.4, help="Carry final weight")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address of TWS/IB Gateway")
    parser.add_argument("--port", type=int, default=7496, help="Port of TWS/IB Gateway")
    parser.add_argument("--account", type=str, default="DU7186453", help="Port of TWS/IB Gateway")
    args = parser.parse_args()

    # Convertir cadenas en estructuras reales
    contracts = eval(args.contracts)
    sectors = eval(args.sectors)
    forecast_scalars = eval(args.forecast_scalars)
    ewma_weights = eval(args.ewma_weights)
    multipliers = eval(args.multipliers)
    ewmacs = eval(args.ewmacs)

    # Llamar a la función principal
    process_multiple_assets(
        contracts,
        sectors,
        args.look_back_period,
        forecast_scalars,
        ewma_weights,
        args.fdm,
        multipliers,
        args.daily_cash_vol_target,
        args.portfolio_name,
        ewmacs,
        args.ewmacs_final_weight,
        args.carry_final_weight,
        ip=args.ip,
        port=args.port,
        account=args.account
    )
