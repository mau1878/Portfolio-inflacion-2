import yfinance as yf
from datetime import datetime, date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import logging
import re

# ------------------------------
# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Inflación mensual estimada (datos corregidos y actualizados)
inflation_rates = {
    2017: [1.6, 2.5, 2.4, 2.6, 1.3, 1.2, 1.7, 1.4, 1.9, 1.5, 1.4, 3.1],
    2018: [1.8, 2.4, 2.3, 2.7, 2.1, 3.7, 3.1, 3.9, 6.5, 5.4, 3.2, 2.6],
    2019: [2.9, 3.8, 4.7, 3.4, 3.1, 2.7, 2.2, 4.0, 5.9, 3.3, 4.3, 3.7],
    2020: [2.3, 2.0, 3.3, 1.5, 1.5, 2.2, 1.9, 2.7, 2.8, 3.8, 3.2, 4.0],
    2021: [4.0, 3.6, 4.8, 4.1, 3.3, 3.2, 3.0, 2.5, 3.5, 3.5, 2.5, 3.8],
    2022: [3.9, 4.7, 6.7, 6.0, 5.1, 5.3, 7.4, 7.0, 6.2, 6.3, 4.9, 5.1],
    2023: [6.0, 6.6, 7.7, 8.4, 7.8, 6.0, 6.3, 12.4, 12.7, 8.3, 12.8, 25.5],
    2024: [20.6, 13.2, 11.0, 9.2, 4.2, 4.6, 4.2, 3.5, 3.5, 3.3, 3.6, 3.3]  # Estimación ficticia
}

# ------------------------------
# Diccionario de tickers y sus divisores (se puede ajustar si es necesario)
splits = {
    'AGRO.BA': (6, 2.1)  # Ajustes para AGRO.BA
    # Añade más tickers con splits si es necesario
}

# ------------------------------
# Función para ajustar precios por splits
def ajustar_precios_por_splits(df, ticker):
    try:
        if ticker in splits:
            split_info = splits[ticker]
            split_date = datetime(2023, 11, 3)  # Fecha de split específica para AGRO.BA
            df.loc[df['Date'] < split_date, 'Close'] /= split_info[0]
            df.loc[df['Date'] >= split_date, 'Close'] *= split_info[1]
            logger.info(f"Ajuste por split aplicado al ticker {ticker}")
    except Exception as e:
        logger.error(f"Error ajustando splits para {ticker}: {e}")
    return df

# ------------------------------
# Función para calcular inflación diaria acumulada dentro de un rango de fechas
def calcular_inflacion_diaria_rango(df, start_year, start_month, end_year, end_month):
    cumulative_inflation = [1.0]  # Inicia con 1.0 para no alterar los valores

    try:
        for year in range(start_year, end_year + 1):
            if year not in inflation_rates:
                logger.warning(f"Tasa de inflación no disponible para el año {year}. Se omitirá.")
                continue

            monthly_inflation = inflation_rates[year]

            # Define el rango de meses para el año actual
            if year == start_year:
                months = range(start_month - 1, 12)  # Desde el mes de inicio hasta diciembre
            elif year == end_year:
                months = range(0, end_month)  # Desde enero hasta el mes final
            else:
                months = range(0, 12)  # Año completo

            for month in months:
                # Filtrar los días que pertenecen al mes actual
                days_in_month_mask = (df['Date'].dt.year == year) & (df['Date'].dt.month == (month + 1))
                num_days = days_in_month_mask.sum()
                if num_days > 0:
                    monthly_rate = monthly_inflation[month]
                    # Calcular la tasa diaria
                    daily_rate = (1 + monthly_rate / 100) ** (1 / num_days) - 1
                    logger.debug(f"Año: {year}, Mes: {month+1}, Tasa Mensual: {monthly_rate}%, Tasa Diaria: {daily_rate:.6f}")
                    # Aplicar la tasa diaria para cada día
                    for _ in range(num_days):
                        cumulative_inflation.append(cumulative_inflation[-1] * (1 + daily_rate))
    except Exception as e:
        logger.error(f"Error calculando inflación: {e}")

    logger.info(f"Cálculo de inflación completado. Total de días calculados: {len(cumulative_inflation)-1}")
    return cumulative_inflation[1:]  # Remover el valor inicial de 1.0

# ------------------------------
# Funciones de caché para optimizar rendimiento
@st.cache_data(ttl=86400)  # Cache de un día
def descargar_datos(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    if not stock_data.empty:
        stock_data.reset_index(inplace=True)
        stock_data = ajustar_precios_por_splits(stock_data, ticker)
        stock_data = stock_data[['Date', 'Close']].rename(columns={'Close': ticker})
    else:
        logger.warning(f"No se encontraron datos para el ticker {ticker} en el rango {start} - {end}.")
    return stock_data

# ------------------------------
# Interfaz de Usuario con Streamlit
st.title("Simulador de Portafolio con Transacciones y Comparación con Inflación")

st.markdown("""
Esta aplicación permite simular cambios en un portafolio de acciones a lo largo del tiempo y comparar su rendimiento con la inflación en Argentina.

## Instrucciones:
1. **Definir Transacciones:** Agrega las transacciones donde añades o retiras tickers en fechas específicas.
2. **Simulación:** La aplicación calculará el valor del portafolio día a día según las transacciones ingresadas.
3. **Comparación con Inflación:** Verás si el portafolio se mantuvo por encima de la inflación acumulada.

**Nota:** Asegúrate de ingresar las fechas en orden cronológico para una simulación precisa.
""")

# ------------------------------
# Sección para ingresar transacciones
st.header("Definir Transacciones de Portafolio")

with st.form(key='transaction_form'):
    num_trans = st.number_input("Número de Transacciones", min_value=1, max_value=20, value=3, step=1)
    transaction_list = []
    for i in range(int(num_trans)):
        st.subheader(f"Transacción {i+1}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date_input = st.date_input(f"Fecha {i+1}", date(2023, 1, 1), key=f'date_{i}')
        with col2:
            action = st.selectbox(f"Acción {i+1}", ('Agregar', 'Retirar'), key=f'action_{i}')
        with col3:
            ticker = st.text_input(f"Ticker {i+1} (e.g., AAPL.BA)", key=f'ticker_{i}')
        with col4:
            quantity = st.number_input(f"Cantidad {i+1}", min_value=1, value=1, step=1, key=f'quantity_{i}')
        transaction_list.append({
            'Date': date_input,
            'Action': action,
            'Ticker': ticker.upper(),
            'Quantity': quantity
        })
    submit_button = st.form_submit_button(label='Actualizar Transacciones')

if submit_button:
    st.success("Transacciones actualizadas!")

# ------------------------------
# Procesar transacciones
if 'transaction_list' not in st.session_state:
    st.session_state.transaction_list = transaction_list
else:
    st.session_state.transaction_list = transaction_list

df_transactions = pd.DataFrame(st.session_state.transaction_list)
df_transactions.sort_values('Date', inplace=True)

st.subheader("Lista de Transacciones")
st.dataframe(df_transactions)

if df_transactions.empty:
    st.warning("No hay transacciones definidas.")
else:
    # Determinar el rango de fechas
    start_date = df_transactions['Date'].min()
    end_date = df_transactions['Date'].max()

    # Descargar datos para todos los tickers involucrados
    tickers = df_transactions['Ticker'].unique().tolist()

    data_frames = []
    for ticker in tickers:
        try:
            stock_data = descargar_datos(ticker, start_date, end_date + pd.Timedelta(days=1))
            if not stock_data.empty:
                data_frames.append(stock_data)
            else:
                st.warning(f"No se encontraron datos para {ticker} en el rango de fechas especificado.")
        except Exception as e:
            st.error(f"Error al descargar datos para {ticker}: {e}")

    if data_frames:
        try:
            # Merge todos los dataframes en uno solo
            df_merged = data_frames[0]
            for df in data_frames[1:]:
                df_merged = pd.merge(df_merged, df, on='Date', how='outer')

            df_merged.sort_values('Date', inplace=True)
            df_merged.fillna(method='ffill', inplace=True)
            df_merged.fillna(method='bfill', inplace=True)

            # Crear un DataFrame de transacciones por fecha y ticker
            portfolio = {}
            for index, row in df_transactions.iterrows():
                txn_date = pd.to_datetime(row['Date'])
                if txn_date not in portfolio:
                    portfolio[txn_date] = {}
                action = row['Action']
                ticker = row['Ticker']
                quantity = row['Quantity']
                if action == 'Agregar':
                    portfolio[txn_date][ticker] = portfolio[txn_date].get(t
