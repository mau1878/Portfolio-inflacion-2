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
# Inflación mensual estimada (datos corregidos)
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
  # Añade más tickers con splits si necesario
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
def calcular_inflacion_diaria_rango(df):
  # Crear una copia de la fecha
  dates = df['Date']
  years = dates.dt.year
  months = dates.dt.month

  daily_inflation = []

  for date_idx, current_date in enumerate(dates):
      year = current_date.year
      month = current_date.month
      if year in inflation_rates:
          monthly_rate = inflation_rates[year][month -1]
          # Get number of days in the month
          num_days = dates[(years == year) & (months == month)].shape[0]
          # Calculate daily rate
          daily_rate = (1 + monthly_rate / 100) ** (1 / num_days) - 1
          daily_inflation.append(daily_rate)
          logger.debug(f"Date: {current_date}, Year: {year}, Month: {month}, Monthly Rate: {monthly_rate}%, Daily Rate: {daily_rate:.6f}")
      else:
          # No inflation data, assume no inflation
          daily_inflation.append(0.0)
          logger.warning(f"No tasa de inflación para el año {year}. Asumiendo 0% para este día.")

  # Calcular Inflation Index como factor acumulativo
  cumulative_inflation = []
  current_cumulative = 1.0
  for rate in daily_inflation:
      current_cumulative *= (1 + rate)
      cumulative_inflation.append(current_cumulative)

  return cumulative_inflation

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
  num_trans = st.number_input("Número de Transacciones", min_value=1, max_value=50, value=3, step=1)
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
  end_date = date.today()

  # Descargar datos para todos los tickers involucrados
  tickers = df_transactions['Ticker'].unique().tolist()

  data_frames = []
  for ticker in tickers:
      try:
          stock_data = descargar_datos(ticker, start_date, end_date)
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
          portfolio_transactions = {}
          for index, row in df_transactions.iterrows():
              txn_date = pd.to_datetime(row['Date'])
              if txn_date not in portfolio_transactions:
                  portfolio_transactions[txn_date] = {}
              action = row['Action']
              ticker = row['Ticker']
              quantity = row['Quantity']
              if action == 'Agregar':
                  portfolio_transactions[txn_date][ticker] = portfolio_transactions[txn_date].get(ticker, 0) + quantity
              elif action == 'Retirar':
                  portfolio_transactions[txn_date][ticker] = portfolio_transactions[txn_date].get(ticker, 0) - quantity

          # Inicializar holdings
          holdings = {}
          portfolio_values = []

          # Resetear el índice a 'Date' para fácil iteración
          df_merged.reset_index(inplace=True)
          all_dates = df_merged['Date']

          # Calcular inflación acumulada
          cumulative_inflation = calcular_inflacion_diaria_rango(df_merged)

          # Agregar 'Inflacion_Acumulada' al DataFrame
          df_merged['Inflacion_Acumulada'] = cumulative_inflation

          # Simular el portafolio y calcular valor diario
          for idx, current_date in enumerate(all_dates):
              # Actualizar holdings si hay transacciones en esta fecha
              if current_date in portfolio_transactions:
                  for ticker, qty_change in portfolio_transactions[current_date].items():
                      holdings[ticker] = holdings.get(ticker, 0) + qty_change
                      if holdings[ticker] <= 0:
                          del holdings[ticker]

              # Calcular el valor del portafolio actual
              total_value = 0
              for ticker, qty in holdings.items():
                  price = current_date[ticker] if ticker in current_date and not pd.isna(current_date[ticker]) else 0
                  total_value += price * qty
              portfolio_values.append(total_value)

          # Agregar 'Portfolio_Value' al DataFrame
          df_merged['Portfolio_Value'] = portfolio_values

          # Calcular 'Inflacion_Index' starting at first investment date
          first_investment_idx = df_merged[df_merged['Portfolio_Value'] > 0].index.min()
          if pd.isna(first_investment_idx):
              st.warning("No hay ninguna transacción que agregue holdings.")
              st.stop()
          
          # Set 'Inflacion_Index' to start at 1 on first investment date
          df_merged['Inflacion_Index'] = np.nan
          df_merged.loc[first_investment_idx, 'Inflacion_Index'] = 1.0

          # Calculate 'Inflacion_Index' from first_investment_idx onwards
          for idx in range(first_investment_idx +1, len(df_merged)):
              previous_inflation = df_merged.at[idx -1, 'Inflacion_Index']
              if not pd.isna(previous_inflation):
                  # Current day cumulative_inflation / cumulative_inflation on first_investment_date
                  infl_change = df_merged.at[idx, 'Inflacion_Acumulada'] / df_merged.at[first_investment_idx, 'Inflacion_Acumulada']
                  df_merged.at[idx, 'Inflacion_Index'] = infl_change

          # Normalizar Inflacion_Index para empezar en 1
          df_merged.loc[first_investment_idx:, 'Inflacion_Index'] = df_merged.loc[first_investment_idx:, 'Inflacion_Index'].fillna(method='ffill')

          # Escalar 'Inflacion_Index' para que coincida con el valor inicial del portafolio
          initial_portfolio_value = df_merged.at[first_investment_idx, 'Portfolio_Value']
          df_merged['Inflacion_Index'] = df_merged['Inflacion_Index'] * initial_portfolio_value

          # Plotting con Plotly
          fig = go.Figure()
          # Portfolio_Value
          fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['Portfolio_Value'],
                                   name='Portafolio', mode='lines', line=dict(color='blue')))
          # Inflacion_Index, scaled to initial portfolio value
          fig.add_trace(go.Scatter(x=df_merged['Date'], y=df_merged['Inflacion_Index'],
                                   name='Inflación', mode='lines', line=dict(color='red', dash='dash')))

          fig.update_layout(
              title='Valor del Portafolio vs Inflación Acumulada',
              xaxis_title='Fecha',
              yaxis_title='Valor (ARS)',
              height=600,
              width=900,
              plot_bgcolor='black',
              paper_bgcolor='black',
              font=dict(color='white'),
              hovermode='closest'
          )
          fig.update_xaxes(showgrid=True, gridcolor='gray')
          fig.update_yaxes(showgrid=True, gridcolor='gray')

          st.plotly_chart(fig, use_container_width=True)

          # Análisis final
          final_portfolio = df_merged['Portfolio_Value'].iloc[-1]
          final_inflation = df_merged['Inflacion_Index'].iloc[-1]
          difference = final_portfolio - final_inflation

          st.markdown(f"**Valor Inicial del Portafolio:** {initial_portfolio_value:.2f} ARS")
          st.markdown(f"**Valor Final del Portafolio:** {final_portfolio:.2f} ARS")
          st.markdown(f"**Valor Final Ajustado por Inflación:** {final_inflation:.2f} ARS")
          st.markdown(f"**Diferencia (Portafolio - Inflación):** {difference:.2f} ARS")
          if difference > 0:
              st.success("¡El portafolio se mantuvo por encima de la inflación!")
          else:
              st.error("El portafolio no logró mantenerse por encima de la inflación.")

      except Exception as e:
          st.error(f"Ocurrió un error durante la simulación: {e}")
  else:
      st.warning("No se pudieron descargar los datos necesarios para la simulación.")
