import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

# ----------------------------
# Black-Scholes 기반 옵션 분석 함수
# ----------------------------
def black_scholes_analysis(option_type, S, K, T, r, sigma):
    """
    option_type: 'call' 또는 'put'
    S : 현재 기초자산 가격
    K : 행사가격
    T : 만기까지 남은 시간 (연 단위)
    r : 무위험 이자율
    sigma : 기초자산 변동성 (연율)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return {
        'Option Type': option_type,
        'Spot Price (S)': S,
        'Strike Price (K)': K,
        'Time to Expiry (T)': T,
        'Interest Rate (r)': r,
        'Volatility (σ)': sigma,
        'd1': d1,
        'd2': d2,
        'Price': price,
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

# ----------------------------
# 옵션 가격 및 Greeks 변화 분석을 위한 데이터프레임 생성
# ----------------------------
def generate_analysis_dataframe(option_type, K, T, r, sigma, S_min=80, S_max=120, num_points=50):
    S_range = np.linspace(S_min, S_max, num_points)
    results = []
    for S in S_range:
        res = black_scholes_analysis(option_type, S, K, T, r, sigma)
        results.append(res)
    return pd.DataFrame(results)

# ----------------------------
# matplotlib를 이용한 정적 시각화 함수
# ----------------------------
def plot_static(df):
    plt.figure(figsize=(12, 8))
    plt.plot(df['Spot Price (S)'], df['Price'], label='Price', lw=2)
    plt.plot(df['Spot Price (S)'], df['Delta'], label='Delta', lw=2)
    plt.plot(df['Spot Price (S)'], df['Gamma'], label='Gamma', lw=2)
    plt.plot(df['Spot Price (S)'], df['Theta'], label='Theta', lw=2)
    plt.plot(df['Spot Price (S)'], df['Vega'], label='Vega', lw=2)
    plt.plot(df['Spot Price (S)'], df['Rho'], label='Rho', lw=2)
    plt.xlabel('Underlying Price (S)')
    plt.ylabel('Value')
    plt.title('Option Price and Greeks vs Underlying Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Plotly를 이용한 인터랙티브 시각화 함수
# ----------------------------
def plot_interactive(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Spot Price (S)'], y=df['Price'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df['Spot Price (S)'], y=df['Delta'], mode='lines', name='Delta'))
    fig.add_trace(go.Scatter(x=df['Spot Price (S)'], y=df['Gamma'], mode='lines', name='Gamma'))
    fig.add_trace(go.Scatter(x=df['Spot Price (S)'], y=df['Theta'], mode='lines', name='Theta'))
    fig.add_trace(go.Scatter(x=df['Spot Price (S)'], y=df['Vega'], mode='lines', name='Vega'))
    fig.add_trace(go.Scatter(x=df['Spot Price (S)'], y=df['Rho'], mode='lines', name='Rho'))
    fig.update_layout(
        title='Option Price and Greeks vs Underlying Price (Interactive)',
        xaxis_title='Underlying Price (S)',
        yaxis_title='Value',
        template='plotly_white',
        legend_title='Metric'
    )
    fig.show()

# ----------------------------
# 사용자 입력에 따른 옵션 분석 및 시각화
# ----------------------------
def user_input_analysis():
    print("옵션의 요소들을 입력하세요:")
    option_type = input("옵션 타입 (call/put): ").strip().lower()
    S = float(input("기초자산 가격 (S): "))
    K = float(input("행사가격 (K): "))
    T = float(input("만기까지 남은 시간 (T, 연 단위): "))
    r = float(input("이자율 (r, 예: 0.05 for 5%): "))
    sigma = float(input("변동성 (σ, 예: 0.2 for 20%): "))
    
    # 단일 옵션 분석 결과 출력
    analysis = black_scholes_analysis(option_type, S, K, T, r, sigma)
    df_result = pd.DataFrame([analysis])
    print("\n옵션 분석 결과:")
    print(df_result)
    
    # 기초자산 가격 범위를 설정하여 시각화를 위한 데이터프레임 생성
    S_min = S * 0.8
    S_max = S * 1.2
    df_range = generate_analysis_dataframe(option_type, K, T, r, sigma, S_min, S_max, num_points=50)
    
    # matplotlib를 이용한 정적 시각화
    plot_static(df_range)
    # plotly를 이용한 인터랙티브 시각화
    plot_interactive(df_range)

# ----------------------------
# 메인 실행 부분
# ----------------------------
if __name__ == "__main__":
    user_input_analysis()
