
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import curve_fit

st.set_page_config(
    page_title="선박 회귀분석 실습",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 선박 데이터 선형·비선형 회귀분석 실습")
st.caption("AIS 선박 데이터를 업로드하여 선형회귀와 2차 비선형회귀를 직접 실습할 수 있습니다.")

# ---------------------------------------------------------
# 알고리즘 설명
# ---------------------------------------------------------
with st.expander("📘 알고리즘 보기", expanded=False):
    st.markdown("""
### 1. 데이터 입력
CSV 또는 Excel 파일을 업로드하고 숫자형 열 중에서 독립변수 X와 종속변수 Y를 선택합니다.

### 2. 선형회귀
선형회귀식은 다음과 같은 형태를 사용합니다.

**y = ax + b**

경사하강법(Gradient Descent)을 이용하여 평균제곱오차(MSE)가 작아지는 방향으로
기울기와 절편을 반복적으로 갱신합니다.

### 3. 비선형회귀
이번 실습에서는 2차 다항식 모델을 사용합니다.

**y = ax² + bx + c**

`curve_fit()`을 이용하여 데이터에 가장 잘 맞는 계수를 찾습니다.

### 4. 결과 비교
실제 데이터와 두 회귀모델을 그래프로 표시하고,
각 모델의 MSE(평균제곱오차)를 비교합니다.

> 강의자료의 단순선형회귀, 잔차분석, 컴퓨터의 선형회귀 수행 방법,
> 경사하강법 개념을 바탕으로 구현했습니다.
""")

# ---------------------------------------------------------
# 예제 데이터
# ---------------------------------------------------------
demo_df = pd.DataFrame({
    "선박길이_m": [50, 75, 100, 125, 150, 175, 200],
    "선박속도_knots": [10.2, 11.8, 14.5, 15.7, 18.3, 19.6, 22.1]
})

st.subheader("1. 데이터 입력")

mode = st.radio(
    "입력 방법",
    ["파일 업로드", "예제 선박 데이터"],
    horizontal=True
)

df = None

if mode == "파일 업로드":
    uploaded = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요.",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                # UTF-8 계열 우선, 실패하면 CP949로 재시도
                raw = uploaded.getvalue()
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(raw), encoding="cp949")
            else:
                df = pd.read_excel(uploaded)

            st.success(f"파일을 불러왔습니다: {uploaded.name}")
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            st.stop()
    else:
        st.info("CSV 또는 Excel 파일을 업로드하세요.")

else:
    df = demo_df.copy()
    st.info("예제 선박 데이터가 선택되었습니다.")

if df is None:
    st.stop()

st.write("#### 입력 데이터")
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# 숫자형 열 선택
# ---------------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("회귀분석을 위해 숫자형 열이 최소 2개 필요합니다.")
    st.stop()

c1, c2 = st.columns(2)

with c1:
    default_x = 0
    x_col = st.selectbox(
        "독립변수 X (원인/입력)",
        numeric_cols,
        index=default_x
    )

with c2:
    default_y = 1 if len(numeric_cols) > 1 else 0
    y_options = [c for c in numeric_cols if c != x_col]
    y_col = st.selectbox(
        "종속변수 Y (결과/예측)",
        y_options,
        index=0
    )

# ---------------------------------------------------------
# 데이터 전처리
# ---------------------------------------------------------
work = df[[x_col, y_col]].copy()
work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
work = work.dropna()

# 유한값만 사용
work = work[
    np.isfinite(work[x_col].to_numpy()) &
    np.isfinite(work[y_col].to_numpy())
]

if len(work) < 3:
    st.error("유효한 데이터가 3개 이상 필요합니다.")
    st.stop()

x = work[x_col].to_numpy(dtype=float)
y = work[y_col].to_numpy(dtype=float)

# ---------------------------------------------------------
# 회귀 함수
# ---------------------------------------------------------
def quadratic(z, a, b, c):
    return a * z**2 + b * z + c

def linear_regression_gradient_descent(x_raw, y_raw, learning_rate=0.05, epochs=5000):
    """
    표준화한 x로 경사하강법을 안정적으로 수행한 뒤
    원래 단위의 y = a*x + b 형태로 변환한다.
    """
    x_mean = np.mean(x_raw)
    x_std = np.std(x_raw)
    if x_std == 0:
        raise ValueError("독립변수 X의 값이 모두 같아서 회귀를 수행할 수 없습니다.")

    xs = (x_raw - x_mean) / x_std

    a_s = 0.0
    b_s = 0.0
    n = len(xs)

    for _ in range(epochs):
        y_pred_s = a_s * xs + b_s
        da = (-2.0 / n) * np.sum(xs * (y_raw - y_pred_s))
        db = (-2.0 / n) * np.sum(y_raw - y_pred_s)

        a_s -= learning_rate * da
        b_s -= learning_rate * db

    # y = a_s*((x-mu)/sigma) + b_s
    #   = (a_s/sigma)x + (b_s - a_s*mu/sigma)
    a = a_s / x_std
    b = b_s - a_s * x_mean / x_std

    return a, b

run = st.button("▶ 회귀분석 실행", type="primary", use_container_width=True)

if run:
    try:
        # -----------------------------
        # 선형회귀
        # -----------------------------
        a, b = linear_regression_gradient_descent(x, y)
        linear_pred = a * x + b

        # -----------------------------
        # 비선형회귀
        # -----------------------------
        # 초기값은 2차/선형 근사를 바탕으로 설정
        p0 = np.polyfit(x, y, 2)
        params, _ = curve_fit(
            quadratic,
            x,
            y,
            p0=p0,
            maxfev=20000
        )
        a2, b2, c2 = params
        nonlinear_pred = quadratic(x, a2, b2, c2)

        # -----------------------------
        # MSE
        # -----------------------------
        linear_mse = float(np.mean((y - linear_pred) ** 2))
        nonlinear_mse = float(np.mean((y - nonlinear_pred) ** 2))

        # -----------------------------
        # 결과 출력
        # -----------------------------
        st.subheader("2. 회귀분석 결과")

        r1, r2 = st.columns(2)

        with r1:
            st.markdown("### 선형회귀")
            st.latex(r"y = ax + b")
            st.code(f"y = {a:.6f}x + {b:.6f}")
            st.metric("선형회귀 MSE", f"{linear_mse:.6f}")

        with r2:
            st.markdown("### 비선형회귀")
            st.latex(r"y = ax^2 + bx + c")
            st.code(f"y = {a2:.8f}x² + {b2:.8f}x + {c2:.8f}")
            st.metric("비선형회귀 MSE", f"{nonlinear_mse:.6f}")

        # -----------------------------
        # 그래프
        # -----------------------------
        st.subheader("3. 결과 그래프")

        x_grid = np.linspace(np.min(x), np.max(x), 300)
        linear_grid = a * x_grid + b
        nonlinear_grid = quadratic(x_grid, a2, b2, c2)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(x, y, s=55, label="Data")
        ax.plot(x_grid, linear_grid, linewidth=2, label="Linear Regression")
        ax.plot(x_grid, nonlinear_grid, linewidth=2, label="Nonlinear Regression")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Linear vs Nonlinear Regression")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # -----------------------------
        # 해석
        # -----------------------------
        st.subheader("4. 자동 해석")

        if linear_mse < nonlinear_mse:
            better = "선형회귀"
        elif nonlinear_mse < linear_mse:
            better = "비선형회귀"
        else:
            better = "두 모델이 동일"

        st.write(
            f"- 선택한 데이터에서 선형회귀 MSE는 **{linear_mse:.6f}**, "
            f"비선형회귀 MSE는 **{nonlinear_mse:.6f}**입니다."
        )
        st.write(
            f"- 현재 데이터에서는 **{better}**의 MSE가 더 작게 나타났습니다."
        )
        st.write(
            "- MSE는 실제값과 예측값의 차이를 제곱하여 평균한 값이므로, "
            "작을수록 해당 모델이 현재 데이터에 더 가깝게 적합되었다고 볼 수 있습니다."
        )
        st.write(
            "- 단, MSE가 작다는 사실만으로 일반적인 예측 성능이 항상 더 좋다고 단정할 수는 없으며, "
            "데이터의 특성과 분석 목적을 함께 고려해야 합니다."
        )

    except Exception as e:
        st.error(f"회귀분석 중 오류가 발생했습니다: {e}")
