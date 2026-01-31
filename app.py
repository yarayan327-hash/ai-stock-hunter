import streamlit as st
import pandas as pd
import pandas_ta as ta
import baostock as bs
import yfinance as yf
import requests
import json
import time
import random
from supabase import create_client
from datetime import datetime, timedelta

# ==========================================
# 0. 核心配置 & 提示词
# ==========================================
SYSTEM_PROMPT = """
你是一个严谨的量化基金经理，擅长“趋势回调策略”。
该股票已经通过了量化初筛（趋势向上 + 极度缩量回调 + 活跃度适中）。
请基于传入的技术数据和资金流向，进行最后的“人工复核”。

⚡ **格式要求 (关键信息背景色高亮)**:
- 关键利好：:green-background[文字]
- 关键风险：:red-background[文字]
- 关键点位：:orange-background[文字]
- 核心评分：:blue-background[文字]

🔥 **分析重点**:
1. **支撑有效性**：当前回调是否在 MA60 或 前期平台 获得支撑？
2. **量能健康度**：下跌是否缩量？主力是否有出逃迹象？

请输出：
### 1. 🎯 投资结论 (评分 0-100)
### 2. 🔍 逻辑拆解 (量价/形态/资金)
### 3. 💡 交易计划 (建议入场位/止损位/第一目标位)
"""

st.set_page_config(page_title="趋势狙击", layout="wide")

@st.cache_resource
def init_supabase():
    try: return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except: return None

def load_user_portfolio(username):
    sb = init_supabase()
    if not sb: return []
    try:
        res = sb.table("user_portfolios").select("portfolio_data").eq("username", username).execute()
        return res.data[0]['portfolio_data'] if res.data else []
    except: return []

def save_user_portfolio(username, portfolio):
    sb = init_supabase()
    if not sb: return
    try:
        existing = sb.table("user_portfolios").select("*").eq("username", username).execute()
        if existing.data:
            sb.table("user_portfolios").update({"portfolio_data": portfolio}).eq("username", username).execute()
        else:
            sb.table("user_portfolios").insert({"username": username, "portfolio_data": portfolio}).execute()
    except: pass

# ==========================================
# 1. 数据清洗 (🔴 修复点)
# ==========================================
def process_data(df):
    if df is None or df.empty: return None, "无数据"
    try:
        # 🔴 关键修复：把 'TurnoverRate' 加入到强制转数字的列表里
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Turnover', 'TurnoverRate']
        
        for c in numeric_cols:
            if c in df.columns:
                # errors='coerce' 会把无法转换的字符变成 NaN (空值)
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # 填充空值，防止计算报错
        df = df.fillna(0)
        
        if 'Turnover' not in df.columns: df['Turnover'] = 0.0
        if 'TurnoverRate' not in df.columns: df['TurnoverRate'] = 0.0
            
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['MA60'] = ta.sma(df['Close'], length=60) 
        kdj = ta.kdj(df['High'], df['Low'], df['Close'])
        df['K'] = kdj['K_9_3']
        df['D'] = kdj['D_9_3']
        df['J'] = kdj['J_9_3']
        df['Vol_MA5'] = ta.sma(df['Volume'], length=5)
        
        return df, None
    except Exception as e: return None, f"清洗失败: {str(e)}"

# ==========================================
# 2. 数据获取
# ==========================================
def get_cn_data_baostock(symbol):
    try:
        code = symbol
        if ".SS" in symbol: code = "sh." + symbol.replace(".SS", "")
        if ".SZ" in symbol: code = "sz." + symbol.replace(".SZ", "")
        if symbol.isdigit():
            code = "sh." + symbol if symbol.startswith("6") else "sz." + symbol

        bs.login()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
        
        rs = bs.query_history_k_data_plus(code,
            "date,open,high,low,close,volume,amount,turn", 
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3")
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
        
        if not data_list: return None, "BaoStock无返回"
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={
            'date':'Date', 'open':'Open', 'high':'High', 
            'low':'Low', 'close':'Close', 'volume':'Volume', 
            'amount':'Turnover', 'turn': 'TurnoverRate'
        })
        df.set_index('Date', inplace=True)
        return process_data(df)
    except Exception as e: return None, f"BS Error: {e}"

def get_hk_us_data_yf(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        if df.empty: return None, "Yahoo未返回数据"
        df['Turnover'] = df['Close'] * df['Volume']
        # 美股/港股 Yahoo 接口不直接给换手率，设为 0 以免报错
        df['TurnoverRate'] = 0.0 
        
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = df.index.tz_localize(None) 
        df.index.name = 'Date'
        return process_data(df)
    except Exception as e: return None, f"YF Error: {e}"

def get_stock_data(ticker):
    ticker = ticker.upper().strip()
    if ticker.startswith("SH.") or ticker.startswith("SZ.") or ticker.endswith(".SS") or ticker.endswith(".SZ") or (ticker.isdigit() and len(ticker)==6):
        return get_cn_data_baostock(ticker)
    else:
        return get_hk_us_data_yf(ticker)

# ==========================================
# 3. 动态选股池
# ==========================================
@st.cache_data(ttl=3600*12)
def get_market_pool_dynamic(market="CN"):
    pool = []
    if market == "CN":
        try:
            bs.login()
            rs_300 = bs.query_hs300_stocks()
            while (rs_300.error_code == '0') & rs_300.next():
                pool.append(rs_300.get_row_data()[1])
            bs.logout()
            random.shuffle(pool)
            return pool[:60] # 随机抽60只扫描，防超时
        except: return ["sh.600519", "sz.300750"]
    elif market == "US":
        return ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "AVGO", "COST", "NFLX", "AMD", "PDD", "BABA"]
    elif market == "HK":
        return ["00700.HK", "03690.HK", "01810.HK", "09988.HK", "00981.HK", "02015.HK", "01024.HK", "00020.HK"]
    return []

# ==========================================
# 4. 全能 Gemini 分析
# ==========================================
def call_gemini_rest(prompt, api_key):
    models_to_try = [
        "gemini-1.5-flash",       
        "gemini-1.5-pro",         
        "gemini-2.0-flash",       
        "gemini-2.0-flash-lite",  
        "gemini-1.5-flash-latest" 
    ]
    
    last_error = ""
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": f"你是量化专家。\n{prompt}"}]}]}
        
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                try:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    return f"✨ **Gemini 分析** (Model: {model})\n\n{text}"
                except: continue
            else:
                time.sleep(0.3)
                continue
        except: continue

    return f"❌ 分析失败，Google API 忙碌。"

def analyze_stock_gemini(ticker, df, news="", holdings=None):
    latest = df.iloc[-1]
    vol_display = f"{latest['Volume']/10000:.1f}万" if latest['Volume'] > 10000 else f"{latest['Volume']:.0f}"
    
    # 换手率显示
    turn_display = "N/A"
    if latest['TurnoverRate'] > 0:
        turn_display = f"{latest['TurnoverRate']:.2f}%"

    trend = "📈 趋势向上" if latest['Close'] > latest['MA60'] else "📉 趋势承压"
    
    tech = f"""
    标的: {ticker}
    现价: {latest['Close']:.2f}
    MA60: {latest['MA60']:.2f} [{trend}]
    J值: {latest['J']:.2f}
    换手率: {turn_display}
    缩量: {'✅ 是' if latest['Volume'] < latest['Vol_MA5'] else '❌ 否'}
    """
    
    task = "【持仓诊断】" if holdings else "【机会挖掘】"
    cost = f"成本: {holdings['cost']}" if holdings else ""
    prompt = f"{SYSTEM_PROMPT}\n任务:{task}\n{tech}\n{cost}\n{news}"
    
    return call_gemini_rest(prompt, st.secrets["GEMINI_API_KEY"])

# ==========================================
# 5. 主界面
# ==========================================
def main():
    if 'current_user' not in st.session_state:
        st.title("🏹 趋势狙击系统")
        u = st.text_input("用户名", placeholder="任意字符登录")
        if st.button("登录") and u:
            st.session_state.current_user = u
            st.session_state.portfolio = load_user_portfolio(u)
            st.rerun()
        return

    with st.sidebar:
        st.header(f"👤 {st.session_state.current_user}")
        if st.button("退出"): del st.session_state.current_user; st.rerun()
        st.divider()
        with st.form("add"):
            st.write("➕ **添加自选**")
            c1, c2 = st.columns(2)
            t = c1.text_input("代码", value="sh.600519")
            c = c2.number_input("成本", 0.0)
            if st.form_submit_button("加入"):
                st.session_state.portfolio.append({'ticker':t.upper(), 'name':t, 'cost':c})
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()
        
        st.divider()
        st.write("📦 **我的持仓**")
        for i, p in enumerate(st.session_state.portfolio):
            c1, c2 = st.columns([0.8, 0.2])
            c1.caption(f"{p['ticker']}") 
            if c2.button("✖", key=f"d{i}"): 
                st.session_state.portfolio.pop(i)
                save_user_portfolio(st.session_state.current_user, st.session_state.portfolio)
                st.rerun()

    st.title("🏹 趋势狙击系统 | 动态漏斗版")
    st.caption("动态数据源：BaoStock (A股成分股) / Yahoo (全球热门)")
    
    tab1, tab2 = st.tabs(["📊 持仓体检", "💎 黄金坑雷达"])
    
    with tab1:
        if st.button("开始体检", type="primary"):
            bar = st.progress(0)
            for i, p in enumerate(st.session_state.portfolio):
                with st.spinner(f"AI 正在分析 {p['ticker']} ..."):
                    df, err = get_stock_data(p['ticker'])
                    if df is not None:
                        res = analyze_stock_gemini(p['ticker'], df, "", p)
                        with st.expander(f"📌 {p['ticker']} 诊断报告", expanded=True): st.markdown(res)
                    else:
                        st.error(f"{p['ticker']} 获取失败: {err}")
                bar.progress((i+1)/len(st.session_state.portfolio))
    
    with tab2:
        c1, c2 = st.columns(2)
        m_type = c1.selectbox("选择市场", ["CN (A股-沪深300)", "US (美股-纳指热门)", "HK (港股-恒生科技)"])
        
        st.write("👇 **量化筛选漏斗参数**")
        m1, m2, m3 = st.columns(3)
        m1.metric("趋势支撑", "价格 > MA60", delta="生命线之上", delta_color="normal")
        m2.metric("超卖指标", "J值 < 30", delta="底部区域", delta_color="inverse")
        m3.metric("活跃区间", "换手率 4% - 10%", delta="资金活跃", delta_color="normal")
        st.markdown("---")
        
        if st.button("🚀 启动漏斗筛选", type="primary"):
            with st.spinner("Step 1: 正在从交易所获取最新成分股名单..."):
                pool = get_market_pool_dynamic(m_type.split()[0])
            
            status = st.status("正在执行漏斗过滤...", expanded=True)
            valid_stocks = []
            
            progress_bar = status.progress(0)
            total_scan = len(pool)
            
            for idx, t in enumerate(pool):
                df, _ = get_stock_data(t)
                
                # 🔴 修复比较逻辑：确保数据存在
                if df is not None and len(df) > 60:
                    latest = df.iloc[-1]
                    
                    # 1. 趋势
                    cond_trend = latest['Close'] > (latest['MA60'] * 0.98)
                    
                    # 2. J值
                    cond_j = latest['J'] < 30
                    
                    # 3. 换手率 (确保类型安全)
                    cond_turn = True
                    # 这里 latest['TurnoverRate'] 已经被 process_data 保证是 float 了
                    if latest['TurnoverRate'] > 0:
                        cond_turn = 4.0 <= latest['TurnoverRate'] <= 10.0
                    
                    if cond_trend and cond_j and cond_turn:
                        valid_stocks.append({'t':t, 'df':df, 'J':latest['J']})
                        turn_str = f"{latest['TurnoverRate']:.1f}%"
                        status.write(f"✅ 命中: {t} | J值: {latest['J']:.1f} | 换手: {turn_str}")
                
                progress_bar.progress((idx + 1) / total_scan)
            
            if not valid_stocks:
                status.update(label="扫描完成：未发现符合【趋势+超卖+活跃】的标的，建议空仓。", state="error")
            else:
                valid_stocks.sort(key=lambda x: x['J'])
                status.update(label=f"扫描完成！筛选出 {len(valid_stocks)} 只优质标的，AI 正在生成策略...", state="complete")
                
                for item in valid_stocks[:3]:
                    with st.spinner(f"Gemini 正在为 {item['t']} 撰写交易计划..."):
                        res = analyze_stock_gemini(item['t'], item['df'])
                        with st.expander(f"💎 {item['t']} - 机会分析 (J={item['J']:.1f})", expanded=True):
                            st.markdown(res)

if __name__ == "__main__":
    main()
