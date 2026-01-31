import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import time
import akshare as ak
from supabase import create_client, Client
from strategy import SYSTEM_PROMPT, GLOBAL_MARKET_POOL

# ==========================================
# 0. 云端数据库连接
# ==========================================
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except: return None

def load_user_portfolio(username):
    supabase = init_supabase()
    if not supabase: return [] 
    try:
        response = supabase.table("user_portfolios").select("portfolio_data").eq("username", username).execute()
        return response.data[0]['portfolio_data'] if response.data else []
    except: return []

def save_user_portfolio(username, portfolio):
    supabase = init_supabase()
    if not supabase: return
    try:
        existing = supabase.table("user_portfolios").select("*").eq("username", username).execute()
        if existing.data:
            supabase.table("user_portfolios").update({"portfolio_data": portfolio}).eq("username", username).execute()
        else:
            supabase.table("user_portfolios").insert({"username": username, "portfolio_data": portfolio}).execute()
    except: pass

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="AI 智能量化投顾 (Pro)", layout="wide")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; color: #333333; }
    h1 { font-size: 41px !important; font-weight: 800 !important; color: #2D3436; }
    div.stButton > button:first-child {
        background-color: #6C5CE7 !important; color: white !important; border-radius: 50px !important; border: none !important;
        padding: 8px 20px !important; box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
    }
    div.stButton > button:first-child:hover { background-color: #5541c9 !important; }
    div[data-testid="stExpander"] { background-color: #FFFFFF !important; border-radius: 20px !important; border: 1px solid #F0F0F0 !important; }
    section[data-testid="stSidebar"] { background-color: #F8F9FA; padding-top: 20px; }
    .stProgress > div > div > div > div { background-color: #6C5CE7; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 动态数据源 (核心逻辑升级)
# ==========================================
def get_dynamic_market_pool(market_type="US", strategy="TURNOVER"):
    """
    根据不同战法获取实时股票池
    """
    pool = []
    
    # === A股策略 (实时动态) ===
    if market_type == "CN":
        try:
            # 获取实时行情
            df_cn = ak.stock_zh_a_spot_em()
            # 过滤掉非主板/创业板 (保留 0, 3, 6 开头)
            df_cn = df_cn[df_cn['代码'].astype(str).str.match(r'^[036]')]
            
            target_df = pd.DataFrame()

            if strategy == "TURNOVER": 
                # 🏛️ 资金战场: 成交额前 50
                target_df = df_cn.sort_values(by="成交额", ascending=False).head(50)
            
            elif strategy == "TURNOVER_RATE": 
                # 🎢 稳健活跃 (原情绪妖股): 
                # 1. 必须收红 (涨幅 > 0)
                active_df = df_cn[df_cn['涨跌幅'] > 0]
                
                # 2. 【核心修改】：换手率区间控制在 4% ~ 10%
                # 这代表股票活跃但未过热，属于健康的主升浪区间
                mask = (active_df['换手率'] >= 4) & (active_df['换手率'] <= 10)
                filtered_df = active_df[mask]
                
                # 在这个区间里，依然按换手率从高到低排序，取前 50
                target_df = filtered_df.sort_values(by="换手率", ascending=False).head(50)
                
            elif strategy == "FLOW": 
                # 💰 主力扫货: 主力净流入前 50
                target_df = df_cn.sort_values(by="主力净流入", ascending=False).head(50)

            for _, row in target_df.iterrows():
                code = row['代码']
                if code.startswith('6') or code.startswith('9'): suffix = ".SS"
                elif code.startswith('0') or code.startswith('3'): suffix = ".SZ"
                else: suffix = ".BJ"
                pool.append(code + suffix)
            return pool
        except Exception as e:
            st.error(f"A股数据源连接失败: {e}")
            return []

    # === 港股策略 ===
    elif market_type == "HK": 
        try:
            df_hk = ak.stock_hk_spot_em()
            top_30 = df_hk.sort_values(by="成交额", ascending=False).head(30)
            for _, row in top_30.iterrows():
                pool.append(str(row['代码']) + ".HK")
            return pool
        except: return []

    # === 美股策略 ===
    else: 
        base_pool = GLOBAL_MARKET_POOL
        if strategy == "TURNOVER_RATE":
            meme_stocks = ["GME", "AMC", "DJT", "MARA", "COIN", "PLTR", "SOFI", "OPEN", "MSTR"]
            return list(set(base_pool + meme_stocks))
        return base_pool

# ==========================================
# 3. 工具函数
# ==========================================
def smart_fix_ticker(ticker_input):
    t = ticker_input.strip().upper()
    if "." in t: return t
    if t.isdigit():
        if len(t) == 4 or len(t) == 5: return f"{t}.HK"
        if len(t) == 6:
            if t.startswith("6") or t.startswith("9"): return f"{t}.SS"
            if t.startswith("0") or t.startswith("3"): return f"{t}.SZ"
    return t

def get_stock_name(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info.get('shortName') or t.info.get('longName') or ticker
    except: return ticker

def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except: return False

def fetch_news_yahoo(ticker, limit=3):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news: return "暂无直接关联新闻"
        summary = ""
        for i, item in enumerate(news):
            if i >= limit: break
            summary += f"- [{item.get('publisher')}] {item.get('title')}\n"
        return summary
    except: return "新闻接口繁忙"

def get_data_and_indicators(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty: return None, "无数据"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['MA60'] = ta.sma(df['Close'], length=60)
        df['J'] = ta.kdj(df['High'], df['Low'], df['Close'])['J_9_3']
        df['Vol_MA5'] = ta.sma(df['Volume'], length=5)
        return df, None
    except Exception as e: return None, str(e)

def market_scanner_filter(ticker_list, status_container=None):
    candidates = []
    total = len(ticker_list)
    if status_container:
        msg_placeholder = status_container.empty()
        progress_bar = status_container.progress(0)
    
    for i, ticker in enumerate(ticker_list):
        if status_container:
            msg_placeholder.caption(f"🔍 [{i+1}/{total}] 扫描中: {ticker}...")
            progress_bar.progress((i + 1) / total)
        
        df, _ = get_data_and_indicators(ticker)
        if df is not None:
            latest = df.iloc[-1]
            try:
                # 狙击逻辑 (J值放宽到35，寻找热门股回调)
                cond1 = latest['Close'] > latest['MA60'] if pd.notna(latest['MA60']) else True
                cond3 = latest['Volume'] < latest['Vol_MA5'] # 缩量
                cond2 = latest['J'] < 35 

                if cond1 and cond2 and cond3:
                    candidates.append({'ticker': ticker, 'price': latest['Close'], 'j_value': latest['J'], 'df': df})
            except: continue
            
    if status_container:
        progress_bar.empty()
        msg_placeholder.write(f"✅ 扫描完成，从 {total} 只热门股中锁定 {len(candidates)} 个回调机会。")
        
    candidates.sort(key=lambda x: x['j_value'])
    return candidates[:5]

def analyze_with_gemini(ticker, df, news, holdings_info=None):
    latest = df.iloc[-1]
    ma60_val = f"{latest['MA60']:.2f}" if 'MA60' in latest and pd.notna(latest['MA60']) else "N/A"
    tech_data = f"现价:{latest['Close']:.2f}, Vol:{latest['Volume']}(5日均:{latest['Vol_MA5']:.0f}), MA60:{ma60_val}, J:{latest['J']:.2f}"
    
    task_type = "【持仓体检】" if holdings_info else "【狙击分析 (热门股回调)】"
    user_ctx = f"持仓成本:{holdings_info['cost']}" if holdings_info else ""
    
    prompt = f"{SYSTEM_PROMPT}\n任务:{task_type}\n数据:{tech_data}\n{user_ctx}\n新闻:{news}"
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model.generate_content(prompt).text

# ==========================================
# 4. 主程序
# ==========================================
def main():
    if 'current_user' not in st.session_state:
        st.title("🔐 AI 投顾 - 登录")
        with st.form("login"):
            u = st.text_input("用户名 (自动创建/读取)")
            if st.form_submit_button("进入"):
                if u:
                    st.session_state.current_user = u.strip()
                    with st.spinner("同步数据中..."):
                        st.session_state.portfolio = load_user_portfolio(u.strip())
                    st.rerun()
        return

    username = st.session_state.current_user
    if 'portfolio' not in st.session_state: st.session_state.portfolio = load_user_portfolio(username)
    auto_key = st.secrets.get("GEMINI_API_KEY", None)

    with st.sidebar:
        st.header(f"👤 {username}")
        if st.button("退出"): del st.session_state.current_user; st.rerun()
        st.markdown("---")
        if auto_key: st.success("✅ Gemini 已连接")
        else: 
            auto_key = st.text_input("Gemini API Key", type="password")
            if auto_key: configure_gemini(auto_key)

        st.markdown("---")
        with st.form("add"):
            st.caption("📝 美股(NVDA) | A股(600519) | 港股(0700)")
            c1, c2 = st.columns([0.6,0.4])
            t = c1.text_input("代码", placeholder="AAPL")
            c = c2.number_input("成本", min_value=0.0)
            if st.form_submit_button("➕"):
                if t:
                    ft = smart_fix_ticker(t)
                    name = get_stock_name(ft)
                    st.session_state.portfolio.append({'ticker': ft, 'name': name, 'cost': c})
                    save_user_portfolio(username, st.session_state.portfolio)
                    st.success(f"已存 {name}")
                    time.sleep(0.5); st.rerun()

        st.markdown("###### 📦 云端持仓")
        for i, item in enumerate(st.session_state.portfolio):
            c1, c2 = st.columns([0.7, 0.3])
            c1.markdown(f"**{item.get('name')}**\n`{item['ticker']}`")
            if c2.button("删", key=f"d{i}"):
                st.session_state.portfolio.pop(i)
                save_user_portfolio(username, st.session_state.portfolio)
                st.rerun()
            st.markdown("---")

    if not auto_key: st.warning("需配置 API Key"); return
    configure_gemini(auto_key)

    st.title("AI 智能量化投顾")
    tab1, tab2 = st.tabs(["🕵️‍♂️ 持仓审计", "🌊 动态市场猎手"])

    with tab1:
        if st.button("🚀 分析持仓"):
            if not st.session_state.portfolio: st.warning("无持仓")
            else:
                s_head = st.empty(); prog = st.progress(0)
                for i, item in enumerate(st.session_state.portfolio):
                    s_head.markdown(f"### 🔄 分析: {item.get('name')}...")
                    df, _ = get_data_and_indicators(item['ticker'])
                    if df is not None:
                        res = analyze_with_gemini(item['ticker'], df, fetch_news_yahoo(item['ticker']), item)
                        with st.expander(f"📄 {item.get('name')} 报告", expanded=True): st.markdown(res, unsafe_allow_html=True)
                    else: st.error(f"❌ {item['ticker']} 数据失败")
                    prog.progress((i+1)/len(st.session_state.portfolio))
                prog.empty(); s_head.success("✅ 完成")

    with tab2:
        st.markdown("#### 🌊 全球资金流向狙击 (动态数据)")
        
        c1, c2 = st.columns(2)
        with c1:
            market_choice = st.selectbox("1. 选择市场", ["🇨🇳 A股", "🇭🇰 港股", "🇺🇸 美股"])
        with c2:
            strategy_choice = st.selectbox("2. 选股战法", 
                                           ["🏛️ 资金战场 (成交额 Top)", 
                                            "🎢 稳健活跃 (换手率 4-10%)", 
                                            "💰 主力扫货 (净流入 Top)"])
        
        # 映射
        strat_map = {
            "🏛️ 资金战场 (成交额 Top)": "TURNOVER",
            "🎢 稳健活跃 (换手率 4-10%)": "TURNOVER_RATE",
            "💰 主力扫货 (净流入 Top)": "FLOW"
        }
        
        if st.button("🌊 启动动态扫描"):
            m_code = "US"
            if "A股" in market_choice: m_code = "CN"
            elif "港股" in market_choice: m_code = "HK"
            s_code = strat_map[strategy_choice]

            with st.spinner(f"正在抓取 {market_choice} 实时榜单..."):
                target_pool = get_dynamic_market_pool(m_code, s_code)
            
            if not target_pool:
                st.error("数据源连接超时或市场休市。")
            else:
                st.success(f"已锁定 {len(target_pool)} 只符合标准的热门标的，开始量化筛选...")
                
                with st.status("🎯 狙击扫描中...", expanded=True) as s:
                    top = market_scanner_filter(target_pool, s)
                    if not top:
                        s.update(label="⚠️ 扫描完成，无回调机会", state="error", expanded=True)
                        st.warning("🔥 提示：当前筛选池中未发现符合'缩量回调+J值低'的标的。市场可能过于强势或过于低迷。")
                    else:
                        s.write(f"🧠 AI 深度研判 Top {len(top)}...")
                        cols = st.columns(2)
                        ai_msg = s.empty(); ai_prog = s.progress(0)
                        for i, item in enumerate(top):
                            ai_msg.write(f"研判: {item['ticker']}...")
                            with cols[i%2]:
                                st.markdown(f"### 🎯 {item['ticker']}")
                                with st.expander("AI 评级", expanded=True):
                                    st.markdown(analyze_with_gemini(item['ticker'], item['df'], fetch_news_yahoo(item['ticker'])), unsafe_allow_html=True)
                            ai_prog.progress((i+1)/len(top))
                        ai_msg.empty(); ai_prog.empty()
                        s.update(label="✅ 任务完成", state="complete", expanded=False)

if __name__ == "__main__":
    main()
