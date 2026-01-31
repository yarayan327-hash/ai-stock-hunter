import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from duckduckgo_search import DDGS
import google.generativeai as genai
import time
from supabase import create_client, Client
from strategy import SYSTEM_PROMPT, GLOBAL_MARKET_POOL

# ==========================================
# 0. 云端数据库连接 (Supabase)
# ==========================================
@st.cache_resource
def init_supabase():
    # 尝试从 secrets 读取，如果本地没有 secrets (比如第一次运行)，则返回 None 避免直接报错闪退
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except:
        return None

def load_user_portfolio(username):
    supabase = init_supabase()
    if not supabase: return [] # 没配置数据库时返回空
    try:
        response = supabase.table("user_portfolios").select("portfolio_data").eq("username", username).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]['portfolio_data']
        else: return []
    except Exception as e:
        return []

def save_user_portfolio(username, portfolio):
    supabase = init_supabase()
    if not supabase: return
    try:
        # 简单逻辑：先查后改/增
        existing = supabase.table("user_portfolios").select("*").eq("username", username).execute()
        if existing.data:
            supabase.table("user_portfolios").update({"portfolio_data": portfolio}).eq("username", username).execute()
        else:
            supabase.table("user_portfolios").insert({"username": username, "portfolio_data": portfolio}).execute()
    except Exception as e:
        st.error(f"保存失败: {e}")

# ==========================================
# 1. 页面配置与 CSS
# ==========================================
st.set_page_config(page_title="AI 智能量化投顾", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; color: #333333; }
    h1 { font-size: 41px !important; font-weight: 800 !important; color: #2D3436; }
    /* 按钮样式 */
    div.stButton > button:first-child {
        background-color: #6C5CE7 !important; color: white !important; border-radius: 50px !important; border: none !important;
        padding: 8px 20px !important; box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
    }
    div.stButton > button:first-child:hover { background-color: #5541c9 !important; }
    /* 卡片与侧边栏 */
    div[data-testid="stExpander"] { background-color: #FFFFFF !important; border-radius: 20px !important; border: 1px solid #F0F0F0 !important; }
    section[data-testid="stSidebar"] { background-color: #F8F9FA; padding-top: 20px; }
    /* 进度条颜色 */
    .stProgress > div > div > div > div { background-color: #6C5CE7; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心工具函数
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

def fetch_news(ticker, limit=3):
    try:
        clean_ticker = ticker.split(".")[0]
        results = DDGS().text(f"{clean_ticker} stock news", max_results=limit)
        return "".join([f"- [{r['title']}] {r['body']}\n" for r in results])
    except: return "暂无新闻"

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

# === 关键修改：UI 优化版筛选器 ===
def market_scanner_filter(ticker_list, status_container=None):
    candidates = []
    total = len(ticker_list)
    
    # 1. 创建占位符 (而不是直接 write)
    if status_container:
        msg_placeholder = status_container.empty() # 这是一个动态文本框
        progress_bar = status_container.progress(0)
    
    for i, ticker in enumerate(ticker_list):
        if status_container:
            # 2. 每次循环只更新这个文本框，不会产生新行
            msg_placeholder.caption(f"🔍 [{i+1}/{total}] 正在扫描: {ticker}...")
            progress_bar.progress((i + 1) / total)
        
        df, _ = get_data_and_indicators(ticker)
        if df is not None:
            latest = df.iloc[-1]
            try:
                # 狙击逻辑
                cond1 = latest['Close'] > latest['MA60'] if pd.notna(latest['MA60']) else True
                cond2 = latest['J'] < 25
                cond3 = latest['Volume'] < latest['Vol_MA5']
                if cond1 and cond2 and cond3:
                    candidates.append({'ticker': ticker, 'price': latest['Close'], 'j_value': latest['J'], 'df': df})
            except: continue
            
    # 3. 扫描结束，清空进度条和文字，只留一条总结
    if status_container:
        progress_bar.empty()
        msg_placeholder.write(f"✅ 扫描覆盖 {total} 只核心资产，初步锁定 {len(candidates)} 个目标。")
        
    candidates.sort(key=lambda x: x['j_value'])
    return candidates[:5]

def analyze_with_gemini(ticker, df, news, holdings_info=None):
    latest = df.iloc[-1]
    ma60_val = f"{latest['MA60']:.2f}" if 'MA60' in latest and pd.notna(latest['MA60']) else "N/A"
    tech_data = f"现价:{latest['Close']:.2f}, Vol:{latest['Volume']}(5日均:{latest['Vol_MA5']:.0f}), MA60:{ma60_val}, J:{latest['J']:.2f}"
    
    task_type = "【持仓体检】" if holdings_info else "【狙击分析】"
    user_ctx = f"持仓成本:{holdings_info['cost']}" if holdings_info else ""
    
    prompt = f"{SYSTEM_PROMPT}\n任务:{task_type}\n数据:{tech_data}\n{user_ctx}\n新闻:{news}"
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model.generate_content(prompt).text

# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- 登录逻辑 ---
    if 'current_user' not in st.session_state:
        st.title("🔐 AI 投顾 - 登录")
        with st.form("login"):
            u = st.text_input("用户名 (自动创建/读取)")
            if st.form_submit_button("进入"):
                if u:
                    st.session_state.current_user = u.strip()
                    # 尝试从云端加载
                    with st.spinner("正在同步云端数据..."):
                        data = load_user_portfolio(st.session_state.current_user)
                        st.session_state.portfolio = data
                    st.rerun()
        return

    # --- 已登录界面 ---
    username = st.session_state.current_user
    auto_key = st.secrets.get("GEMINI_API_KEY", None)

    with st.sidebar:
        st.header(f"👤 {username}")
        if st.button("退出"):
            del st.session_state.current_user
            st.rerun()
        st.markdown("---")
        
        if auto_key: st.success("✅ Gemini 已连接")
        else: 
            auto_key = st.text_input("Gemini API Key", type="password")
            if auto_key: configure_gemini(auto_key)

        st.markdown("---")
        # 添加持仓
        with st.form("add"):
            c1, c2 = st.columns([0.6,0.4])
            t = c1.text_input("代码", placeholder="600519")
            c = c2.number_input("成本", min_value=0.0)
            if st.form_submit_button("➕"):
                if t:
                    ft = smart_fix_ticker(t)
                    name = get_stock_name(ft)
                    st.session_state.portfolio.append({'ticker': ft, 'name': name, 'cost': c})
                    save_user_portfolio(username, st.session_state.portfolio) # 保存云端
                    st.success(f"已存 {name}")
                    time.sleep(0.5)
                    st.rerun()

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
    tab1, tab2 = st.tabs(["🕵️‍♂️ 持仓审计", "🎯 市场猎手"])

    # Tab 1: 持仓
    with tab1:
        if st.button("🚀 分析持仓"):
            if not st.session_state.portfolio: st.warning("请先添加持仓")
            else:
                with st.status("🚀 正在审计持仓...", expanded=True) as s:
                    # 使用占位符，避免刷屏
                    msg = s.empty()
                    prog = s.progress(0)
                    total = len(st.session_state.portfolio)
                    
                    for i, item in enumerate(st.session_state.portfolio):
                        msg.write(f"🔄 正在分析: {item.get('name')}...")
                        prog.progress((i)/total)
                        
                        df, _ = get_data_and_indicators(item['ticker'])
                        if df is not None:
                            res = analyze_with_gemini(item['ticker'], df, fetch_news(item['ticker']), item)
                            with st.expander(f"📄 {item.get('name')} 报告", expanded=True): st.markdown(res, unsafe_allow_html=True)
                        prog.progress((i+1)/total)
                    
                    msg.empty()
                    prog.empty()
                    s.update(label="✅ 所有持仓审计完成！", state="complete", expanded=False)

    # Tab 2: 猎手
    with tab2:
        if st.button("🎯 启动狙击扫描"):
            # 动效优化：状态框会自动收起
            with st.status("🎯 全市场扫描初始化...", expanded=True) as s:
                # 1. 扫描阶段 (内部使用了占位符，不会刷屏)
                top = market_scanner_filter(GLOBAL_MARKET_POOL, s)
                
                if not top: 
                    s.update(label="⚠️ 扫描完成，无缩量超卖机会", state="error", expanded=True)
                    st.warning("当前市场无合适标的。")
                else:
                    s.write(f"🧠 AI 正在深度研判 {len(top)} 只标的...")
                    cols = st.columns(2)
                    
                    # 2. AI 分析阶段
                    ai_msg = s.empty()
                    ai_prog = s.progress(0)
                    
                    for i, item in enumerate(top):
                        ai_msg.write(f"正在研判: {item['ticker']}...")
                        ai_prog.progress(i / len(top))
                        
                        with cols[i%2]:
                            st.markdown(f"### 🎯 {item['ticker']}")
                            with st.expander("查看狙击评级", expanded=True):
                                st.markdown(analyze_with_gemini(item['ticker'], item['df'], fetch_news(item['ticker'])), unsafe_allow_html=True)
                        
                        ai_prog.progress((i+1)/len(top))
                    
                    ai_msg.empty() # 清除 "正在研判..."
                    ai_prog.empty() # 清除进度条
                    s.update(label="✅ 狙击任务执行完毕！", state="complete", expanded=False) # 自动收起

if __name__ == "__main__":
    main()