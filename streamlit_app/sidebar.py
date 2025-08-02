import streamlit as st

def menu():
    st.sidebar.title("船舶优化软件")
    return st.sidebar.radio("导航菜单", ["船型数据生成", "船型性能预测", "船型优化"])
