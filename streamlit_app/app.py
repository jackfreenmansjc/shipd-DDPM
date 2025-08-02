import streamlit as st
import sidebar, page_data_generation, page_performance, page_optimization

def main():
    st.set_page_config(layout="wide", page_title="Ship Design Optimizer")
    page = sidebar.menu()

    if page == "船型数据生成":
        page_data_generation.render()
    elif page == "船型性能分析":
        page_performance.render()
    elif page == "船型优化":
        page_optimization.render()

if __name__ == "__main__":
    main()
