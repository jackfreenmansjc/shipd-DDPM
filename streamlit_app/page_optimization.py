import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core import model_trainer

def render():
    st.title("船型优化")

    if st.button("训练神经网络模型"):
        model_trainer.train_model()
        st.success("训练完成")

    if st.button("使用DDPM生成优化船型"):
        new_ships = model_trainer.optimize_with_ddpm()
        st.write("已生成新船型参数")
        st.dataframe(new_ships)
