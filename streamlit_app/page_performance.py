import streamlit as st
import pandas as pd
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def render():
    st.title("船型性能分析")

    file = st.file_uploader("选择船型参数或STL文件", type=["csv", "stl"])
    
    if file and file.name.endswith(".csv"):
        df = pd.read_csv(file)
        st.write("参数内容：", df)

    elif file and file.name.endswith(".stl"):
        stl_data = mesh.Mesh.from_file(file)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_data.vectors))
        scale = stl_data.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        st.pyplot(fig)
