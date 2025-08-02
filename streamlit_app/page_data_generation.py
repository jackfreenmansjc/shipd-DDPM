import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from core import parameter_generator, hull_generator, physics_calculator, dataset_manager
from core.parameter_generator import SeedGenerator
from core.PCGen import pcgen
from core.STLGen import stlgen
def render():
    st.title("船型数据生成：复现SHIP_D")
    
    col1, col2,col3,col4 = st.columns(4)

    with col1:
        if st.button("生成船舶参数种子"):
            seeds = SeedGenerator().generate_seeds(20)
            st.success(f"生成了 {len(seeds)} 组参数")

    with col2:
        if st.button("计算船舶几何参数"):
            # 示例：你可以加载已有参数并调用函数，比如 Calc_GeometricProperties(seeds[0])
            st.info("正在计算船舶几何参数...")
            seeds = SeedGenerator().cal_geometric()
            # result = Calc_GeometricProperties(seeds[0])  # 例如计算第一个
            # st.write(result)
            st.success("计算完成！已保存。")
    with col3:
        if st.button("计算船舶静力学特性"):
            # 示例：你可以加载已有参数并调用函数，比如 Calc_GeometricProperties(seeds[0])
            st.info("正在计算静力学特性...")
            #seeds1 = SeedGenerator().cal_RW_para()
            seeds2 = SeedGenerator().cal_CW_para()
            seeds3 = SeedGenerator().cal_Maxbox_para()
            # result = Calc_GeometricProperties(seeds[0])  # 例如计算第一个
            # st.write(result)
            st.success("计算完成！已保存。")
    with col4:
        if st.button("生成点云和stl文件"):
            # 示例：你可以加载已有参数并调用函数，比如 Calc_GeometricProperties(seeds[0])
            st.info("正在生成点云和stl文件...")
            pcgen()
            stlgen()
            # result = Calc_GeometricProperties(seeds[0])  # 例如计算第一个
            # st.write(result)
            st.success("计算完成！已保存。")
    #uploaded = st.file_uploader("导入已有参数 CSV 文件", type="csv")
    #if uploaded:
        #st.write("已导入参数文件")

    #if st.button("生成船体点云与STL"):
        #hull_generator.generate_all()
        #st.success("完成船型几何生成")

    #if st.button("计算物理特性"):
        #physics_calculator.compute_all()
        #st.success("完成物理特性计算")

    #if st.button("保存数据集"):
        #dataset_manager.save()
        #st.success("数据已保存至 dataset/ 目录")
